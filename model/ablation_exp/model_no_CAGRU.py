# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrequencyBranch(nn.Module):
    def __init__(self, in_freq=45, out_freq=64):
        super().__init__()
        # 在 in_freq=45（融合后频率点）上做局部卷积
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        # 1x1 卷积调维：in_freq → out_freq
        self.reduce = nn.Conv1d(in_freq, out_freq, kernel_size=1)

    def forward(self, x):
        B, T, C, Freq = x.shape

        x = x.reshape(B * T * C, 1, Freq)  # (B*T*C, 1, 45)

        x = self.local_conv(x)  # (B*T*C, 1, 45)

        x = self.reduce(x.permute(0, 2, 1))  # (B*T*C, 45, 1) → (B*T*C, 64, 1)

        x = x.permute(0, 2, 1)  # (B*T*C, 1, 64)

        x = x.reshape(B, T, C, 64)  # (B, T, C, 64)

        return x  # (B, T, C, 64)


class TCNBlock(nn.Module):
    """单个 TCN 残差块"""

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        self.total_padding = dilation * (kernel_size - 1)  # 全部用于左填充

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差路径调整
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        residual = self.residual(x)

        out = F.pad(x, (self.total_padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (self.total_padding, 0))
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class TimeBranch(nn.Module):
    def __init__(self, in_time=256, out_time=64):
        super().__init__()

        self.tcn = nn.Sequential(
            TCNBlock(in_channels=1, out_channels=32, kernel_size=3, dilation=1),
            TCNBlock(in_channels=32, out_channels=out_time, kernel_size=3, dilation=2),
        )

    def forward(self, x):
        B, T, C, t = x.shape

        x = x.reshape(B * T * C, t, 1)  # (B*T*C, 256, 1)

        x = x.permute(0, 2, 1)  # → (B*T*C, 1, 256)

        x = self.tcn(x)  # (B*T*C, 64, 256)

        x = x.mean(dim=-1)  # (B*T*C, 64)

        x = x.reshape(B, T, C, 64)  # (B, T, C, 64)

        return x  # (B, T, C, 64)


class MultiFreqModalFusion(nn.Module):
    def __init__(self, num_modalities=3, embed_dim=45):  
        super().__init__()
        self.num_modalities = num_modalities
        self.embed_dim = embed_dim

        self.attn = nn.MultiheadAttention(embed_dim, num_heads=5, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim) 

    def forward(self, multi_modal):  # multi_modal: (B, T, C, 135=45*3)

        B, T, C, D = multi_modal.shape
    
        x_flat = multi_modal.reshape(B * T * C, -1)  # (B*T*C, 135)
        x_seq = x_flat.view(B * T * C, self.num_modalities, self.embed_dim)  # (B*T*C, 3, 45)

        fused, attn_weights = self.attn(x_seq, x_seq, x_seq)  # fused: (B*T*C, 3, 45)

        fused_mean = fused.mean(dim=1)  # (B*T*C, 45)
        fused_norm = self.norm(fused_mean)  # (B*T*C, 45)

        fused_avg = fused_norm.reshape(B, T, C, self.embed_dim)  # (B, T, C, 45)

        return fused_avg, attn_weights  # attn_weights: (B*T*C, 3, 3) 

class ChannelAttGatedGRUCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, qk_dim):
        super(ChannelAttGatedGRUCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.q_proj_r = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_r = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_r = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.q_proj_z = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_z = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_z = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.q_proj_n = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.k_proj_n = nn.Linear(input_dim + hidden_dim, qk_dim)
        self.v_proj_n = nn.Linear(input_dim + hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化线性层
        for layer in [self.q_proj_r, self.k_proj_r,
                      self.q_proj_z, self.k_proj_z,
                      self.q_proj_n, self.k_proj_n]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        # 对 V 投影层使用正交初始化
        nn.init.orthogonal_(self.v_proj_r.weight)
        nn.init.orthogonal_(self.v_proj_z.weight)
        nn.init.orthogonal_(self.v_proj_n.weight)

        # 偏置项仍初始化为 0
        if self.v_proj_r.bias is not None:
            nn.init.zeros_(self.v_proj_r.bias)
        if self.v_proj_z.bias is not None:
            nn.init.zeros_(self.v_proj_z.bias)
        if self.v_proj_n.bias is not None:
            nn.init.zeros_(self.v_proj_n.bias)

    def self_attention(self, x, q_proj, k_proj, v_proj):

        Q = q_proj(x)  # [C, qv_size]
        K = k_proj(x)  # [C, qv_size]
        V = v_proj(x)  # [C, hidden_size]

        # 计算注意力权重: A = softmax(QK^T / sqrt(d_k))
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))  # [C, C]
        attn_weights = F.softmax(scores, dim=-1)  # [C, C]

        # 加权融合: Att = A @ V
        attn_output = torch.matmul(attn_weights, V)  # [C, hidden_dim]

        return attn_output, attn_weights

    def forward(self, x, h):
        """
        x: (B, C, input_dim)
        h: (B, C, hidden_dim)
        """
        # 重置门
        r, r_matrix = self.self_attention(torch.cat([x, h], dim=2), self.q_proj_r, self.k_proj_r, self.v_proj_r)  # [C, hidden_dim]

        # 更新门
        z, z_matrix = self.self_attention(torch.cat([x, h], dim=2), self.q_proj_z, self.k_proj_z, self.v_proj_z)  # [C, hidden_dim]

        reset = torch.sigmoid(r)  # [C, hidden_dim]
        update = torch.sigmoid(z)  # [C, hidden_dim]
        # 候选状态
        n, n_matrix = self.self_attention(torch.cat([x, h * reset], dim=2), self.q_proj_n, self.k_proj_n, self.v_proj_n)  # [C, hidden_dim]

        out_inputs = torch.tanh(n)  # [C, hidden_dim]

        new_state = (1 - update) * h + update * out_inputs  # [C, hidden_dim]

        new_state = self.norm(new_state)

        return new_state


class ChannelAttGatedGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, qk_dim, num_layers, batch_first=True):
        super(ChannelAttGatedGRU, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.batch_first = batch_first

        self.cells = nn.ModuleList([ChannelAttGatedGRUCell(input_dim, hidden_dim, qk_dim) for _ in range(num_layers)])

    def forward(self, x, h_prev=None):

        if self.batch_first:
            x = x.transpose(0, 1)  # (B, T, C, input_dim) -> (T, B, C, input_dim)

        seq_len, batch_size, C, _ = x.size()

        # 初始化隐藏状态
        if h_prev is None:
            h_prev = torch.zeros(self.num_layers, batch_size, C, self.hidden_dim,
                             device=x.device, dtype=x.dtype)

        layer_output_list = []
        cur_layer_output = x

        for layer in range(self.num_layers):
            h_layer = []
            h = h_prev[layer]
            for t in range(seq_len):
                h = self.cells[layer](cur_layer_output[t], h)
                h_layer.append(h)

            layer_output = torch.stack(h_layer, dim=0)  # (T, B, C, hidden_dim)
            cur_layer_output = layer_output
            layer_output_list.append(layer_output)

        h_final = layer_output_list[-1]  # 最后时刻隐藏状态

        if self.batch_first:
            output = h_final.transpose(0, 1)  # (B, T, C, hidden_dim)

        return output

# -------------------------------
# 模型主结构 (集成所有模块)
# -------------------------------

class MainModel(nn.Module):
    def __init__(self, Freq=135, Time=256, num_classes=2, config=None):
        super().__init__()
        self.config = config or {}  
        # 频域多模态融合 (embed_dim=Freq//3=45)
        self.multi_fusion = MultiFreqModalFusion(num_modalities=3, embed_dim=Freq // 3)

        # 时频特征提取
        self.freq_branch = FrequencyBranch(in_freq=Freq//3, out_freq=64)
        self.time_branch = TimeBranch(in_time=Time, out_time=64)

        #self.cagru = ChannelAttGatedGRU(input_dim=64 + 64, hidden_dim=64, qk_dim=64, num_layers=1, batch_first=True)

        # FC 分类
        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x, return_features=False, feature_level='final'):
        # x: (B, T=9, C=22, 391) [前Freq=135: 多模态STFT | 后Time=256]
        Freq = self.config.get('Freq', 135) if self.config else 135 
        Time = self.config.get('Time', 256) if self.config else 256
        multi_modal = x[:, :, :, :Freq]
        time_raw = x[:, :, :, Freq:]

        # 频域多模态融合
        fused_freq, attn_map = self.multi_fusion(multi_modal)

        freq_out = self.freq_branch(fused_freq)
        time_out = self.time_branch(time_raw)

        # 拼接节点特征
        x_feat = torch.cat([freq_out, time_out], dim=3)

        #x_tgcn = self.cagru(x_feat)  # (B, T=9, C=22, 64)

        # 时间注意力池化
        x_time_avg = x_feat.mean(dim=1).mean(dim=1)  # (B, 128)

        # 分类
        logits = self.classifier(x_time_avg) # (B, num_classes=2)

        if return_features:
            if feature_level == 'freq':
                return logits, freq_out.mean(dim=1).mean(dim=1)  # (B, F)
            elif feature_level == 'time':
                return logits, time_out.mean(dim=1).mean(dim=1)  # (B, T)
            elif feature_level == 'fusion':
                return logits, x_feat.mean(dim=1).mean(dim=1)  # (B, 128)
            elif feature_level == 'final':  # 默认
                return logits, x_time_avg  # (B, 64)
            else:
                raise ValueError("feature_level must be in ['freq', 'time', 'fusion', 'final']")
        else:
            return logits

