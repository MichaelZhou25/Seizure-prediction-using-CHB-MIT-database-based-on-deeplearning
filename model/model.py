import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm  # 更好的进度条
import warnings
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

# 忽略常见警告
warnings.filterwarnings("ignore")

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)

# -------------------------------
# 配置参数
# -------------------------------
PATIENT_ID = 23
DATA_DIR = Path("D:\\陈教授组\\mymodel\\data")
PREICTAL_PATH = DATA_DIR / "preictal" / f"preictal_fragments{PATIENT_ID:02d}.h5"
INTERICTAL_PATH = DATA_DIR / "interictal" / f"interictal_fragments{PATIENT_ID:02d}.h5"

# 信号参数
FS = 256  # 采样率 (Hz)
N_FFT = 256
HOP_LENGTH = 128
WIN_LENGTH = 256

# 设备选择（GPU 加速）
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"使用设备: {DEVICE}")



#一、数据预处理部分

# -------------------------------
# 滤波函数
# -------------------------------
def bandpass_filter(data, lowcut, highcut, fs, axis=-1, order=4):
    """带通滤波器（可选）"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=axis)

def notch_filter(data, freq, fs, Q=30, axis=-1):
    """带陷滤波器"""
    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data, axis=axis)

def remove_dc_component(data, axis=-1):
    """去除直流分量"""
    mean_val = np.mean(data, axis=axis, keepdims=True)
    return data - mean_val

def apply_all_filters(data, fs=FS):
    """
    应用滤波链：工频陷波 + 二次谐波陷波 + 去除 DC
    输入: (..., channels, timepoints)
    """
    filtered = notch_filter(data, 60.0, fs=fs)
    filtered = notch_filter(filtered, 120.0, fs=fs)
    filtered = remove_dc_component(filtered)
    return filtered


def generating_raw_data_matching_stft(data_array, n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE):
    if data_array.size == 0:
        return np.array([])

    # 转为 tensor 并移到设备
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)  # (N, C, T)

    return data_tensor.unfold(-1, n_fft, hop_length).permute(0, 1, 3, 2)


# -------------------------------
# STFT 变换函数（优化版，支持批量 + GPU）
# -------------------------------
def apply_stft_to_data(data_array, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH, fs=FS, device=DEVICE):
    """
    对三维数组应用 STFT，返回对数幅度谱
    输入: (n_samples, n_channels, n_timepoints)
    输出: (n_samples, n_channels, freq_bins, time_frames) 的 log-magnitude
    """
    if data_array.size == 0:
        return np.array([])

    n_samples, n_channels, n_timepoints = data_array.shape

    # 转为 tensor 并移到设备
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)  # (N, C, T)

    # 预定义窗函数（移到设备）
    window = torch.hann_window(win_length).to(device)

    # 批量处理所有通道和样本（利用广播）
    data_flat = data_tensor.view(-1, n_timepoints)  # (N*C, T)

    # 应用 STFT（返回复数）
    stft_complex = torch.stft(
        data_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,           # ✅ 时间对齐
        return_complex=True    # ✅ 复数输出
    )  # shape: (N*C, F, T_frames)

    # 恢复通道维度
    freq_bins = n_fft // 2 + 1
    time_frames = stft_complex.shape[-1]
    stft_complex = stft_complex.view(n_samples, n_channels, freq_bins, time_frames)

    # 删除 0Hz (DC 分量) —— 合理，尤其对 EEG
    stft_complex = stft_complex[:, :, 1:, 1:-1]  # (N, C, F-1, T')

    # 转为对数幅度谱：20 * log10(|Z| + ε)
    magnitude = stft_complex.abs()  # (N, C, F-1, T')
    log_magnitude_stft = 20 * torch.log10(magnitude + 1e-8)

    return log_magnitude_stft

# -------------------------------
# 主处理函数：逐片段处理并保存
# -------------------------------
import h5py
import numpy as np
from tqdm import tqdm

def process_and_save_fragments(input_path, data_type):
    """
    逐个加载 HDF5 片段，滤波 → STFT → log-magnitude
    并在 Batch 维度上拼接所有片段的结果

    Returns:
        concatenated_data: np.array, shape (Total_Samples, C, F, T)
    """
    print(f"\n🚀 正在处理 {data_type} 数据...")

    input_list = []  # 用于收集所有模型输入

    if not input_path.exists():
        print(f"❌ 文件不存在: {input_path}")
        return np.array([]), 0

    with h5py.File(input_path, 'r') as infile:
        keys = sorted(infile.keys())  # 按名称排序，保证顺序一致

        for key in tqdm(keys, desc=f"{data_type.upper()} Processing", unit="frag"):
            try:
                raw_data = infile[key][()] 

                filtered_data = apply_all_filters(raw_data)

                stft_log_mag = apply_stft_to_data(filtered_data)  # (B, C, F, T)

                raw_data = generating_raw_data_matching_stft(filtered_data)

                model_input = torch.cat([stft_log_mag, raw_data], dim=2)

                model_input = model_input.cpu().numpy()  # 转回 CPU 和 NumPy

                input_list.append(model_input)

            except Exception as e:
                print(f"  ❌ 处理片段 {key} 时出错: {e}")
                continue

    # ✅ 在 Batch 维度 (axis=0) 上拼接所有片段
    if len(input_list) == 0:
        print(f"⚠️  没有成功处理任何片段，返回空数组")
        return np.array([]), 0

    concatenated_data = np.concatenate(input_list, axis=0)  # (Total_Batch, C, F, T)
    print(f"   {data_type}数据输出形状: {concatenated_data.shape}")

    return concatenated_data

# 二、模型部分

class FrequencyBranch(nn.Module):
    def __init__(self, F_in=128, F_out=64):
        super().__init__()
        # 在 F_in=128（频率点）上做局部卷积
        self.local_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        # 1x1 卷积降维：F_in → F_out
        self.reduce = nn.Conv1d(F_in, F_out, kernel_size=1)

    def forward(self, x):
        # x: (B, C, F=128, T=9)
        B, C, Freq, T = x.shape

        # 重排：每个 (B, c, t) 独立处理
        x = x.permute(0, 1, 3, 2).reshape(B * C * T, Freq, 1)  # (B*C*T, 128, 1)
        x = x.permute(0, 2, 1)  # → (B*C*T, 1, 128) ✅ Length=128

        x = self.local_conv(x)  # (B*C*T, 1, 128) → 局部频率模式

        # 1x1 卷积降维 F=128 → F_out=64
        x = self.reduce(x.permute(0, 2, 1))  # → (B*C*T, 128, 1) → (B*C*T, 64, 1)
        x = x.permute(0, 2, 1)  # (B*C*T, 1, 64)

        # 恢复形状
        x = x.reshape(B, C, T, 64).permute(0, 1, 3, 2)  # (B, C, 64, T)
        print('FrequencyBranch完成')
        return x  # (B, C, 64, 9)

class TCNBlock(nn.Module):
    """单个 TCN 残差块（修正版：保持序列长度）"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, downsample=False):
        super().__init__()
        self.downsample = downsample
        
        # ✅ 正确 padding：保证输入输出长度一致
        self.padding = dilation * (kernel_size - 1) // 2  # 例如 dilation=2 → padding=2

        # 残差路径
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 1x1 卷积调整残差维度
        if in_channels != out_channels or downsample:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

        # 最大池化用于下采样（可选）
        self.pool = nn.MaxPool1d(2, stride=2) if downsample else None

    def forward(self, x):
        residual = self.residual(x)
        if self.pool:
            residual = self.pool(residual)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.pool:
            out = self.pool(out)

        # ✅ 现在 out 和 residual 长度一致
        out += residual
        out = self.relu(out)
        print("TCNBlock完成")
        return out

class TimeBranch(nn.Module):
    def __init__(self, t_in=256, t_out=128):
        super().__init__()
        # TCN 在 t=256 上建模局部时间动态
        self.tcn = nn.Sequential(
            TCNBlock(in_channels=1, out_channels=32, kernel_size=3, dilation=1),
            TCNBlock(in_channels=32, out_channels=64, kernel_size=3, dilation=2),
            TCNBlock(in_channels=64, out_channels=128, kernel_size=3, dilation=4),  # 感受野很大
        )
        # TCNBlock(in_channels=128, out_channels=128, kernel_size=3, dilation=8),
        # 最终 1x1 卷积降维（可选）
        self.reduce = nn.Conv1d(128, t_out, kernel_size=1)

    def forward(self, x):
        # x: (B, C, 256, T=9)
        B, C, t, T = x.shape

        # 重排：每个 (B, c, t) 独立处理
        x = x.permute(0, 1, 3, 2).reshape(B * C * T, t, 1)  # (B*C*T, 256, 1)
        x = x.permute(0, 2, 1)  # → (B*C*T, 1, 256) ✅ in_channels=1, Length=256

        # TCN 建模：在 t=256 上提取深层时序特征
        x = self.tcn(x)  # (B*C*T, 128, 256) → 注意：长度仍为 256（因果 padding）

        # 1x1 卷积降维：128 → t_out=128（可保持）
        x = self.reduce(x)  # (B*C*T, 128, 256) → (B*C*T, 128, 256)

        # 全局平均池化压缩时间窗口（t=256 → 1）
        x = x.mean(dim=-1, keepdim=True)  # (B*C*T, 128, 1)

        # 或者使用 AdaptivePool:
        # x = nn.AdaptiveAvgPool1d(1)(x)  # 同上

        # 恢复形状
        x = x.reshape(B, C, T, 128).permute(0, 1, 3, 2)  # (B, C, 128, T)
        print("TimeBranch完成")
        return x  # (B, C, 128, 9)

class AdjacencyMatrixLearning(nn.Module):
    def __init__(self, C=22, T=9, hidden_dim=64):
        super().__init__()
        self.C = C
        self.T = T
        # 投影网络
        self.W1 = nn.Linear(64, hidden_dim)  # freq → hidden
        self.W2 = nn.Linear(64, hidden_dim)
        self.W3 = nn.Linear(128, hidden_dim) # time → hidden
        self.W4 = nn.Linear(128, hidden_dim)

    def forward(self, freq_feat, time_out):
        # freq_feat: (B, C, 64, T)
        # time_out:  (B, C, 128, T)
        B, C, _, T = freq_feat.shape

        # 转置为 (B, T, C, F/t)
        freq = freq_feat.permute(0, 3, 1, 2)  # (B, T, C, 64)
        time = time_out.permute(0, 3, 1, 2)   # (B, T, C, 128)

        # 投影到低维空间
        freq_flat = freq.reshape(B * T, C, 64)
        time_flat = time.reshape(B * T, C, 128)

        # Frequency Path
        f1 = torch.relu(self.W1(freq_flat))  # (BT, C, h)
        f2 = torch.relu(self.W2(freq_flat))
        A_freq = torch.bmm(f1, f2.transpose(1, 2))  # (BT, C, C)

        # Time Path
        t1 = torch.relu(self.W3(time_flat))  # (BT, C, h)
        t2 = torch.relu(self.W4(time_flat))
        A_time = torch.bmm(t1, t2.transpose(1, 2))  # (BT, C, C)

        # 融合 + Softmax
        A = A_freq + A_time  # (BT, C, C)
        A = A / (C ** 0.5)  # Scale
        A = torch.softmax(A, dim=-1)  # (BT, C, C)

        # 恢复时间维度
        A = A.reshape(B, T, C, C)  # (B, T, C, C)
        print("AdjacencyMatrixLearning完成")
        return A  # 动态邻接矩阵
    
class TemporalGCN(nn.Module):
    def __init__(self, in_channels=192, hidden_channels=128, num_layers=2, C=22):
        super().__init__()
        self.C = C
        self.num_layers = num_layers
        
        # 图卷积层（1x1 Conv 实现 GCN）
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # 使用 1x1 Conv 模拟 GCN: (B, C, F) → (B, C, F')
            '''
            self.convs.append(
                nn.Conv1d(C, C, kernel_size=1)  # 在节点维度上操作
            )
            '''
            self.convs.append(
                nn.Linear(in_channels, hidden_channels)  # 特征变换
            )
            in_channels = hidden_channels

        self.out_channels = hidden_channels

    def forward(self, x, A):
        # x: (B, T, C, F) = (B, 9, C, 192)
        # A: (B, T, C, C)
        B, T, C, Freq = x.shape

        # 存储每一步的图表示
        outputs = []

        for t in range(T):
            # 当前时间步特征: (B, C, F)
            xt = x[:, t, :, :]  # (B, C, F)
            At = A[:, t, :, :]  # (B, C, C)

            # GCN 层数
            for i in range(self.num_layers):
                # 1. 邻居聚合: X' = A @ X
                xt = torch.bmm(At, xt)  # (B, C, F)

                # 2. 特征变换 + Conv1d-like update
                conv_idx = i 
                #xt = self.convs[conv_idx](xt.transpose(1, 2)).transpose(1, 2)  # (B, C, F)
                xt = self.convs[conv_idx](xt)  # (B, C, H)

                # 3. ReLU
                xt = F.relu(xt)

            # 保存当前时间步的图表示
            outputs.append(xt)  # (B, C, H)

        # 拼接所有时间步: (B, T, C, H)
        x_gcn = torch.stack(outputs, dim=1)  # (B, T, C, H)
        x_gcn = x_gcn.mean(dim = 2)               #(B, T, H)
        print("Temporal GCN完成")
        return x_gcn  # (B, T, H)

class TemporalModeler(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.out_dim = 2 * hidden_dim  # bidirectional

    def forward(self, x):
        # x: (B, T, H)
        B, T, H = x.shape

        # Bi-GRU over time
        x, _ = self.gru(x)  # (B, T, 2*H)

        print("TemporalModeler完成")
        return x  # (B, T, 2H)
    
class TimeAttentionPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_channels, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (B, T, C, H)
        B, T, H = x.shape

        # 注意力权重
        weights = self.attention(x)  # (B, T, 1)
        weights = torch.softmax(weights, dim=1)

        # 加权求和
        x = (x * weights).sum(dim=1)  # (B, H)
        print("TimeattentionPooling完成")
        return x  # (B, H)
    
class DualPathModel(nn.Module):
    def __init__(self, C=22, Freq=128, t=256, T=9, num_classes=2):
        super().__init__()
        self.C, self.T = C, T

        # 特征提取
        self.freq_branch = FrequencyBranch(F_in=Freq, F_out=64)
        self.time_branch = TimeBranch(t_in=t, t_out=128)

        # 邻接矩阵学习
        self.adj_block = AdjacencyMatrixLearning(C=C, T=T)

        # 空间建模：GCN（每时间步独立）
        self.spatial_gcn = TemporalGCN(in_channels=64+128, hidden_channels=128, C=C)

        # 时间建模：Bi-GRU
        self.temporal_rnn = TemporalModeler(input_dim=128, hidden_dim=64)

        # 时间注意力池化
        self.time_attn_pool = TimeAttentionPooling(2 * 64)  # 2*hidden_dim

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, C, 384, T)
        freq_feat = x[:, :, :128, :]   # (B, C, 128, 9)
        time_feat = x[:, :, 128:, :]   # (B, C, 256, 9)

        freq_out = self.freq_branch(freq_feat)  # (B, C, 64, 9)
        time_out = self.time_branch(time_feat)  # (B, C, 128, 9)

        # 拼接节点特征: (B, C, 192, T)
        x_feat = torch.cat([freq_out, time_out], dim=2)
        x_feat = x_feat.permute(0, 3, 1, 2)  # (B, T, C, 192)

        # 学习动态邻接矩阵 A: (B, T, C, C)
        A = self.adj_block(freq_out, time_out)

        # 空间建模：GCN 在每个时间步上
        x_gcn = self.spatial_gcn(x_feat, A)  # (B, T, C, 128)

        # 时间建模：Bi-GRU 在 T=9 上
        x_temporal = self.temporal_rnn(x_gcn)  # (B, T, C, 256)

        # 时间注意力池化: (B, T, C, 256) → (B, C, 256)
        x_attn = self.time_attn_pool(x_temporal)

        # 分类
        logits = self.classifier(x_attn)
        
        return logits, A  # 可视化邻接矩阵
# -------------------------------
# 执行
# -------------------------------
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

# 训练参数
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
PATIENCE = 10

# -------------------------------
# 训练函数
# -------------------------------
def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """简单的训练函数，只返回最佳验证准确率"""
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_acc = 0.0
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        i = 0
        for batch_x, batch_y in train_loader:
            print(f"Epoch {epoch+1}, Batch {i+1}")
            i += 1
            batch_x, batch_y = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            if i == 3:
                break
        
        # 验证阶段
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
                logits, _ = model(batch_x)
                _, predicted = torch.max(logits, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        val_acc = 100.0 * val_correct / val_total
        print(f"验证准确率: {val_acc:.2f}%")

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
    
    return best_val_acc

def evaluate_model(model, test_loader):
    """评估模型，返回准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
            logits, _ = model(batch_x)
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return 100.0 * correct / total

# -------------------------------
# 主执行代码
# -------------------------------
if __name__ == "__main__":
    # 处理发作前数据
    X_preictal = process_and_save_fragments(PREICTAL_PATH, "preictal")

    # 处理发作间期数据
    X_interictal = process_and_save_fragments(INTERICTAL_PATH, "interictal")
     
    # 保证样本数1比1
    if X_preictal.shape[0] < X_interictal.shape[0]:
        print("Preictal 样本数少于 Interictal")
        X_interictal = X_interictal[:X_preictal.shape[0]]
        print(f"已裁剪 Interictal 至 {X_interictal.shape[0]} 个样本")
    else:
        print("Interictal 样本数少于或等于 Preictal")
        X_preictal = X_preictal[:X_interictal.shape[0]]
        print(f"已裁剪 Preictal 至 {X_preictal.shape[0]} 个样本")
    
    print(f"患者编号: CHB-{PATIENT_ID:02d}")
    print(f"Preictal数据维度: {X_preictal.shape}")
    print(f"Interictal数据维度: {X_interictal.shape}")
    
    # 构建数据集和标签
    X = np.concatenate([X_preictal, X_interictal], axis=0)
    y = np.concatenate([
        np.ones(X_preictal.shape[0]),
        np.zeros(X_interictal.shape[0])
    ], axis=0)

    print(f"\n✅ 数据集构建完成")
    print(f"特征 X 形状: {X.shape}")   
    print(f"标签 y 形状: {y.shape}")  
    print(f"类别分布: {np.bincount(y.astype(int))}")  

    # 五折交叉验证
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []

    print(f"\n🚀 开始五折交叉验证...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/5 ---")
        
        # 准备数据
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # 创建DataLoader
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
        
        # 初始化模型
        model = DualPathModel(C=22, Freq=128, t=256, T=9, num_classes=2)
        
        # 训练模型
        print("开始训练...")
        best_acc = train_model(model, train_loader, val_loader)
        
        # 最终评估
        final_acc = evaluate_model(model, val_loader)
        fold_accuracies.append(final_acc)
        
        print(f"Fold {fold_idx + 1} 准确率: {final_acc:.2f}%")
        
        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算最终结果
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n🎯 === 五折交叉验证结果 ===")
    print(f"患者 CHB{PATIENT_ID:02d}")
    print(f"各折准确率: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"最佳准确率: {max(fold_accuracies):.2f}%")
    print(f"最差准确率: {min(fold_accuracies):.2f}%")