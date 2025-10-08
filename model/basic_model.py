import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm  # 进度条
import warnings
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import time  


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

    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=axis)

def notch_filter(data, freq, fs, Q=30, axis=-1):

    b, a = iirnotch(freq, Q, fs)
    return filtfilt(b, a, data, axis=axis)

def remove_dc_component(data, axis=-1):

    mean_val = np.mean(data, axis=axis, keepdims=True)
    return data - mean_val

def apply_all_filters(data, fs=FS):

    filtered = notch_filter(data, 60.0, fs=fs)
    filtered = notch_filter(filtered, 120.0, fs=fs)
    filtered = remove_dc_component(filtered)
    return filtered


def generating_raw_data_matching_stft(data_array, n_fft=N_FFT, hop_length=HOP_LENGTH, device=DEVICE):
    if data_array.size == 0:
        return np.array([])

    # 转为 tensor 并移到设备
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)  # (N, C, timepoints)

    return data_tensor.unfold(-1, n_fft, hop_length).permute(0, 2, 1, 3)  # (N, C, T, t) -> (N, T, C, t)


# -------------------------------
# STFT 变换函数（支持批量 + GPU）
# -------------------------------
def apply_stft_to_data(data_array, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       win_length=WIN_LENGTH, fs=FS, device=DEVICE):

    if data_array.size == 0:
        return np.array([])

    n_samples, n_channels, n_timepoints = data_array.shape

    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)  # (N, C, timepoints)

    window = torch.hann_window(win_length).to(device)

    data_flat = data_tensor.view(-1, n_timepoints)  # (N*C, timepoints)

    # 应用 STFT（返回复数）
    stft_complex = torch.stft(
        data_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,          
        return_complex=True   
    )  # shape: (N*C, F0, T_frames)

    # 恢复通道维度
    freq_bins = n_fft // 2 + 1
    time_frames = stft_complex.shape[-1]
    stft_complex = stft_complex.view(n_samples, n_channels, freq_bins, time_frames)

    # 删除 0Hz (DC 分量) 和 T_frames的首尾帧
    stft_complex = stft_complex[:, :, 1:, 1:-1]  # (N, C, F, T)

    # 转为对数幅度谱：20 * log10(|Z| + ε)
    magnitude = stft_complex.abs()  # (N, C, F, T)
    log_magnitude_stft = 20 * torch.log10(magnitude + 1e-8)

    return log_magnitude_stft.permute(0, 3, 1, 2)  # (N, C, F, T) → (N, T, C, F)

# -------------------------------
# 主处理函数：逐片段处理并保存
# -------------------------------
def process_and_save_fragments(input_path, data_type):

    print(f"\n正在处理 {data_type} 数据...")

    input_list = []  # 用于收集所有模型输入

    with h5py.File(input_path, 'r') as infile:
        keys = sorted(infile.keys())  # 按名称排序，保证顺序一致

        for key in tqdm(keys, desc=f"{data_type.upper()} Processing", unit="frag"):
            raw_data = infile[key][()] 

            filtered_data = apply_all_filters(raw_data)

            stft_log_mag = apply_stft_to_data(filtered_data)  # (B, T, C, F)
            print(f"STFT 对数幅度谱形状: {stft_log_mag.shape}")
            raw_data = generating_raw_data_matching_stft(filtered_data) # (B, T, C, t)
            print(f"原始时域数据形状: {raw_data.shape}")
            model_input = torch.cat([stft_log_mag, raw_data], dim=-1)  # (B, T, C, F+t)

            model_input = model_input.cpu().numpy()  # 转回 CPU 和 NumPy

            input_list.append(model_input)

    concatenated_data = np.concatenate(input_list, axis=0)  # (Total_samples, T, C, F+t)
    print(f"{data_type}数据输出形状: {concatenated_data.shape}")

    return concatenated_data

# 二、模型部分

# -------------------------------
# 模型各个模块
# -------------------------------
class FrequencyBranch(nn.Module):
    def __init__(self, in_freq=128, out_freq=64):
        super().__init__()
        # 在 in_freq=128（频率点）上做局部卷积
        self.local_conv = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, 3, padding=1),
            nn.BatchNorm1d(1),
            nn.ReLU()
        )
        # 1x1 卷积降维：in_freq → out_freq
        self.reduce = nn.Conv1d(in_freq, out_freq, kernel_size=1)

    def forward(self, x):

        B, T, C, Freq = x.shape

        x = x.reshape(B * T * C, 1, Freq)  # (B*T*C, 1, 128)

        x = self.local_conv(x)  # (B*T*C, 1, 128)

        x = self.reduce(x.permute(0, 2, 1))  # (B*T*C, 128, 1) → (B*T*C, 64, 1)

        x = x.permute(0, 2, 1)  # (B*T*C, 1, 64)

        x = x.reshape(B, T, C, 64)  # (B, T, C, 64)

        return x  # (B, T, C, 64)

class TCNBlock(nn.Module):
    """单个 TCN 残差块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, downsample=False):
        super().__init__()
        self.downsample = downsample
        self.total_padding = dilation * (kernel_size - 1)  # 全部用于左填充

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 残差路径调整
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels or downsample else nn.Identity()
        self.pool = nn.MaxPool1d(2, stride=2) if downsample else None

    def forward(self, x):
        
        residual = self.residual(x)  
        if self.pool:
            residual = self.pool(residual)

        out = F.pad(x, (self.total_padding, 0))
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = F.pad(out, (self.total_padding, 0))  
        out = self.conv2(out)
        out = self.bn2(out)

        if self.pool:
            out = self.pool(out)

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

class AdjacencyMatrixLearning(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # 投影网络
        self.W1 = nn.Linear(64, hidden_dim)  
        self.W2 = nn.Linear(64, hidden_dim)
        self.W3 = nn.Linear(64, hidden_dim) 
        self.W4 = nn.Linear(64, hidden_dim)

    def forward(self, freq_feature, time_feature):

        B, T, C, _ = freq_feature.shape

        # 投影到低维空间
        freq_feature = freq_feature.reshape(B * T, C, 64)
        time_feature = time_feature.reshape(B * T, C, 64)

        # Frequency Path
        f1 = torch.relu(self.W1(freq_feature))  # (B*T, C, hidden_dim)
        f2 = torch.relu(self.W2(freq_feature))
        A_freq = torch.bmm(f1, f2.transpose(1, 2)) # (B*T, C, C)

        # Time Path
        t1 = torch.relu(self.W3(time_feature))  # (B*T, C, hidden_dim)
        t2 = torch.relu(self.W4(time_feature))
        A_time = torch.bmm(t1, t2.transpose(1, 2)) # (B*T, C, C)

        # 融合 + Softmax
        A = A_freq + A_time  # (B*T, C, C)
        A = A / (C ** 0.5)  # Scale
        A = torch.softmax(A, dim=-1)  # (B*T, C, C)

        A = A.reshape(B, T, C, C)  # (B, T, C, C)

        return A  
    
class TemporalGCN(nn.Module):
    def __init__(self, gcn_input_dim=64+64, gcn_layers=2, gru_hidden_dim=32):
        super().__init__()

        self.num_layers = gcn_layers
        self.gru_hidden_dim = gru_hidden_dim

        # 图卷积层：使用 Linear 实现特征变换
        self.graph_convs = nn.ModuleList()
        for i in range(gcn_layers):
            self.graph_convs.append(
                nn.Linear(gcn_input_dim, gcn_input_dim // 2)
            )
            gcn_input_dim = gcn_input_dim // 2  

        self.gcn_output_dim = gcn_input_dim

        # Bi-GRU 层
        self.gru = nn.GRU(
            input_size = self.gcn_output_dim,
            hidden_size = gru_hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.gru_output_dim = 2 * gru_hidden_dim  # bidirectional

    def forward(self, x, A):

        B, T, C, FT = x.shape

        x = x.reshape(B * T, C, FT)  # (B*T, C, 128)
        A = A.reshape(B * T, C, C)  # (B*T, C, C)

        # 多层 GCN 传播
        for i in range(self.num_layers):
            # 邻居聚合: X' = A @ X
            x = torch.bmm(A, x)  # (B*T, C, 128)

            # 特征变换: Linear layer
            x = self.graph_convs[i](x)  # (B*T, C, H)

            # 非线性激活
            x = F.relu(x)

        x = x.reshape(B, T, C, self.gcn_output_dim) # (B*T, C, 32) -> (B, T, C, 32)

        # 在电极通道维度取平均: (B, T, C, 32) -> (B, T, 32)
        x_gcn = x.mean(dim=2)  

        x_tgcn, _ = self.gru(x_gcn)  # (B, T, 64)

        return x_tgcn  # (B, T, 64)

# -------------------------------
# 模型主结构
# -------------------------------
class MainModel(nn.Module):
    def __init__(self, Freq=128, Time=256, num_classes=2):
        super().__init__()

        # 特征提取
        self.freq_branch = FrequencyBranch(in_freq=Freq, out_freq=64)
        self.time_branch = TimeBranch(in_time=Time, out_time=64)

        # 邻接矩阵学习
        self.adj_block = AdjacencyMatrixLearning(hidden_dim=32)

        # TGCN
        self.temporal_gcn = TemporalGCN(gcn_input_dim=64+64)

        # FC 分类
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: (B, T=9, C=22, 384)
        freq_feat = x[:, :, :, :128]   # (B, 9, 22, 128)
        time_feat = x[:, :, :, 128:]   # (B, 9, 22, 256)

        freq_out = self.freq_branch(freq_feat)  # (B, 9, 22, 64)
        time_out = self.time_branch(time_feat)  # (B, 9, 22, 64)

        # 拼接节点特征
        x_feat = torch.cat([freq_out, time_out], dim=3) # (B, 9, 22, 128)

        # 学习动态邻接矩阵
        A = self.adj_block(freq_out, time_out) # (B, 9, 22, 22)

        # TGCN
        x_tgcn = self.temporal_gcn(x_feat, A)  # (B, 9, 64)

        # 时间注意力池化
        x_time_avg = x_tgcn.mean(dim=1)  # (B, 64)

        # 分类
        logits = self.classifier(x_time_avg) # (B, num_classes=2)

        return logits, A  
    
# 三、训练和评估部分
# -------------------------------
# 超参数设置
# -------------------------------
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
PATIENCE = 10

# -------------------------------
# 训练函数
# -------------------------------
def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.title('训练和验证损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='训练准确率')
    plt.plot(epochs, val_accuracies, 'r-', label='验证准确率')
    plt.title('训练和验证准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()

def train_model(model, train_loader, val_loader, num_epochs=NUM_EPOCHS, patience=PATIENCE):
    """
    训练模型并返回最佳验证准确率
    """
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_acc = 0.0
    early_stop_counter = 0
    best_model_state = None 

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 从 sklearn 导入评估指标
    from sklearn.metrics import confusion_matrix, recall_score, accuracy_score

    # 初始化总耗时
    total_start_time = time.time()

    for epoch in range(num_epochs):
        # --- 开始计时 ---
        epoch_start_time = time.time()

        # 训练阶段
        train_loss = 0
        train_correct = 0
        train_total = 0

        model.train()
        for i, data in enumerate(train_loader):
            batch_x, batch_y = data[0], data[1]
            batch_x, batch_y = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(batch_x)
            _, predicted = torch.max(logits, 1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
            if (i > 0 and i % 19 == 0) or (i == len(train_loader) - 1):
                print("第{}轮，第{}个batch，训练损失：{:.2f}，训练准确率：{:.2f}%".format(epoch+1, i+1, train_loss/(i+1), 100.0 * train_correct / train_total))

        # 计算并记录训练指标
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        # 收集所有真实标签和预测结果，用于 sklearn 计算
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.float().to(DEVICE), batch_y.long().to(DEVICE)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)
                
                # 收集到 CPU 列表中
                all_labels.extend(batch_y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

        # 计算平均验证损失
        avg_val_loss = val_loss / len(val_loader)
        # 使用 sklearn 计算各项指标
        val_acc = accuracy_score(all_labels, all_preds) * 100.0
        # Sensitivity (Recall for the positive class, assuming 1 is positive)
        sensitivity = recall_score(all_labels, all_preds, pos_label=1) * 100.0
        # Specificity (Recall for the negative class)
        specificity = recall_score(all_labels, all_preds, pos_label=0) * 100.0

        # 记录验证指标 (✅ 关键修复)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # --- 结束计时并计算耗时 ---
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - total_start_time

        # 打印结果（包含耗时信息）
        print(f"验证损失: {avg_val_loss:.2f}, "
              f"验证准确率: {val_acc:.2f}%, "
              f"验证Sensitivity: {sensitivity:.2f}%, "
              f"验证Specificity: {specificity:.2f}% | "
              f"本轮耗时: {epoch_duration:.1f}s | "
              f"总耗时: {total_elapsed_time:.1f}s")

        # 早停检查
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            early_stop_counter = 0
            # 保存最佳模型参数
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"⚠️ 早停触发，当前最佳验证准确率: {best_val_acc:.2f}% | 总训练耗时: {total_elapsed_time:.1f}s")
                break
    
    # 绘制损失和准确率曲线
    plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies)
    print(f"最佳验证准确率: {best_val_acc:.2f}% | 本次训练总耗时: {total_elapsed_time:.1f}s")

    # 恢复最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 清理 GPU 缓存 (✅ 增强内存管理)
    torch.cuda.empty_cache()
    # 删除不再需要的变量
    del all_labels, all_preds, best_model_state
    torch.cuda.empty_cache()

    return best_val_acc

# 四、主执行代码

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
        model = MainModel(Freq=128, Time=256, num_classes=2)

        # 训练模型
        print("开始训练...")
        best_acc = train_model(model, train_loader, val_loader)
        fold_accuracies.append(best_acc)

        print(f"Fold {fold_idx + 1} 准确率: {best_acc:.2f}%")

        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算最终结果
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    print(f"\n=== 五折交叉验证结果 ===")
    print(f"患者 CHB{PATIENT_ID:02d}")
    print(f"各折准确率: {[f'{acc:.2f}%' for acc in fold_accuracies]}")
    print(f"平均准确率: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"最佳准确率: {max(fold_accuracies):.2f}%")
    print(f"最差准确率: {min(fold_accuracies):.2f}%")