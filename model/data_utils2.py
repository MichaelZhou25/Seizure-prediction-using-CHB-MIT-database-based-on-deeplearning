import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 滤波函数
# -------------------------------
def bandpass_filter(data, lowcut, highcut, fs, axis=-1, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=axis)

def remove_dc_component(data, axis=-1):
    mean_val = np.mean(data, axis=axis, keepdims=True)
    return data - mean_val

def apply_all_filters(data, fs):
    # data: (N, C, T)
    filtered = bandpass_filter(data, 0.5, 45.0, fs=fs)
    filtered = remove_dc_component(filtered)
    return filtered

# -------------------------------
# 辅助函数：实例归一化
# -------------------------------
def instance_norm_2d(tensor):
    """
    对 STFT 谱图进行归一化
    Input: (N, C, F, T)
    Norm:  针对每个样本(N)和每个通道(C)，在(F,T)面上做归一化
    """
    # dim=(2, 3) 即 (Freq, Time) 维度
    mean = tensor.mean(dim=(2, 3), keepdim=True)
    std = tensor.std(dim=(2, 3), keepdim=True)
    return (tensor - mean) / (std + 1e-8)

def instance_norm_1d(tensor):
    """
    对 Raw Data 切片进行归一化
    Input: (N, T_frames, C, Win_Len)
    Norm:  针对最后一个维度(Win_Len)做归一化
    """
    mean = tensor.mean(dim=-1, keepdim=True)
    std = tensor.std(dim=-1, keepdim=True)
    return (tensor - mean) / (std + 1e-8)

# -------------------------------
# 原始数据切片函数
# -------------------------------
def generating_raw_data_matching_stft(data_array, n_fft, hop_length, device):
    if data_array.size == 0:
        return np.array([])

    # data_array: (N, C, T_total)
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
    
    # unfold: 类似于滑动窗口
    # Input: (N, C, T_total)
    # -> unfold(-1, n_fft, hop_length)
    # Output: (N, C, n_frames, n_fft)
    unfolded = data_tensor.unfold(-1, n_fft, hop_length)
    
    # 调整维度以匹配 STFT 输出
    # (N, C, n_frames, n_fft) -> (N, n_frames, C, n_fft)
    raw_output = unfolded.permute(0, 2, 1, 3)
    
    # 【新增】对原始波形做 Instance Normalization
    raw_output = instance_norm_1d(raw_output)
    
    return raw_output

# -------------------------------
# 多尺度相位STFT 变换函数
# -------------------------------
def apply_stft_to_data(data_array, config):
    n_fft = config.get('N_FFT', 256)
    hop_length = config.get('HOP_LENGTH', 128)
    win_length_long = config.get('WIN_LENGTH_LONG', 256)
    win_length_short = config.get('WIN_LENGTH_SHORT', 128)
    device = config.get('DEVICE', 'cuda')

    if data_array.size == 0:
        return np.array([])

    # data_array: (N, C, T)
    n_samples, n_channels, n_timepoints = data_array.shape
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)

    # -----------------------
    # 1. 长窗 STFT
    # -----------------------
    window_long = torch.hann_window(win_length_long).to(device)
    # view: (N, C, T) -> (N*C, T) 以适应 torch.stft
    data_flat = data_tensor.view(-1, n_timepoints)
    
    stft_complex_long = torch.stft(
        data_flat, n_fft=n_fft, hop_length=hop_length, 
        win_length=win_length_long, window=window_long, 
        center=True, return_complex=True
    )
    # stft output: (N*C, Freq, Time)
    
    # 恢复维度: (N, C, Freq, Time)
    freq_bins_long = n_fft // 2 + 1
    time_frames = stft_complex_long.shape[-1]
    stft_complex_long = stft_complex_long.view(n_samples, n_channels, freq_bins_long, time_frames)
    
    # 切片: 取 1-45Hz (索引 1:46)
    # 维度: (N, C, 45, Time)
    stft_complex_long = stft_complex_long[:, :, 1:46, 1:-1] 

    magnitude_long = stft_complex_long.abs()
    phase_long = torch.angle(stft_complex_long)
    
    # Log 变换 + 【修正】Instance Norm
    log_magnitude_long = 20 * torch.log10(magnitude_long + 1e-8)
    log_magnitude_long_norm = instance_norm_2d(log_magnitude_long)
    
    # 【新增】相位归一化: [-pi, pi] -> [-1, 1]
    phase_long_norm = phase_long / np.pi

    # -----------------------
    # 2. 短窗 STFT
    # -----------------------
    window_short = torch.hann_window(win_length_short).to(device)
    stft_complex_short = torch.stft(
        data_flat, n_fft=n_fft // 2, hop_length=hop_length, 
        win_length=win_length_short, window=window_short, 
        center=True, return_complex=True
    )
    
    # 恢复维度 & 切片
    freq_bins_short = (n_fft // 2) // 2 + 1
    stft_complex_short = stft_complex_short.view(n_samples, n_channels, freq_bins_short, stft_complex_short.shape[-1])
    stft_complex_short = stft_complex_short[:, :, 1:24, 1:-1] # (N, C, 23, Time)
    
    log_magnitude_short = 20 * torch.log10(stft_complex_short.abs() + 1e-8)

    # 填充到 45 bin
    target_bins = 45
    pad_short = torch.zeros(n_samples, n_channels, target_bins - log_magnitude_short.shape[2], log_magnitude_short.shape[-1]).to(device)
    log_magnitude_short = torch.cat([log_magnitude_short, pad_short], dim=2)
    
    # 【修正】Instance Norm
    log_magnitude_short_norm = instance_norm_2d(log_magnitude_short)

    # -----------------------
    # 3. 拼接
    # -----------------------
    # 拼接: (N, C, 45, T) * 3 -> (N, C, 135, T)
    multi_modal_stft = torch.cat([log_magnitude_long_norm, phase_long_norm, log_magnitude_short_norm], dim=2)
    
    # 调整维度: (N, C, F, T) -> (N, T, C, F)
    # 这里的 T 对应 raw data 的切片数，F=135
    multi_modal_stft = multi_modal_stft.permute(0, 3, 1, 2)
    
    # Output: (N, T_frames, C, F_features)
    return multi_modal_stft

# -------------------------------
# 主处理函数
# -------------------------------
def process_and_save_fragments(data_type, patient_id, config):
    data_dir = config.get('DATA_DIR')
    input_path = data_dir / data_type / f"{data_type}_fragments{patient_id:02d}.h5"
    print(f"\n正在处理 {data_type} 数据... (路径: {input_path})")
    
    input_list = []
    
    with h5py.File(input_path, 'r') as infile:
        keys = sorted(infile.keys())
        for key in tqdm(keys, desc=f"{data_type.upper()} Processing", unit="frag"):
            
            # raw_data: (N_samples, Channels, Time)
            # 注意：这里的 N_samples 可能只是 1，如果是单个 fragment
            raw_data = infile[key][()] 
            
            # 1. 滤波
            filtered_data = apply_all_filters(raw_data, config['FS'])
            
            # 2. STFT (分 Batch 处理)
            stft_results = [] # 使用列表收集结果，比反复 concat 更高效
            batch_size = 500
            for i in range(0, len(filtered_data), batch_size):
                batch_data = filtered_data[i:i + batch_size]
                if batch_data.shape[0] == 0: continue
                
                stft_batch = apply_stft_to_data(batch_data, config)
                stft_results.append(stft_batch)
            
            if not stft_results: continue
            
            # 合并 Batch: (N, T, C, 135)
            stft_multi_modal = torch.cat(stft_results, dim=0) 

            # 3. Raw Data Matching (包含 Instance Norm)
            # Output: (N, T, C, 256)
            raw_data_matching = generating_raw_data_matching_stft(filtered_data, config['N_FFT'], config['HOP_LENGTH'], config['DEVICE'])
            
            # 确保 Raw Data 和 STFT 的样本数一致（处理边界情况）
            min_len = min(stft_multi_modal.shape[0], raw_data_matching.shape[0])
            stft_multi_modal = stft_multi_modal[:min_len]
            raw_data_matching = raw_data_matching[:min_len]

            # 4. 最终拼接
            # STFT (N, T, C, 135) + Raw (N, T, C, 256) -> (N, T, C, 391)
            model_input = torch.cat([stft_multi_modal, raw_data_matching], dim=-1)

            model_input = model_input.cpu().numpy()
            input_list.append(model_input)

    if not input_list:
        return np.array([])

    concatenated_data = np.concatenate(input_list, axis=0)
    print(f"{data_type}数据输出形状: {concatenated_data.shape}")

    return concatenated_data