# data_utils.py
import h5py
import numpy as np
import torch
from scipy.signal import butter, filtfilt, iirnotch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# -------------------------------
# 滤波函数（MODIFIED: 添加0.5-45Hz带通滤波）
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

def apply_all_filters(data, fs):  # MODIFIED: 添加带通滤波
    data = data.astype(np.float32)
    filtered = bandpass_filter(data, 0.5, 45.0, fs=fs)  # 新: 0.5-45Hz带通
    # filtered = notch_filter(filtered, 60.0, fs=fs)
    # filtered = notch_filter(filtered, 120.0, fs=fs)
    filtered = remove_dc_component(filtered)
    return filtered

def generating_raw_data_matching_stft(data_array, n_fft, hop_length, device):  # 不变
    if data_array.size == 0:
        return np.array([])

    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)
    return data_tensor.unfold(-1, n_fft, hop_length).permute(0, 2, 1, 3)

# -------------------------------
# 多尺度相位STFT 变换函数（MODIFIED: 缩短到0.5-45Hz，45 bin）
# -------------------------------
def apply_stft_to_data(data_array, config):  # MODIFIED: 改为接受config
    n_fft = config.get('N_FFT', 256)
    hop_length = config.get('HOP_LENGTH', 128)
    win_length_long = config.get('WIN_LENGTH_LONG', 256)
    win_length_short = config.get('WIN_LENGTH_SHORT', 128)
    fs = config.get('FS', 256)
    device = config.get('DEVICE', 'cuda')

    if data_array.size == 0:
        return np.array([])

    n_samples, n_channels, n_timepoints = data_array.shape
    data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device)

    # 长窗STFT (幅度 + 相位) - MODIFIED: 切片到1:46 (1-45Hz, 45 bin)
    window_long = torch.hann_window(win_length_long).to(device)
    data_flat_long = data_tensor.view(-1, n_timepoints)
    stft_complex_long = torch.stft(
        data_flat_long,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length_long,
        window=window_long,
        center=True,
        return_complex=True
    )

    freq_bins_long = n_fft // 2 + 1
    time_frames_long = stft_complex_long.shape[-1]
    stft_complex_long = stft_complex_long.view(n_samples, n_channels, freq_bins_long, time_frames_long)
    stft_complex_long = stft_complex_long[:, :, 1:46, 1:-1]  # MODIFIED: 1:46 -> 45 bin (1-45Hz)

    magnitude_long = stft_complex_long.abs()
    phase_long = torch.angle(stft_complex_long)
    log_magnitude_long = 20 * torch.log10(magnitude_long + 1e-8)

    # 短窗STFT (仅幅度，多尺度) - MODIFIED: 调整到~23 bin，填充到45
    window_short = torch.hann_window(win_length_short).to(device)
    stft_complex_short = torch.stft(
        data_flat_long,
        n_fft=n_fft // 2,
        hop_length=hop_length,
        win_length=win_length_short,
        window=window_short,
        center=True,
        return_complex=True
    )

    freq_bins_short = (n_fft // 2) // 2 + 1
    time_frames_short = stft_complex_short.shape[-1]
    stft_complex_short = stft_complex_short.view(n_samples, n_channels, freq_bins_short, time_frames_short)
    stft_complex_short = stft_complex_short[:, :, 1:24, 1:-1]  # MODIFIED: 1:24 -> ~23 bin (1-23Hz)
    log_magnitude_short = 20 * torch.log10(stft_complex_short.abs() + 1e-8)

    # 填充短尺度到45 bin (零填充)
    target_bins = 45
    pad_short = torch.zeros_like(log_magnitude_long[:, :, :target_bins - log_magnitude_short.shape[-2], :])
    log_magnitude_short = torch.cat([log_magnitude_short, pad_short], dim=2)

    # 标准化每个模态
    log_magnitude_long_norm = (log_magnitude_long - log_magnitude_long.mean(dim=(0,1,2), keepdim=True)) / (log_magnitude_long.std(dim=(0,1,2), keepdim=True) + 1e-8)

    log_magnitude_short_norm = (log_magnitude_short - log_magnitude_short.mean(dim=(0,1,2), keepdim=True)) / (log_magnitude_short.std(dim=(0,1,2), keepdim=True) + 1e-8)

    # 拼接多模态: (N, T, C, 135=45*3)
    multi_modal_stft = torch.cat([log_magnitude_long_norm, phase_long, log_magnitude_short_norm], dim=-2).permute(0, 3, 1, 2)

    return multi_modal_stft

# -------------------------------
# 主处理函数（MODIFIED: 传入fs到滤波）
# -------------------------------
def process_and_save_fragments(data_type, patient_id, config):
    data_dir = config.get('DATA_DIR')
    input_path = data_dir / data_type / f"{data_type}_fragments{patient_id:02d}.h5"
    print(f"\n正在处理 {data_type} 数据... (路径: {input_path})")
    input_list = []
    with h5py.File(input_path, 'r') as infile:
        keys = sorted(infile.keys())
        for key in tqdm(keys, desc=f"{data_type.upper()} Processing", unit="frag"):

            raw_data = infile[key][()]

            filtered_data = apply_all_filters(raw_data, config['FS'])
            batch_size = 500
            for i in range(0, len(filtered_data), batch_size):
                batch_data = filtered_data[i:i + batch_size]
                stft_batch = apply_stft_to_data(batch_data, config)  # 分批次处理
                # 合并结果
                stft_multi_modal = torch.cat([stft_multi_modal, stft_batch], dim=0) if i > 0 else stft_batch

            print(stft_multi_modal.shape)

            stft_multi_modal_mean = stft_multi_modal.mean(dim=(0,1,2), keepdim=True) # (1, 1, 1, 135)
            stft_multi_modal_std = stft_multi_modal.std(dim=(0,1,2), keepdim=True)
            stft_multi_modal_norm = (stft_multi_modal - stft_multi_modal_mean) / (stft_multi_modal_std + 1e-8)

            raw_data_matching = generating_raw_data_matching_stft(filtered_data, config['N_FFT'], config['HOP_LENGTH'], config['DEVICE'])

            raw_data_mean = raw_data_matching.mean(dim=(0,1,2), keepdim=True)
            raw_data_std = raw_data_matching.std(dim=(0,1,2), keepdim=True)
            raw_data_norm = (raw_data_matching - raw_data_mean) / (raw_data_std + 1e-8)
            print(raw_data_norm.shape)

            model_input = torch.cat([stft_multi_modal_norm, raw_data_norm], dim=-1)  # (n, T, C, 135+256=391)

            model_input = model_input.cpu().numpy()

            input_list.append(model_input)

    concatenated_data = np.concatenate(input_list, axis=0)
    print(f"{data_type}数据输出形状: {concatenated_data.shape}")

    return concatenated_data