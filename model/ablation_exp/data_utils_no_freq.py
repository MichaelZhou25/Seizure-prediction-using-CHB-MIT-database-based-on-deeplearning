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
    filtered = bandpass_filter(data, 0.5, 45.0, fs=fs)  # 新: 0.5-45Hz带通
    # filtered = notch_filter(filtered, 60.0, fs=fs)
    # filtered = notch_filter(filtered, 120.0, fs=fs)
    filtered = remove_dc_component(filtered)
    return filtered

def process_and_save_fragments(data_type, patient_id, config):
    data_dir = config.get('DATA_DIR')
    input_path = data_dir / data_type / f"{data_type}_fragments{patient_id:02d}.h5"
    print(f"\n正在处理 {data_type} 数据... (路径: {input_path})")
    input_list = []
    with h5py.File(input_path, 'r') as infile:
        keys = sorted(infile.keys())
        for key in tqdm(keys, desc=f"{data_type.upper()} Processing", unit="frag"):

            raw_data = infile[key][()]  # shape: (n, C, 1280) 或 (C, 1280)？需确认！

            # 应用滤波
            filtered_data = apply_all_filters(raw_data, config['FS'])

            # 标准化：按通道和时间维度标准化
            # 假设 raw_data 形状为 (n, C, T)，则 axis=(0,2) 计算每个通道的均值/标准差
            raw_data_mean = np.mean(filtered_data, axis=(0, 2), keepdims=True)  # (1, C, 1)
            raw_data_std = np.std(filtered_data, axis=(0, 2), keepdims=True)    # (1, C, 1)
            raw_data_norm = (filtered_data - raw_data_mean) / (raw_data_std + 1e-8)  # (n, C, T)

            # 保存到列表
            input_list.append(raw_data_norm)

    # 合并所有片段
    concatenated_data = np.concatenate(input_list, axis=0)  # (N, C, T)
    print(f"{data_type}数据输出形状: {concatenated_data.shape}")

    return concatenated_data