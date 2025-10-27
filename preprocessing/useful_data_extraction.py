import h5py
import json
import pandas as pd
import pyedflib
import numpy as np
from pathlib import Path
import os
from typing import List, Tuple, Optional

DATA_ROOT = Path(r'D:\陈教授组\CHB-MIT') 
SAVE_DIR = r"D:\陈教授组\mymodel\data2"
USEFUL_DATA_INFO_ROOT = Path(r'D:\陈教授组\mymodel\data\useful_data_info.json')
PATIENT_NUM = [i for i in range(1, 25) if i not in [12, 13, 15, 24]]  # 假设有24个患者
SAMPLE_LENGTH = 5  # 秒
CHANNEL_NUM = 22  # 通道数
SAMPLING_OVERLAP = 1  # 重叠比例
TARGET_CHANNELS = [
    'C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 'FP1-F3', 'FP1-F7', 'FP2-F4',
    'FP2-F8', 'FT10-T8', 'FT9-FT10', 'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P7-T7', 'P8-O2', 'T7-FT9',
    'T7-P7', 'T8-P8'
]

def read_edf_segment(
    file_path: Path,
    start_sec: float,
    end_sec: float,
    target_channels: List[str]
) -> Tuple[Optional[np.ndarray], List[str], float, List[str]]:
  
    with pyedflib.EdfReader(str(file_path)) as f:
        sample_rate = f.getSampleFrequency(0)
        all_file_channels = f.getSignalLabels()

        # 找出文件中存在且在目标列表中的通道
        common_channels = [ch for ch in target_channels if ch in all_file_channels]

        if not common_channels:
            print(f"  文件 {file_path.name} 中未找到任何目标通道。")
            return None, [], sample_rate, []

        # 去重，保持顺序
        seen = set()
        final_channels = []
        removed_duplicates = []
        for ch in common_channels:
            if ch in seen:
                removed_duplicates.append(ch)
            else:
                seen.add(ch)
                final_channels.append(ch)

        if removed_duplicates:
            print(f"  在文件 {file_path.name} 中检测到并移除了重复通道: {removed_duplicates}")

        # 计算样本索引
        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        # 读取数据
        data = np.vstack([
            f.readSignal(all_file_channels.index(ch), start_sample, end_sample - start_sample)
            for ch in final_channels
        ])

        return data, final_channels, sample_rate, removed_duplicates


def process_segments(segment_list: List, data_label: str, parent_file_path: Path) -> List[dict]:

    all_data = []
    global_segment_index = 0  # 用于全局编号

    for i, segment in enumerate(segment_list, 1):

        file_name, time_interval = segment
        global_segment_index += 1
        start_sec, end_sec = time_interval
        print(f"正在处理第 {global_segment_index} 个 {data_label} 段: {file_name} ({start_sec:.1f} - {end_sec:.1f}s)")
        file_path = Path(parent_file_path) / file_name
        if not file_path.exists():
            print(f"  警告: 文件 {file_path} 不存在，跳过...")
            continue

        try:
            data_segment, channels, sample_rate, removed = read_edf_segment(
                file_path, start_sec, end_sec, TARGET_CHANNELS
            )

            if data_segment is None:
                print(f"  文件 {file_name} 中无有效目标通道，跳过此时间段。")
                continue

            duration = data_segment.shape[1] / sample_rate

            print(f"  读取成功: 形状 {data_segment.shape}, 时长 {duration:.1f}s, "
                      f"保留 {len(channels)} 个通道")
            if removed:
                print(f"  移除重复通道: {list(set(removed))}")

            all_data.append({
                'data': data_segment,
                'channels': channels,
                'sample_rate': sample_rate,
                'duration_seconds': duration,
                'file': file_name,
                'start_sec': start_sec,
                'end_sec': end_sec
            })

        except Exception as e:
            print(f"  读取 {file_name} 时出错: {e}")
            continue

        print("-" * 60)

    print(f"\n✅ 总共成功提取了 {len(all_data)} 个独立的 {data_label} 段。\n")
    return all_data


if __name__ == "__main__":

    patient_data_statistics_list = []  # 用于存储所有患者的数据统计信息
    
    with open(USEFUL_DATA_INFO_ROOT, 'r', encoding='utf-8') as f:
        useful_data_info = json.load(f)

    for i, patient_data in enumerate(useful_data_info):
        preictal_segments = patient_data[0]
        interictal_segments = patient_data[1]
        patient = PATIENT_NUM[i]
        print("🚀 开始处理 preictal (预发作期) 数据...\n")
        all_preictal_data = process_segments(preictal_segments, "preictal", parent_file_path = DATA_ROOT / f"chb{patient:02d}")
       
        # 处理 interictal 数据
        print("🚀 开始处理 interictal (间歇期) 数据...\n")
        all_interictal_data = process_segments(interictal_segments, "interictal", parent_file_path = DATA_ROOT / f"chb{patient:02d}")
        
        # 统计 preictal 和 interictal 的总时长（秒），合并所有段
        total_preictal_duration = sum([d['duration_seconds'] for d in all_preictal_data])
        total_interictal_duration = sum([d['duration_seconds'] for d in all_interictal_data])

        # 新增：构建子目录路径
        preictal_subdir = os.path.join(SAVE_DIR, "preictal")
        interictal_subdir = os.path.join(SAVE_DIR, "interictal")

        # 确保所有需要的目录都存在（自动创建多级目录）
        os.makedirs(preictal_subdir, exist_ok=True)
        os.makedirs(interictal_subdir, exist_ok=True)

        # 构建保存路径
        save_path_preictal = os.path.join(preictal_subdir, f"preictal_fragments{patient:02d}.h5")
        save_path_interictal = os.path.join(interictal_subdir, f"interictal_fragments{patient:02d}.h5")

        # 分割并保存 preictal 片段
        if total_preictal_duration < total_interictal_duration:
            preictal_step, interictal_step = SAMPLE_LENGTH * SAMPLING_OVERLAP, SAMPLE_LENGTH
        else:
            preictal_step, interictal_step = SAMPLE_LENGTH, SAMPLE_LENGTH * SAMPLING_OVERLAP
            print(f"⚠️ Patient {patient:02d} 的 interictal 总时长不够，调整采样步长为 {interictal_step}s")

        with h5py.File(save_path_preictal, 'w') as f:
            total_preictal_sample = 0
            for seg_idx, seg in enumerate(all_preictal_data, 1):
                preictal_list = []
                data = seg['data']  # shape: (channels, timepoints)
                sample_rate = seg['sample_rate']
                total_points = data.shape[1]
                window_points = int(SAMPLE_LENGTH * sample_rate)
                step_points = int(preictal_step * sample_rate)
                # 只保留前22通道
                data = data[:CHANNEL_NUM, :]
                # 计算片段数
                a = int(np.floor((data.shape[1] - window_points) / step_points) + 1)
                total_preictal_sample += a
                for i in range(a):
                    start = i * step_points
                    end = start + window_points
                    if end > total_points:
                        break
                    frag = data[:, start:end]
                    preictal_list.append(frag)
                if preictal_list:  # 避免空列表堆叠
                    preictal_fragments = np.stack(preictal_list, axis=0)
                    f.create_dataset(f'fragment_{seg_idx:02d}', data=preictal_fragments, compression='gzip')
                    print(f"preictal文件 {seg['file']} 形状：{preictal_fragments.shape}")
                else:
                    print(f"preictal文件 {seg['file']} 未生成任何片段")
            print(f"所有 preictal 段总样本数: {total_preictal_sample}")

        # 分割并保存 interictal 片段
        with h5py.File(save_path_interictal, 'w') as f:
            total_interictal_sample = 0
            for seg_idx, seg in enumerate(all_interictal_data, 1):
                interictal_list = []
                data = seg['data']
                sample_rate = seg['sample_rate']
                total_points = data.shape[1]
                window_points = int(SAMPLE_LENGTH * sample_rate)
                step_points = int(interictal_step * sample_rate)  # 注意：这里 K_DETERMINED 未使用
                data = data[:CHANNEL_NUM, :]
                b = int(np.floor((data.shape[1] - window_points) / step_points) + 1)
                total_interictal_sample += b
                for i in range(b):
                    start = i * step_points
                    end = start + window_points
                    if end > total_points:
                        break
                    frag = data[:, start:end]
                    interictal_list.append(frag)
                if interictal_list:  # 避免空列表堆叠
                    interictal_fragments = np.stack(interictal_list, axis=0)
                    f.create_dataset(f'fragment_{seg_idx:02d}', data=interictal_fragments, compression='gzip')
                    print(f"interictal文件 {seg['file']} 形状：{interictal_fragments.shape}")
                else:
                    print(f"interictal文件 {seg['file']} 未生成任何片段")
                if total_interictal_sample >= total_preictal_sample:
                    print("已达到与 preictal 相同的样本数，停止处理更多 interictal 段。")
                    break
            print(f"所有 interictal 段总样本数: {total_interictal_sample}")
            
        print(f"Patient {patient:02d} 数据已保存为 HDF5 格式：")
        patient_data_statistics = pd.Series({
            'Patient_Number': patient,
            'Preictal_Time_Duration': round(total_preictal_duration,2) ,
            'Interictal_Time_Duration': round(total_interictal_duration,2),
            'Total_Preictal_Samples_Selected': total_preictal_sample,
            'Total_Interictal_Samples_Selected': total_interictal_sample
        })
        patient_data_statistics_list.append(patient_data_statistics)

    # 将所有 Series 转换为 DataFrame 表格
    data_table = pd.concat(patient_data_statistics_list, axis=1).T
    data_table.reset_index(drop=True, inplace=True)

    # 保存为 CSV 文件
    csv_save_path = os.path.join(SAVE_DIR, "patient_data_statistics.csv")
    data_table.to_csv(csv_save_path, index=False, float_format='%.4f')
    print(f"所有患者的数据统计信息已保存为 CSV 文件：{csv_save_path}")
