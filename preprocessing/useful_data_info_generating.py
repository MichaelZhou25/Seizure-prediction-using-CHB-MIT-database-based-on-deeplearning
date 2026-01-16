import json
from pathlib import Path
import portion as P
from typing import List, Dict, Tuple, Optional

#CHB-MIT
DATASET = 'CHB-MIT'
DATA_DIR = Path(r'D:\SeizurePrediction\CHB-MIT')
RESULT_DIR = Path(r'D:\SeizurePrediction\EpilepsyPrediction\datainfo\CHBMIT_useful_data_info.json')
NUM_PATIENT = 24
PATIENT_DELETED = [0,12,13,15,24]
PREICTAL_DURATION_MINUTES = 35
PREICTAL_END_MINUTES = 5
MIN_SEIZURE_INTERVAL_MINUTES = 40
EXCLUDED_TIME = 240
TIME_SEPARATION_MARK = ':'
EDF_FILE_SEIZURE_START = 'File Start Time:'
EDF_FILE_SEIZURE_END = 'File End Time:'
'''
#Siena
DATASET = 'Siena'
DATA_DIR = Path(r'D:\SeizurePrediction\siena-scalp-eeg-database-1.0.0')
RESULT_DIR = Path(r'D:\SeizurePrediction\EpilepsyPrediction\datainfo\Siena_useful_data_info.json')
NUM_PATIENT = 17
PATIENT_DELETED = [0,2,4,8,15]
PREICTAL_DURATION_MINUTES = 35
PREICTAL_END_MINUTES = 5
MIN_SEIZURE_INTERVAL_MINUTES = 40
EXCLUDED_TIME = 60
TIME_SEPARATION_MARK = '.'
EDF_FILE_SEIZURE_START = 'Registration start time:'
EDF_FILE_SEIZURE_END = 'Registration end time:'
'''
def parse_time_to_seconds(time_str: str) -> int:
    """将时间字符串转换为秒数"""
    try:
        parts = time_str.split(TIME_SEPARATION_MARK)
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return int(time_str)
    except:
        return 0

def summarize_edf_files(summary_text: str) -> List[Dict]:
    """解析所有文件的基本信息"""
    files_info = []
    lines = summary_text.strip().split('\n')
    
    current_file = None
    for line in lines:
        line = line.strip()
        if line.startswith('File Name:') or line.startswith('File name:'):
            current_file = line.split(':', 1)[1].strip()
        elif line.startswith(EDF_FILE_SEIZURE_START) and current_file:
            start_time_str = line.split(':', 1)[1].strip()
        elif line.startswith(EDF_FILE_SEIZURE_END) and current_file:
            end_time_str = line.split(':', 1)[1].strip()
            
            # 计算文件时长（秒）
            start_seconds = parse_time_to_seconds(start_time_str)
            end_seconds = parse_time_to_seconds(end_time_str)
            
            # 处理跨夜情况
            if end_seconds < start_seconds:
                end_seconds += 24 * 3600  
                        
            files_info.append({
                'filename': current_file,
                'start_time_str': start_time_str,
                'end_time_str': end_time_str,
                'start_time_seconds': start_seconds,
                'end_time_seconds': end_seconds,
                'duration_seconds': end_seconds - start_seconds
            })
            current_file = None
    
    if not files_info:  # Check if list is empty
        print("No EDF files were found or parsed.")
        return [], []  # Or handle appropriately
    # 计算文件间的间隔时间
    files_info[0]['gap_before'] = 0       
    for i in range(1, len(files_info)):
        prev_file = files_info[i - 1]
        curr_file = files_info[i]
        prev_end_time = prev_file['end_time_seconds']
        curr_start_time = curr_file['start_time_seconds']
                
        # 处理跨夜情况
        if curr_start_time < prev_end_time:
            curr_start_time += 24 * 3600  

        gap_seconds = curr_start_time - prev_end_time
        gap_seconds = max(0, gap_seconds)
        curr_file['gap_before'] = gap_seconds

    # 计算文件累积时间
    cumulative_time = 0
    for file_info in files_info:
        file_info['cumulative_start'] = cumulative_time + file_info['gap_before']
        file_info['cumulative_end'] = file_info['cumulative_start'] + file_info['duration_seconds'] 
        file_info['cumulative_interval'] = (file_info['cumulative_start'], file_info['cumulative_end'])
        cumulative_time = file_info['cumulative_end']

    return files_info

def extract_seizures_with_cumulative_time(files_info: List[Dict], summary_text) -> List[Dict]:
    """从文件信息中提取所有发作，使用累积时间"""
    seizures = []
    lines = summary_text.strip().split('\n')
    
    current_file = None
    current_file_info = None
    
    for line in lines:
        line = line.strip()

        if DATASET == 'CHB-MIT':
            if line.startswith('File Name:'):
                current_file = line.split(':', 1)[1].strip()
                current_file_info = next((f for f in files_info if f['filename'] == current_file), None)
            elif 'Seizure' in line and 'Start Time:' in line and current_file and current_file_info:
                seizure_start_in_file = float(line.split(':', 1)[1].strip().split()[0])
                seizure_start_cumulative = current_file_info['cumulative_start'] + seizure_start_in_file
                seizures.append({
                    'filename': current_file,
                    'start_in_file': seizure_start_in_file,
                    'start_cumulative': seizure_start_cumulative,
                    'end_in_file': None,
                    'end_cumulative': None
                })

            elif 'Seizure' in line and 'End Time:' in line and current_file and current_file_info and seizures:
                seizure_end_in_file = float(line.split(':', 1)[1].strip().split()[0])
                seizure_end_cumulative = current_file_info['cumulative_start'] + seizure_end_in_file
                seizures[-1]['end_in_file'] = seizure_end_in_file
                seizures[-1]['end_cumulative'] = seizure_end_cumulative
                seizures[-1]['interval_cumulative'] = (seizures[-1]['start_cumulative'], seizures[-1]['end_cumulative'])

        if DATASET == 'Siena':
            if line.startswith('File name:'):
                current_file = line.split(':', 1)[1].strip()
                current_file_info = next((f for f in files_info if f['filename'] == current_file), None)
            elif 'Seizure' in line and 'start time:' in line and current_file and current_file_info:
                seizure_start_in_file = parse_time_to_seconds(line.split(':', 1)[1].strip()) - current_file_info['start_time_seconds']
                if seizure_start_in_file < 0:
                    seizure_start_in_file += 24 * 3600
                seizure_start_cumulative = current_file_info['cumulative_start'] + seizure_start_in_file
                seizures.append({
                    'filename': current_file,
                    'start_in_file': seizure_start_in_file,
                    'start_cumulative': seizure_start_cumulative,
                    'end_in_file': None,
                    'end_cumulative': None
                })

            elif 'Seizure' in line and 'end time:' in line and current_file and current_file_info and seizures:
                seizure_end_in_file = parse_time_to_seconds(line.split(':', 1)[1].strip()) - current_file_info['start_time_seconds']
                if seizure_end_in_file < 0:
                    seizure_end_in_file += 24 * 3600

                seizure_end_cumulative = current_file_info['cumulative_start'] + seizure_end_in_file
                seizures[-1]['end_in_file'] = seizure_end_in_file
                seizures[-1]['end_cumulative'] = seizure_end_cumulative
                seizures[-1]['interval_cumulative'] = (seizures[-1]['start_cumulative'], seizures[-1]['end_cumulative'])

    return seizures

def select_valid_seizures(seizures: List[Dict], min_interval_minutes: int = MIN_SEIZURE_INTERVAL_MINUTES) -> List[Dict]:
    """过滤间隔过短的发作"""

    filtered_seizures = [seizures[0]]  # 保留第一个发作
    
    for i in range(1, len(seizures)):
        prev_seizure = seizures[i-1]
        curr_seizure = seizures[i]
        interval_minutes = (curr_seizure['start_cumulative'] - prev_seizure['end_cumulative']) / 60
        
        if interval_minutes >= min_interval_minutes:
            filtered_seizures.append(curr_seizure)
    
    return filtered_seizures

def generate_preictal_segments(seizure: Dict, files_info: List[Dict], 
                                    preictal_duration_minutes: int, preictal_end_minutes: int) -> List[Tuple]:
    """为单个发作生成预发作期段"""
    segments = []
    
    # 计算预发作期的累积时间范围
    seizure_start_cumulative = seizure['start_cumulative']
    preictal_start_cumulative = seizure_start_cumulative - (preictal_duration_minutes * 60)
    preictal_end_cumulative = seizure_start_cumulative - (preictal_end_minutes * 60)
    
    # 确保预发作期不早于第一个文件开始
    first_file_start = files_info[0]['cumulative_start']
    if preictal_start_cumulative < first_file_start:
        print('有数据在第一个文件之前。')
        preictal_start_cumulative = first_file_start
    
    # 如果调整后的预发作期太短，则跳过
    if preictal_end_cumulative <= preictal_start_cumulative:
        print('无效数据。')
        return None
    
    # 遍历文件，找到与预发作期重叠的部分
    preictal_time_interval = P.closed(preictal_start_cumulative, preictal_end_cumulative)
    for file_info in files_info:
        file_interval = P.closed(file_info['cumulative_interval'][0], file_info['cumulative_interval'][1])  
        segment_interval = file_interval & preictal_time_interval 
        if not segment_interval.empty:  
            start_in_file = segment_interval.lower - file_info['cumulative_start']
            end_in_file = segment_interval.upper - file_info['cumulative_start']
            segments.append([
                file_info['filename'],
                [start_in_file, end_in_file]
            ])

    return segments

def extract_preictals_and_interictals(summary_text: str, 
                                    preictal_duration_minutes: int = PREICTAL_DURATION_MINUTES,
                                    preictal_end_minutes: int = PREICTAL_END_MINUTES,
                                    excluded_time: int = EXCLUDED_TIME) -> Dict:
    
    # 解析所有文件信息
    files_info = summarize_edf_files(summary_text)
    if not files_info:
        return {}

    total_interval = P.closed(0, files_info[-1]['cumulative_end'])

    # 提取所有发作信息
    seizures = extract_seizures_with_cumulative_time(files_info,summary_text)
    
    # 过滤间隔过短的发作
    filtered_seizures = select_valid_seizures(seizures)
    
    # 为每个有效发作生成预发作期
    preictal_segments = []
    for i, seizure in enumerate(filtered_seizures, 1):
        preictal_segments.extend(generate_preictal_segments(
            seizure, files_info, preictal_duration_minutes, preictal_end_minutes
        ))

    #生成所有发作间期信息
    seizures_time_separate = None
    for seizure in seizures:
        if seizures_time_separate is None:
            seizures_time_separate = P.closed(seizure['interval_cumulative'][0] - excluded_time*60, seizure['interval_cumulative'][1] + excluded_time*60)
        else:
            seizures_time_separate = seizures_time_separate | P.closed(seizure['interval_cumulative'][0] - excluded_time*60, seizure['interval_cumulative'][1] + excluded_time*60)
    interictals_time = total_interval - seizures_time_separate
    interictal_segments = []
    for file_info in files_info:
        interictal_intervals = interictals_time & P.closed(file_info['cumulative_interval'][0], file_info['cumulative_interval'][1])
        if not interictal_intervals.empty:
            for interictal_interval in interictal_intervals:
                interictal_segments.append([
                    file_info['filename'],
                    [interictal_interval.lower - file_info['cumulative_start'], interictal_interval.upper - file_info['cumulative_start']]
                ])

    return preictal_segments, interictal_segments

def useful_data_info_generating():
    data_info_list = []
    for i in range(NUM_PATIENT+1):
        if i in PATIENT_DELETED:
            continue
        if DATASET == 'CHB-MIT':
            summary_path = DATA_DIR / f'chb{i:02d}' / f'chb{i:02d}-summary.txt'
        if DATASET == 'Siena':
            summary_path = DATA_DIR / f'PN{i:02d}' / f'Seizures-list-PN{i:02d}.txt'

        if summary_path.exists():
            with open(summary_path, 'r', encoding='utf-8', errors='ignore') as f:
                summary_text = f.read()
            preictals, interictals = extract_preictals_and_interictals(summary_text)
            if DATASET == 'CHB-MIT':
                print(f"患者 chb{i:02d} 预发作期段数: {len(preictals)}, 发作间期段数: {len(interictals)}")
            if DATASET == 'Siena':
                print(f"患者 PN{i:02d} 预发作期段数: {len(preictals)}, 发作间期段数: {len(interictals)}")
            data_info_list.append((preictals, interictals))

        else:
            print(f"未找到文件: {summary_path}")

    return data_info_list

if __name__ == "__main__":
    data_info = useful_data_info_generating()

    with open(RESULT_DIR, 'w', encoding='utf-8') as f:
        json.dump(data_info, f, indent=2, ensure_ascii=False)

    print("✅ 数据已成功保存为 JSON 文件！")

