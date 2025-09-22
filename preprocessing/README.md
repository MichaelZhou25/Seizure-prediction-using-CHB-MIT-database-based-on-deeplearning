# 基于CHB-MIT数据库的癫痫发作预测深度学习系统

## 项目概述

本项目是一个基于深度学习的癫痫发作预测系统，使用CHB-MIT脑电图（EEG）数据库进行训练和测试。该系统能够通过分析脑电图信号的时频特征来预测癫痫发作的发生。

## 数据库介绍

**CHB-MIT数据库**是一个广泛使用的癫痫研究数据集，包含：
- 24名儿童患者的长期EEG记录
- 每个患者包含多个.edf格式的脑电图文件
- 详细的发作时间标注信息
- 采样率为256Hz的多通道EEG数据

## 项目结构

```
preprocessing/
├── README.md                           # 项目说明文档
├── datachecking.ipynb                  # 数据质量检查
├── data_correction.ipynb               # 数据校正处理
└── data_extraction_and_splitting.ipynb # 数据提取与分割（核心模块）
```

## 核心功能模块

### 1. 数据提取与分割 (`data_extraction_and_splitting.ipynb`)

这是项目的核心模块，主要功能包括：

#### 数据时间段提取
- **预发作期（Preictal）提取**：发作前35分钟到发作前5分钟的EEG数据
- **间歇期（Interictal）提取**：排除发作及其前后缓冲时间的正常EEG数据
- **自动处理跨文件时间段**：智能处理跨越多个EDF文件的时间片段

#### 信号处理流程
1. **通道选择**：选择22个标准脑电通道
2. **带通滤波**：0.5-40Hz频带滤波，去除噪声和伪迹
3. **时间窗口分割**：5秒窗口，考虑数据平衡的动态步长
4. **短时傅里叶变换（STFT）**：提取时频域特征
5. **时频谱可视化**：生成直观的时频图展示

#### 数据平衡策略
- 计算预发作期与间歇期的时长比例K
- 预发作期使用步长 = 5 × K 秒
- 间歇期使用步长 = 5秒
- 确保两类数据样本数量平衡

### 2. 数据质量检查 (`datachecking.ipynb`)
- 检查EDF文件完整性
- 验证时间标注准确性
- 识别缺失或损坏的数据文件

### 3. 数据校正处理 (`data_correction.ipynb`)
- 处理时间戳不一致问题
- 修正文件间的时间间隙
- 标准化通道命名

## 技术特点

### 时间区间处理算法
- **区间运算**：支持复杂的时间区间并集、差集、交集运算
- **跨文件处理**：自动处理跨越多个EDF文件的连续时间段
- **累积时间计算**：建立全局时间轴，准确定位发作时间

### 信号处理技术
- **多通道并行处理**：高效处理22通道EEG数据
- **自适应滤波**：Butterworth带通滤波器
- **STFT参数优化**：
  - FFT点数：256
  - 窗口长度：256点
  - 跳跃长度：64点
  - 汉宁窗函数

### 可视化功能
- **时频谱图**：使用pcolormesh生成高质量时频图
- **对比显示**：预发作期与间歇期的直观对比
- **多色彩映射**：支持不同的色彩方案展示

## 环境要求

```python
# 主要依赖包
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.signal
import pyedflib
from pathlib import Path
```

### 安装依赖
```bash
pip install numpy torch matplotlib scipy pyedflib
```

## 使用方法

### 1. 数据准备
```python
# 设置数据路径
DATA_DIR = Path('D:/陈教授组/CHB-MIT')
PATIENT_NUM = 1  # 选择患者编号
```

### 2. 运行数据提取
```python
# 提取预发作期和间歇期数据
preictals, interictals = data_splitting()
```

### 3. 信号处理
```python
# 应用带通滤波
filtered_preictal_fragments = apply_bandpass_filter(preictal_fragments)
filtered_interictal_fragments = apply_bandpass_filter(interictal_fragments)

# 执行STFT变换
stft_preictal_fragments = apply_stft_to_fragments(filtered_preictal_fragments)
stft_interictal_fragments = apply_stft_to_fragments(filtered_interictal_fragments)
```

## 输出结果

### 数据格式
- **预发作期片段**：形状为 (22, 1280) 的数组列表，每个代表5秒的22通道EEG数据
- **间歇期片段**：同样格式的正常EEG数据片段
- **STFT结果**：形状为 (22, 129, 20) 的复数张量，表示 (通道, 频率, 时间)

### 可视化输出
- 预发作期时频谱图
- 间歇期时频谱图
- 功率谱密度对比图

## 核心算法

### 预发作期提取算法
```python
def extract_preictals_and_interictals(summary_text, 
                                    preictal_duration_minutes=35,
                                    preictal_end_minutes=5,
                                    min_seizure_interval_minutes=40,
                                    excluded_time=240):
```

### 时间区间运算
```python
def interval_difference(main_interval, remove_intervals):
def interval_union(intervals):
def interval_intersection_list(main_interval, intervals):
```

## 配置参数

| 参数名称 | 默认值 | 说明 |
|---------|--------|------|
| `preictal_duration_minutes` | 35 | 预发作期总时长（分钟） |
| `preictal_end_minutes` | 5 | 预发作期结束缓冲时间（分钟） |
| `min_seizure_interval_minutes` | 40 | 最小发作间隔（分钟） |
| `excluded_time` | 240 | 发作前后排除时间（秒） |
| `S` | 5 | 时间窗口长度（秒） |
| `CHANNEL_NUM` | 22 | 使用的EEG通道数 |

## 目标通道列表

系统使用以下22个标准EEG通道：
```python
TARGET_CHANNELS = [
    'C3-P3', 'C4-P4', 'CZ-PZ', 'F3-C3', 'F4-C4', 'F7-T7', 'F8-T8', 
    'FP1-F3', 'FP1-F7', 'FP2-F4', 'FP2-F8', 'FT10-T8', 'FT9-FT10', 
    'FZ-CZ', 'P3-O1', 'P4-O2', 'P7-O1', 'P7-T7', 'P8-O2', 'T7-FT9', 
    'T7-P7', 'T8-P8'
]
```

## 注意事项

1. **数据路径**：确保CHB-MIT数据库路径正确设置
2. **内存管理**：处理大量EEG数据时注意内存使用
3. **文件完整性**：运行前检查所有EDF文件的完整性
4. **时间标注**：确保summary文件中的时间标注准确无误

## 下一步计划



