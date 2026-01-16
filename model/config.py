# config.py
from pathlib import Path
import torch

config = {
    #留一法特殊变量
    # 患者与路径
    'ALL_PATIENT_IDS': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23], #CHB-MIT
    #'ALL_PATIENT_IDS': [i for i in range(18) if i not in [0, 2, 4, 8, 15]], #Seina
    # NEW: 手动指定Test患者ID（修改此值以测试不同患者）
    'TEST_PATIENT_ID': 1,
    # NEW: 每个train患者的采样比例（1/3 = 0.333）
    'SAMPLE_FRACTION': 1/3, #CHB-MIT
    # NEW: val split比例 (10% of train)
    'VAL_SPLIT': 0.1,


    'PATIENT_ID': 10 ,
    'DATA_DIR': Path("C:\EpilepsyPrediction\data_CHBMIT_1_1"),
    'SAVE_DIR': None,  # 将在main中动态设置
    'PREFIX': 'chb',


    # 信号参数 (多尺度STFT)
    'Channel_NUM': 22,
    'FS': 256,
    'N_FFT': 256,
    'HOP_LENGTH': 128,
    'WIN_LENGTH_LONG': 256,  # 长窗
    'WIN_LENGTH_SHORT': 128,  # 短窗 for 多尺度

    # 模型参数
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.002,
    'NUM_EPOCHS': 20,
    'PATIENCE': 5,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_CLASSES': 2,

    # 模型维度
    'Freq': 135,  # 3*45 bin
    'Time': 256,

}