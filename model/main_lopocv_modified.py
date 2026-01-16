# main_single_test.py
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from data_utils import process_and_save_fragments
from model import MainModel
from train_utils import train_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from config import config
import gc  # 垃圾回收

# 自定义 Dataset，按需加载数据（避免内存爆炸）
class EpilepsyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X  # 保持为 numpy array，不立即转 tensor
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 每次只加载一个样本
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def evaluate_model(model, test_loader, device, batch_size, fs):
    """评估模型在test set上的指标"""
    model.eval()
    all_test_labels = []
    all_test_preds = []
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            test_loss += loss.item()

            _, predicted = torch.max(logits, 1)
            all_test_labels.extend(batch_y.cpu().numpy())
            all_test_preds.extend(predicted.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    test_acc = accuracy_score(all_test_labels, all_test_preds) * 100.0
    test_precision = precision_score(all_test_labels, all_test_preds, pos_label=1) * 100.0
    test_sensitivity = recall_score(all_test_labels, all_test_preds, pos_label=1) * 100.0
    test_specificity = recall_score(all_test_labels, all_test_preds, pos_label=0) * 100.0
    test_f1 = f1_score(all_test_labels, all_test_preds, pos_label=1) * 100.0
    test_fpr = (1 - test_specificity / 100.0) * 100.0
    total_hours = len(test_loader) * batch_size * 5 / 3600.0  # 假设每个fragment 5s
    test_fpr_per_hour = test_fpr / total_hours if total_hours > 0 else 0.0

    test_metrics = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_acc,
        'test_precision': test_precision,
        'test_sensitivity': test_sensitivity,
        'test_specificity': test_specificity,
        'test_f1_score': test_f1,
        'test_fpr_h': test_fpr_per_hour
    }
    return test_metrics


if __name__ == "__main__":
    for i in config['ALL_PATIENT_IDS']:
        if i != 4 and i != 6 and i != 8 an d i != 17:
            continue
        config['TEST_PATIENT_ID'] = i
        test_pid = config['TEST_PATIENT_ID']
        # 动态设置SAVE_DIR（基于config）
        config['SAVE_DIR'] = Path(f"C:\\EpilepsyPrediction\\{config['PREFIX']}_single_test_{config['TEST_PATIENT_ID']:02d}")
        config['SAVE_DIR'].mkdir(parents=True, exist_ok=True)

        train_pids = [pid for pid in config['ALL_PATIENT_IDS'] if pid != test_pid]

        print(f"\n=== Single Test: Leave-Out {config['PREFIX']}-{test_pid:02d} (Train on others) ===")

        # 处理Train数据：其余患者，每个采样1/3
        print("\n处理Train Preictal 数据...")
        all_train_X_pre = []
        all_train_pre_lengths = []
        all_train_pre_sampled = []
        for pid in train_pids:
            print(f"  处理患者 {pid:02d}...")
            X_pre = process_and_save_fragments("preictal", pid, config)
            original_len = len(X_pre)
            all_train_pre_lengths.append(original_len)
            sample_size = int(original_len * config['SAMPLE_FRACTION'])
            if sample_size > 0:
                indices = np.random.choice(original_len, sample_size, replace=False)
                X_pre = X_pre[indices]
            all_train_pre_sampled.append(len(X_pre))
            all_train_X_pre.append(X_pre)
        train_X_pre = np.concatenate(all_train_X_pre, axis=0)
        print(f"Train Preictal (采样后): {train_X_pre.shape}")

        print("\n处理Train Interictal数据...")
        all_train_X_inter = []
        all_train_inter_lengths = []
        all_train_inter_sampled = []
        for pid in train_pids:
            print(f"  处理患者 {pid:02d}...")
            X_inter = process_and_save_fragments("interictal", pid, config)
            original_len = len(X_inter)
            all_train_inter_lengths.append(original_len)
            sample_size = int(original_len * config['SAMPLE_FRACTION'])
            if sample_size   > 0:
                indices = np.random.choice(original_len, sample_size, replace=False)
                X_inter = X_inter[indices]
            all_train_inter_sampled.append(len(X_inter))
            all_train_X_inter.append(X_inter)
        train_X_inter = np.concatenate(all_train_X_inter, axis=0)
        print(f"Train Interictal (采样后): {train_X_inter.shape}")

        # 平衡Train数据
        if train_X_pre.shape[0] < train_X_inter.shape[0]:
            train_X_inter = train_X_inter[:train_X_pre.shape[0]]
        else:
            train_X_pre = train_X_pre[:train_X_inter.shape[0]]
        print(f"Train数据平衡后: Pre {train_X_pre.shape[0]}, Inter {train_X_inter.shape[0]}")

        # 构建Train数据集
        train_X = np.concatenate([train_X_pre, train_X_inter], axis=0)
        train_y = np.concatenate([np.ones(train_X_pre.shape[0]), np.zeros(train_X_inter.shape[0])], axis=0)

        # 从Train中split 90% train + 10% val (stratified)
        train_split_X, val_X, train_split_y, val_y = train_test_split(
            train_X, train_y, test_size=config['VAL_SPLIT'], stratify=train_y, random_state=42
        )
        print(f"Train split: {len(train_split_X)} 样本, Val: {len(val_X)} 样本")

        # 处理Test数据：Test患者全数据（无采样）
        print(f"\n处理Test Preictal数据 ({config['PREFIX']}-{test_pid:02d}, 全数据)...")
        test_X_pre = process_and_save_fragments("preictal", test_pid, config)
        print(f"Test Preictal: {test_X_pre.shape}")

        print(f"处理Test Interictal数据 ({config['PREFIX']}-{test_pid:02d}, 全数据)...")
        test_X_inter = process_and_save_fragments("interictal", test_pid, config)
        print(f"Test Interictal: {test_X_inter.shape}")

        # 平衡Test数据
        if test_X_pre.shape[0] < test_X_inter.shape[0]:
            test_X_inter = test_X_inter[:test_X_pre.shape[0]]
        else:
            test_X_pre = test_X_pre[:test_X_inter.shape[0]]
        print(f"Test数据平衡后: Pre {test_X_pre.shape[0]}, Inter {test_X_inter.shape[0]}")

        test_X = np.concatenate([test_X_pre, test_X_inter], axis=0)
        test_y = np.concatenate([np.ones(test_X_pre.shape[0]), np.zeros(test_X_inter.shape[0])], axis=0)
        print(f"Test数据集: {test_X.shape}, 类别分布: {np.bincount(test_y.astype(int))}")

        # ✅ 创建自定义 Dataset（按需加载，节省内存）
        train_dataset = EpilepsyDataset(train_split_X, train_split_y)
        val_dataset = EpilepsyDataset(val_X, val_y)
        test_dataset = EpilepsyDataset(test_X, test_y)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

        # 初始化模型
        model = MainModel(Freq=config['Freq'], Time=config['Time'], num_classes=config['NUM_CLASSES'], config=config)

        # 训练 (使用train/val)
        print("开始训练...")
        val_metrics = train_model(model, train_loader, val_loader, fold_idx=f"single_test_{test_pid:02d}", config=config, save_dir=config['SAVE_DIR'])

        # 在best model上评估Test
        print("在Test set上评估...")
        test_metrics = evaluate_model(model, test_loader, config['DEVICE'], config['BATCH_SIZE'], config['FS'])

        # 保存Test指标到TXT
        test_txt_path = config['SAVE_DIR'] / "test_metrics.txt"
        with open(test_txt_path, 'w') as f:
            f.write(f"Single Test Patient {config['PREFIX']}-{test_pid:02d} 指标\n")
            f.write("=" * 30 + "\n")
            for key, value in test_metrics.items():
                if 'accuracy' in key or 'precision' in key or 'sensitivity' in key or 'specificity' in key or 'f1_score' in key:
                    f.write(f"{key.replace('test_', '').title()}: {value:.2f}%\n")
                elif 'loss' in key:
                    f.write(f"{key.replace('test_', '').title()}: {value:.4f}\n")
                else:
                    f.write(f"{key.replace('test_', '').title()}: {value:.6f}\n")
        print(f"Test指标已保存至: {test_txt_path}")

        # 打印Test指标
        print(f"Test Metrics: Acc: {test_metrics['test_accuracy']:.2f}% | Prec: {test_metrics['test_precision']:.2f}% | Sens: {test_metrics['test_sensitivity']:.2f}% | Spec: {test_metrics['test_specificity']:.2f}% | F1: {test_metrics['test_f1_score']:.2f}% | FPR/h: {test_metrics['test_fpr_h']:.6f}")

        # 打印Val指标（从train_model返回）
        print(f"Val Metrics: Acc: {val_metrics['val_accuracy']:.2f}% | Prec: {val_metrics['val_precision']:.2f}% | Sens: {val_metrics['val_sensitivity']:.2f}% | Spec: {val_metrics['val_specificity']:.2f}% | F1: {val_metrics['val_f1_score']:.2f}% | FPR/h: {val_metrics['val_fpr_h']:.6f}")

        # 清理内存
        del model, train_X, train_y, train_split_X, val_X, train_split_y, val_y, test_X, test_y, test_X_pre, test_X_inter, train_X_pre, train_X_inter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()  # 强制垃圾回收
        print(f"✅ 患者 {test_pid:02d} 测试完成，内存已清理。")