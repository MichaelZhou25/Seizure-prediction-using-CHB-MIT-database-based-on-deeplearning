# main_all.py
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from data_utils import process_and_save_fragments
from model import MainModel
from train_utils import train_model
from config import config

# -------------------------------
# 配置参数（所有参数集中在此字典，支持全局修改）
# -------------------------------
# 动态设置SAVE_DIR（基于config）
config['SAVE_DIR'] = Path(r"/mix5fold/tgcn")
config['SAVE_DIR'].mkdir(parents=True, exist_ok=True)

print(f"使用设备: {config['DEVICE']}")
print(f"配置加载完成: {len(config)} 个参数")
print(f"混合患者: CHB-{', '.join(f'{pid:02d}' for pid in config['ALL_PATIENT_IDS'])}")
print(f"每个患者采样比例: {config['SAMPLE_FRACTION']:.3f} (1/3)")

if __name__ == "__main__":
    # 收集所有患者的发作前数据（每个患者下采样1/3）
    all_X_preictal = []
    all_pre_lengths = []  # 记录每个患者的原始长度
    all_pre_sampled_lengths = []  # 记录采样后长度
    for pid in config['ALL_PATIENT_IDS']:
        print(f"\n处理患者 {pid:02d} 的 preictal 数据...")
        X_pre = process_and_save_fragments("preictal", pid, config)
        original_len = len(X_pre)
        all_pre_lengths.append(original_len)

        # NEW: 每个患者下采样到1/3
        sample_size = int(original_len * config['SAMPLE_FRACTION'])
        if sample_size > 0:
            indices_pre = np.random.choice(original_len, sample_size, replace=False)
            X_pre = X_pre[indices_pre]
        all_pre_sampled_lengths.append(len(X_pre))
        all_X_preictal.append(X_pre)
        print(f"  原始样本: {original_len} -> 采样后: {len(X_pre)}")

    X_preictal = np.concatenate(all_X_preictal, axis=0)
    print(f"所有患者 Preictal 数据维度 (采样后): {X_preictal.shape}")
    print(f"每个患者 Preictal 样本数 (原始/采样): {list(zip(all_pre_lengths, all_pre_sampled_lengths))}")

    # 收集所有患者的发作间期数据（每个患者下采样1/3）
    all_X_interictal = []
    all_inter_lengths = []  # 记录每个患者的原始长度
    all_inter_sampled_lengths = []  # 记录采样后长度
    for pid in config['ALL_PATIENT_IDS']:
        print(f"\n处理患者 {pid:02d} 的 interictal 数据...")
        X_inter = process_and_save_fragments("interictal", pid, config)
        original_len = len(X_inter)
        all_inter_lengths.append(original_len)

        # NEW: 每个患者下采样到1/3
        sample_size = int(original_len * config['SAMPLE_FRACTION'])
        if sample_size > 0:
            indices_inter = np.random.choice(original_len, sample_size, replace=False)
            X_inter = X_inter[indices_inter]
        all_inter_sampled_lengths.append(len(X_inter))
        all_X_interictal.append(X_inter)
        print(f"  原始样本: {original_len} -> 采样后: {len(X_inter)}")

    X_interictal = np.concatenate(all_X_interictal, axis=0)
    print(f"所有患者 Interictal 数据维度 (采样后): {X_interictal.shape}")
    print(f"每个患者 Interictal 样本数 (原始/采样): {list(zip(all_inter_lengths, all_inter_sampled_lengths))}")

    # 保证样本数1比1（全局平衡）
    if X_preictal.shape[0] < X_interictal.shape[0]:
        print("Preictal 样本数少于 Interictal")
        X_interictal = X_interictal[:X_preictal.shape[0]]
        print(f"已裁剪 Interictal 至 {X_interictal.shape[0]} 个样本")
    else:
        print("Interictal 样本数少于或等于 Preictal")
        X_preictal = X_preictal[:X_interictal.shape[0]]
        print(f"已裁剪 Preictal 至 {X_preictal.shape[0]} 个样本")

    # 构建数据集和标签
    X = np.concatenate([X_preictal, X_interictal], axis=0)
    y = np.concatenate([
        np.ones(X_preictal.shape[0]),
        np.zeros(X_interictal.shape[0])
    ], axis=0)

    print(f"特征 X 形状 (最终): {X.shape}")
    print(f"标签 y 形状: {y.shape}")
    print(f"类别分布: {np.bincount(y.astype(int))}")

    # 五折交叉验证（收集指标）
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_metrics_list = []  # 收集每个fold的最佳指标

    print(f"\n开始五折交叉验证...")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n--- Fold {fold_idx}/5 ---")

        # 准备数据
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        # 创建DataLoader（使用config['BATCH_SIZE']）
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['BATCH_SIZE'], shuffle=False)

        print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")

        # 初始化模型 (传入config)
        model = MainModel(Freq=config['Freq'], Time=config['Time'], num_classes=config['NUM_CLASSES'], config=config)

        # 训练模型 (获取指标字典)
        print("开始训练...")
        fold_metrics = train_model(model, train_loader, val_loader, fold_idx=fold_idx, config=config,
                                   save_dir=config['SAVE_DIR'])
        fold_metrics_list.append(fold_metrics)

        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算并打印均值指标
    mean_metrics = {key: np.mean([m[key] for m in fold_metrics_list]) for key in fold_metrics_list[0].keys()}
    print(f"\n=== 五折交叉验证结果 ===")
    print(f"患者混合: CHB-{', '.join(f'{pid:02d}' for pid in config['ALL_PATIENT_IDS'])}")
    print(f"平均指标:")
    print(f"Mean Val Accuracy: {mean_metrics['val_accuracy']:.2f}%")
    print(f"Mean Val Precision: {mean_metrics['val_precision']:.2f}%")
    print(f"Mean Val Sensitivity: {mean_metrics['val_sensitivity']:.2f}%")
    print(f"Mean Val Specificity: {mean_metrics['val_specificity']:.2f}%")
    print(f"Mean Val F1-Score: {mean_metrics['val_f1_score']:.2f}%")
    print(f"Mean Val FPR/h: {mean_metrics['val_fpr_h']:.6f}")

    # 保存均值到txt文件
    txt_path = config['SAVE_DIR'] / "fold_results_mean.txt"
    with open(txt_path, 'w') as f:
        f.write(f"患者混合 CHB-{', '.join(f'{pid:02d}' for pid in config['ALL_PATIENT_IDS'])} - 五折交叉验证均值指标\n")
        f.write(f"(每个患者采样 {config['SAMPLE_FRACTION']:.3f} 比例，下采样后总样本: {len(y)})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean Val Accuracy: {mean_metrics['val_accuracy']:.2f}%\n")
        f.write(f"Mean Val Precision: {mean_metrics['val_precision']:.2f}%\n")
        f.write(f"Mean Val Sensitivity: {mean_metrics['val_sensitivity']:.2f}%\n")
        f.write(f"Mean Val Specificity: {mean_metrics['val_specificity']:.2f}%\n")
        f.write(f"Mean Val F1-Score: {mean_metrics['val_f1_score']:.2f}%\n")
        f.write(f"Mean Val FPR/h: {mean_metrics['val_fpr_h']:.6f}\n")

    print(f"均值指标已保存至: {txt_path}")

    # NEW: 显式清理大数组以释放内存
    del X, y, X_preictal, X_interictal
    if torch.cuda.is_available():
        torch.cuda.empty_cache()