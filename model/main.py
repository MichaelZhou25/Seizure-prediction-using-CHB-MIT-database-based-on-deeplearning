# main.py
import numpy as np
import torch
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from data_utils import process_and_save_fragments
from model import MainModel
from train_utils import train_model
from config import config
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
'''
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
'''

# 动态设置SAVE_DIR（基于config）
config['SAVE_DIR'] = Path(f"D:\SeizurePrediction\EpilepsyPrediction\chb{config['PATIENT_ID']:02d}")
config['SAVE_DIR'].mkdir(parents=True, exist_ok=True)

print(f"使用设备: {config['DEVICE']}")
print(f"配置加载完成: {len(config)} 个参数")

if __name__ == "__main__":
    prefix = config['PREFIX']
    tsne_save_dir = os.path.join(config['SAVE_DIR'],'tsne_plots')
    os.makedirs(tsne_save_dir, exist_ok=True)  # 确保文件夹存在
    # 处理发作前数据 (传入config)
    X_preictal = process_and_save_fragments("preictal", config['PATIENT_ID'], config)

    # 处理发作间期数据
    X_interictal = process_and_save_fragments("interictal", config['PATIENT_ID'], config)

    # 保证样本数1比1
    if X_preictal.shape[0] < X_interictal.shape[0]:
        print("Preictal 样本数少于 Interictal")
        X_interictal = X_interictal[:X_preictal.shape[0]]
        print(f"已裁剪 Interictal 至 {X_interictal.shape[0]} 个样本")
    else:
        print("Interictal 样本数少于或等于 Preictal")
        X_preictal = X_preictal[:X_interictal.shape[0]]
        print(f"已裁剪 Preictal 至 {X_preictal.shape[0]} 个样本")

    print(f"患者编号: {prefix}-{config['PATIENT_ID']:02d}")
    print(f"Preictal数据维度: {X_preictal.shape}")
    print(f"Interictal数据维度: {X_interictal.shape}")

    # 构建数据集和标签
    X = np.concatenate([X_preictal, X_interictal], axis=0)
    y = np.concatenate([
        np.ones(X_preictal.shape[0]),
        np.zeros(X_interictal.shape[0])
    ], axis=0)

    print(f"特征 X 形状: {X.shape}")
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
        model = MainModel(Freq=135, Time=config['Time'], num_classes=config['NUM_CLASSES'], config=config)
        model = model.to(config['DEVICE'])
        model.train()

        # 3. 提取“初始化”特征（未训练）
        model.eval()
        all_features_init = []
        all_labels_init = []

        with torch.no_grad():
            i = 0
            for batch_x, batch_y in train_loader:
                i += 1
                batch_x = batch_x.to(config['DEVICE'])  # 假设你有 device = 'cuda' or 'cpu'
                _, features = model(batch_x, return_features=True)  # features: (batch_size, 64)
                all_features_init.append(features.cpu().numpy())
                all_labels_init.append(batch_y.cpu().numpy())
                if i == 2:
                    break

        # 拼接所有 batch
        features_init = np.concatenate(all_features_init, axis=0)  # (n_train, 64)
        y_train_np = np.concatenate(all_labels_init, axis=0)  # (n_train,)

        # 4. 标准化 + TSNE（初始化）
        scaler_init = StandardScaler()
        features_init_scaled = scaler_init.fit_transform(features_init)
        tsne_init = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        features_init_tsne = tsne_init.fit_transform(features_init_scaled)

        # 5. 可视化“初始化”状态
        plt.figure(figsize=(8, 6))
        plt.scatter(features_init_tsne[:, 0], features_init_tsne[:, 1], c=y_train_np, cmap='viridis', alpha=0.7)
        plt.title(f'Fold {fold_idx} - TSNE (Before Training)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(label='Label (1=Preictal, 0=Interictal)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(tsne_save_dir, f'fold_{fold_idx}_tsne_init.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # 训练模型 (获取指标字典)
        print("开始训练...")
        fold_metrics = train_model(model, train_loader, val_loader, fold_idx=fold_idx, config=config, save_dir=config['SAVE_DIR'])
        fold_metrics_list.append(fold_metrics)

        # 8. 提取“训练后”特征 → 按 batch 提取
        model.eval()
        all_features_trained = []
        all_labels_trained = []

        with torch.no_grad():
            i = 0
            for batch_x, batch_y in train_loader:
                i += 1
                batch_x = batch_x.to(config['DEVICE'])
                _, features = model(batch_x, return_features=True)
                all_features_trained.append(features.cpu().numpy())
                all_labels_trained.append(batch_y.cpu().numpy())
                if i == 2:
                    break

        features_trained = np.concatenate(all_features_trained, axis=0)  # (n_train, 64)
        y_train_np_trained = np.concatenate(all_labels_trained, axis=0)  # (n_train,)

        # 9. 标准化 + TSNE（训练后）
        scaler_trained = StandardScaler()
        features_trained_scaled = scaler_trained.fit_transform(features_trained)
        tsne_trained = TSNE(n_components=2, random_state=0, perplexity=30, n_iter=1000)
        features_trained_tsne = tsne_trained.fit_transform(features_trained_scaled)

        # 10. 保存“训练后”状态图
        plt.figure(figsize=(8, 6))
        plt.scatter(features_trained_tsne[:, 0], features_trained_tsne[:, 1], c=y_train_np_trained, cmap='viridis',
                    alpha=0.7)
        plt.title(f'Fold {fold_idx} - TSNE (After Training)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(label='Label (1=Preictal, 0=Interictal)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(tsne_save_dir, f'fold_{fold_idx}_tsne_trained.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # 11. 保存对比图
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        axes[0].scatter(features_init_tsne[:, 0], features_init_tsne[:, 1], c=y_train_np, cmap='viridis', alpha=0.7)
        axes[0].set_title(f'Fold {fold_idx} - Before Training')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')

        axes[1].scatter(features_trained_tsne[:, 0], features_trained_tsne[:, 1], c=y_train_np_trained, cmap='viridis',
                        alpha=0.7)
        axes[1].set_title(f'Fold {fold_idx} - After Training')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')

        plt.suptitle(f'Fold {fold_idx} - Feature Space Evolution')
        plt.tight_layout()
        plt.savefig(os.path.join(tsne_save_dir, f'fold_{fold_idx}_tsne_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()


        # 清理GPU内存
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 计算并打印均值指标
    mean_metrics = {key: np.mean([m[key] for m in fold_metrics_list]) for key in fold_metrics_list[0].keys()}
    print(f"\n=== 五折交叉验证结果 ===")
    print(f"患者 {prefix}{config['PATIENT_ID']:02d}")
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
        f.write(f"患者 {prefix}{config['PATIENT_ID']:02d} - 五折交叉验证均值指标\n")
        f.write("=" * 50 + "\n")
        f.write(f"Mean Val Accuracy: {mean_metrics['val_accuracy']:.2f}%\n")
        f.write(f"Mean Val Precision: {mean_metrics['val_precision']:.2f}%\n")
        f.write(f"Mean Val Sensitivity: {mean_metrics['val_sensitivity']:.2f}%\n")
        f.write(f"Mean Val Specificity: {mean_metrics['val_specificity']:.2f}%\n")
        f.write(f"Mean Val F1-Score: {mean_metrics['val_f1_score']:.2f}%\n")
        f.write(f"Mean Val FPR/h: {mean_metrics['val_fpr_h']:.6f}\n")

    print(f"均值指标已保存至: {txt_path}")