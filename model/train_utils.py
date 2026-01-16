 # train_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import os  # for saving paths
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np  # for mean calc if needed
from pathlib import Path  # 确保save_dir兼容Path


def plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=None):
    """
    绘制并可选保存损失/准确率曲线
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()

    # 保存PNG
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"损失/准确率曲线已保存至: {save_path}")

    plt.show()


def save_fold_metrics(metrics, fold_idx, save_dir):
    """
    保存每个fold的指标到TXT
    """
    txt_path = save_dir / f"fold_{fold_idx}_metrics.txt"
    with open(txt_path, 'w') as f:
        f.write(f"Fold {fold_idx} 最佳验证指标\n")
        f.write("=" * 30 + "\n")
        for key, value in metrics.items():
            if 'accuracy' in key or 'precision' in key or 'sensitivity' in key or 'specificity' in key or 'f1_score' in key:
                f.write(f"{key.replace('val_', '').title()}: {value:.2f}%\n")
            elif 'loss' in key:
                f.write(f"{key.replace('val_', '').title()}: {value:.4f}\n")
            else:
                f.write(f"{key.replace('val_', '').title()}: {value:.6f}\n")
    print(f"Fold {fold_idx} 指标已保存至: {txt_path}")


def train_model(model, train_loader, val_loader, fold_idx, config=None, num_epochs=None, patience=None, device=None,
                save_dir=None):
    """
    注: 早停机制改为基于F1-score
    """
    config = config or {}
    num_epochs = config.get('NUM_EPOCHS', 20) if num_epochs is None else num_epochs
    patience = config.get('PATIENCE', 5) if patience is None else patience
    device = config.get('DEVICE', 'cuda') if device is None else device
    batch_size = config.get('BATCH_SIZE', 64)
    fs = config.get('FS', 256)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.get('LEARNING_RATE', 0.001), weight_decay=1e-4)

    best_val_f1 = 0.0  # MODIFIED: 改为跟踪best_val_f1
    best_val_metrics = {}  # 最佳指标字典
    early_stop_counter = 0
    best_model_state = None

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    total_start_time = time.time()

    # 使用config参数
    BATCH_SIZE = batch_size
    FS = fs

    for epoch in range(1, num_epochs + 1):

        epoch_start_time = time.time()

        # 训练阶段
        train_loss = 0
        train_correct = 0
        train_total = 0

        model.train()
        for i, data in enumerate(train_loader, 1):
            batch_x, batch_y = data[0], data[1]
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
            optimizer.zero_grad()
            logits = model(batch_x, return_features=False)  # 新: 忽略A, attn_map
            _, predicted = torch.max(logits, 1)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total

        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # 验证阶段
        model.eval()
        val_loss = 0.0

        all_val_labels = []
        all_val_preds = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
                logits = model(batch_x, return_features=False)  # 新
                loss = criterion(logits, batch_y)
                val_loss += loss.item()

                _, predicted = torch.max(logits, 1)

                all_val_labels.extend(batch_y.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_val_labels, all_val_preds) * 100.0
        precision = precision_score(all_val_labels, all_val_preds, pos_label=1) * 100.0
        sensitivity = recall_score(all_val_labels, all_val_preds, pos_label=1) * 100.0
        specificity = recall_score(all_val_labels, all_val_preds, pos_label=0) * 100.0
        f1 = f1_score(all_val_labels, all_val_preds, pos_label=1) * 100.0
        fpr = (1 - specificity / 100.0) * 100.0  # 假阳性率（百分比）
        total_hours = len(val_loader) * BATCH_SIZE * 5 / 3600.0
        fpr_per_hour = fpr / total_hours if total_hours > 0 else 0.0

        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        # --- 结束计时并计算耗时 ---
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        total_elapsed_time = epoch_end_time - total_start_time

        # 每epoch打印train_loss, val_loss & 所有val指标
        print(f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(
            f"  Val Metrics: Acc: {val_acc:.2f}% | Prec: {precision:.2f}% | Sens: {sensitivity:.2f}% | Spec: {specificity:.2f}% | F1: {f1:.2f}% | FPR: {fpr:.2f}% | FPR/h: {fpr_per_hour:.6f}")
        print(f"  Epoch Time: {epoch_duration:.1f}s | Total Time: {total_elapsed_time:.1f}s")

        # MODIFIED: 早停检查改为基于F1
        current_metrics = {
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'val_precision': precision,
            'val_sensitivity': sensitivity,
            'val_specificity': specificity,
            'val_f1_score': f1,
            'val_fpr_h': fpr_per_hour
        }

        if f1 > best_val_f1:  # MODIFIED: 基于val_f1
            best_val_f1 = f1
            early_stop_counter = 0  # 重置计数器
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_val_metrics = current_metrics.copy()  # 更新最佳指标
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(
                    f"早停触发！当前最佳验证F1: {best_val_f1:.2f}% | 总训练耗时: {total_elapsed_time:.1f}s")  # MODIFIED: 消息改为F1
                break

    # 绘制并保存损失和准确率曲线
    loss_acc_path = save_dir / f"fold_{fold_idx}_loss_acc.png" if save_dir else None
    plot_loss_and_accuracy(train_losses, val_losses, train_accuracies, val_accuracies, save_path=loss_acc_path)
    print(
        f"Fold {fold_idx} 最佳验证F1: {best_val_f1:.2f}% | 本次训练总耗时: {total_elapsed_time:.1f}s")  # MODIFIED: 打印改为F1

    # 恢复最佳模型参数
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # 绘制并保存最佳模型的混淆矩阵
    model.eval()
    all_val_labels = []
    all_val_preds = []

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.float().to(device), batch_y.long().to(device)
            logits = model(batch_x, return_features=False)
            _, predicted = torch.max(logits, 1)
            all_val_labels.extend(batch_y.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(all_val_labels, all_val_preds)
    class_names = ['Interictal', 'Preictal']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix - Best Model\nVal F1: {best_val_f1:.2f}%')  # MODIFIED: 标题改为F1
    # plt.title(f'Confusion Matrix - Best Model\nVal Accuracy: {best_val_acc:.2f}%')  # 原标题保留注释

    # 保存CM PNG
    cm_path = save_dir / f"fold_{fold_idx}_confusion_matrix.png" if save_dir else None
    if cm_path:
        os.makedirs(os.path.dirname(cm_path), exist_ok=True)
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        print(f"混淆矩阵已保存至: {cm_path}")
    plt.show()

    # 保存每个fold的指标TXT
    if save_dir:
        save_fold_metrics(best_val_metrics, fold_idx, save_dir)

    # 在fold结束时打印全部最佳验证指标
    print(f"\n--- Fold {fold_idx} 最佳验证指标 ---")
    print(f"Val Loss: {best_val_metrics['val_loss']:.4f}")
    print(f"Val Accuracy: { best_val_metrics['val_accuracy']:.2f}%")
    print(f"Val Precision: {best_val_metrics['val_precision']:.2f}%")
    print(f"Val Sensitivity: {best_val_metrics['val_sensitivity']:.2f}%")
    print(f"Val Specificity: {best_val_metrics['val_specificity']:.2f}%")
    print(f"Val F1-Score: {best_val_metrics['val_f1_score']:.2f}%")
    print(f"Val FPR/h: {best_val_metrics['val_fpr_h']:.6f}")
    print("-----------------------------------\n")

    del all_val_labels, all_val_preds, best_model_state
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_val_metrics  # 返回最佳指标字典