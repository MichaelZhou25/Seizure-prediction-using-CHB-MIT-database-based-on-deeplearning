import re
import csv
import os

def extract_accuracy(file_path):
    # 尝试多种编码
    for encoding in ['utf-8', 'gbk', 'gb2312', 'cp936']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            # 如果成功读取，尝试提取 accuracy
            match = re.search(r'Accuracy:\s*([\d.]+)%', content)
            if match:
                return float(match.group(1))
            else:
                raise ValueError(f"未找到 Accuracy 行 in {file_path}")
        except UnicodeDecodeError:
            continue  # 尝试下一种编码
        except Exception as e:
            raise e  # 其他错误（如正则失败）直接抛出

    raise UnicodeDecodeError(f"无法用常见编码读取文件: {file_path}")

# 病人列表
patient_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 17, 18, 19, 20, 21, 22, 23]

# 存储结果
results = []
total_accuracy = 0.0

# 提取每个病人的 accuracy
for i in patient_list:
    filepath = rf'C:\EpilepsyPrediction\data_CHBMIT_1_1\chb_single_test_{i:02d}\test_metrics.txt'
    acc = extract_accuracy(filepath)
    results.append((i, acc))
    total_accuracy += acc

# 计算平均准确率
average_accuracy = total_accuracy / len(patient_list)

# 写入 CSV 文件
output_csv = r'C:\EpilepsyPrediction\patient_accuracies.csv'
with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['patient', 'accuracy'])  # 表头
    for patient, acc in results:
        writer.writerow([patient, acc])
    writer.writerow(['Average', average_accuracy])

print(f"结果已保存到: {output_csv}")
print(f"平均准确率: {average_accuracy:.2f}%")