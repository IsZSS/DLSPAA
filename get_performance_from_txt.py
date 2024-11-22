import csv
import glob
import re

import settings


def get_evaluation_data(file_name):
    # 打开文件
    with open(file_name, 'r') as f:
        content = f.read()

    # 用正则表达式匹配每个 epoch 对应的 Recall@5、Recall@10、NDCG@5 和 NDCG@10
    pattern = r"epoch: (\d+).*?Recall@5: ([\d.]+).*?NDCG@5: ([\d.]+).*?Recall@10: ([\d.]+).*?NDCG@10: ([\d.]+)"
    matches = re.findall(pattern, content, re.DOTALL)

    # 计算每个 epoch 的 Recall@5、Recall@10、NDCG@5 和 NDCG@10 的 sum
    sums = {}
    recall_5, ndcg_5, recall_10, ndcg_10 = 0, 0, 0, 0
    for match in matches:
        epoch = int(match[0])
        recall_5 = float(match[1])
        ndcg_5 = float(match[2])
        recall_10 = float(match[3])
        ndcg_10 = float(match[4])
        sum_values = recall_5 + ndcg_5 + recall_10 + ndcg_10
        sums[epoch] = sum_values

    # 找出 sum 最高的 epoch
    max_epoch = max(sums, key=sums.get)

    # 输出结果
    print(f"Sum of values for each epoch: {sums}")
    print(f"The epoch with the highest sum is {max_epoch} with a sum of {sums[max_epoch]}")

    # 找出 sum 最高的三个 epoch
    top_epochs = sorted(sums, key=sums.get, reverse=True)[:3]

    # 计算每个 top epoch 的平均指标
    avg_recall_5, avg_ndcg_5, avg_recall_10, avg_ndcg_10 = 0, 0, 0, 0
    for epoch in top_epochs:
        match = next(match for match in matches if int(match[0]) == epoch)
        avg_recall_5 += float(match[1])
        avg_ndcg_5 += float(match[2])
        avg_recall_10 += float(match[3])
        avg_ndcg_10 += float(match[4])
    avg_recall_5 /= 3
    avg_ndcg_5 /= 3
    avg_recall_10 /= 3
    avg_ndcg_10 /= 3

    avg_sum = avg_ndcg_5 + avg_ndcg_10 + avg_recall_5 + avg_recall_10

    with open(f"./results/max_epoch_result.csv", 'a', newline='', encoding='utf-8') as file:
        file_writer = csv.writer(file)
        # file_writer.writerow([recall_5, recall_10, ndcg_5, ndcg_10, sums[max_epoch], file_name])
        file_writer.writerow([recall_5, recall_10, ndcg_5, ndcg_10, avg_sum, file_name])


# def get_evaluation_data(file_name):
#     epoch_data = {epoch: {5: {}, 10: {}} for epoch in range(settings.epoch)}
#     max_performance = max_recall_5 = max_recall_10 = max_ndcg_5 = max_ndcg_10 = -1
#     with open(file_name) as f:
#         content = f.readlines()
#         for line in content:
#             if line.startswith('Recall@5:'):
#                 recall_5 = re.findall(r'Recall@5: ([\d.]+),', line)
#                 ndcg_5 = re.findall(r'NDCG@5: ([\d.]+),', line)
#             elif line.startswith('Recall@10:'):
#                 recall_10 = re.findall(r'Recall@10: ([\d.]+),', line)
#                 ndcg_10 = re.findall(r'NDCG@10: ([\d.]+),', line)
#         for i in range(0, settings.epoch):
#             line_data_5 = lines[8 + 4 * i].split(':')
#             line_data_10 = lines[9 + 4 * i].split(':')
#             recall_5 = float(line_data_5[1].split(',')[0])
#             recall_10 = float(line_data_10[1].split(',')[0])
#             ndcg_5 = float(line_data_5[2].split(',')[0])
#             ndcg_10 = float(line_data_10[2].split(',')[0])
#             epoch_data[i][5]['recall'] = recall_5
#             epoch_data[i][10]['recall'] = recall_10
#             epoch_data[i][5]['ndcg'] = ndcg_5
#             epoch_data[i][10]['ndcg'] = ndcg_10
#             performance = recall_5 + recall_10 + ndcg_5 + ndcg_10
#             if performance > max_performance:
#                 max_recall_5 = recall_5
#                 max_recall_10 = recall_10
#                 max_ndcg_5 = ndcg_5
#                 max_ndcg_10 = ndcg_10
#
#     with open(f"./results/max_epoch_result.csv", 'a', newline='', encoding='utf-8') as file:
#         file_writer = csv.writer(file)
#         file_writer.writerow(
#             [max_recall_5, max_recall_10, max_ndcg_5, max_ndcg_10, max_performance, file_name])


def get_average(data, k, key, epoch):
    value = []
    for i in epoch:
        value.append(data[i][k][key])
    return sum(value) / len(value)


for i, file_name in enumerate(glob.glob(f'./results/Original PHO+future1+embed60+epochs15+LR0.0001*.txt')):
    get_evaluation_data(file_name)
