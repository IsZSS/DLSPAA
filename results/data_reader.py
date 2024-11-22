import os
import pickle
import shutil

import pandas as pd

import settings

csv_template_path = f"./results/template.csv"


def clear_log_meta_model(output_file_name, run_num):
    """
    自动清理本次运行生成的 log，meta，model 文件
    """
    run_name = f"{output_file_name} {run_num}"
    try:
        os.remove(f"./results/{run_name}_log")
    except OSError:
        pass
    try:
        os.remove(f"./results/{run_name}_meta")
    except OSError:
        pass
    try:
        os.remove(f"./results/{run_name}_model")
    except OSError:
        pass


def calculate_average(output_file_name, run_count):
    # 跨不同 epoch 计算最优指标
    csv_path = f"./results/{output_file_name}.csv"
    csv_calculate_average(output_file_name, csv_path, run_count)

    # 依据同一 epoch 计算最优指标
    if settings.enable_cal_epoch_metrics:
        single_epoch_csv_path = f"./results/EPOCH_{output_file_name}.csv"
        csv_calculate_average(output_file_name, single_epoch_csv_path, run_count)


def csv_calculate_average(output_file_name, file_path, run_count):
    # load the csv file
    df = pd.read_csv(file_path)

    # Extract the first `num_cols` columns of the last `n` rows
    last_n_rows = df.iloc[-run_count:, :5]

    # Calculate the average of each column
    averages = last_n_rows.mean()

    # append the averages to the end of the csv file
    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        f.write('\n')
        f.write(','.join(averages.astype(str)))
        f.write(f',{output_file_name}')
        f.write('\n')


def print_max_single_epoch_to_csv(output_file_name, run_num):
    epoch_data = {epoch: {5: {}, 10: {}} for epoch in range(settings.epoch)}
    max_recall_5 = max_recall_10 = max_ndcg_5 = max_ndcg_10 = max_epoch_performance = -1

    run_name = f"{output_file_name} {run_num}"
    with open(f"./results/{run_name}.txt") as f:
        lines = f.readlines()
        for i in range(0, settings.epoch):
            line_data_5 = lines[8 + 4 * i].split(':')
            line_data_10 = lines[9 + 4 * i].split(':')
            recall_5 = float(line_data_5[1].split(',')[0])
            recall_10 = float(line_data_10[1].split(',')[0])
            ndcg_5 = float(line_data_5[2].split(',')[0])
            ndcg_10 = float(line_data_10[2].split(',')[0])
            epoch_data[i][5]['recall'] = recall_5
            epoch_data[i][10]['recall'] = recall_10
            epoch_data[i][5]['ndcg'] = ndcg_5
            epoch_data[i][10]['ndcg'] = ndcg_10

            epoch_performance_sum = recall_5 + recall_10 + ndcg_5 + ndcg_10
            if epoch_performance_sum > max_epoch_performance:
                max_recall_5 = recall_5
                max_recall_10 = recall_10
                max_ndcg_5 = ndcg_5
                max_ndcg_10 = ndcg_10
                max_epoch_performance = epoch_performance_sum

    max_from_single_epoch = [max_recall_5, max_recall_10, max_ndcg_5, max_ndcg_10, max_epoch_performance, run_name]

    csv_epoch_path = f"./results/EPOCH_{output_file_name}.csv"
    if run_num == 1:  # 第一次运行完成时创建针对当前任务的 csv 文件（run_num 从 1 开始计数）
        shutil.copyfile(csv_template_path, csv_epoch_path)
    with open(csv_epoch_path, 'a', newline='', encoding='utf-8') as f:
        f.write('\n')
        f.write(','.join([str(x) for x in max_from_single_epoch]))


def print_output_to_file(output_file_name, run_num):
    run_name = f"{output_file_name} {run_num}"

    log_name = f"./results/{run_name}_log"
    file = open(log_name, 'rb')

    outfile = open(f"./results/{run_name}.txt", 'a')

    epoch_data = {epoch: {1: {}, 5: {}, 10: {}} for epoch in range(settings.epoch)}

    for i in range(4):
        data = pickle.load(file)
        if i == 1:  # recall
            max_local_recall = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, recalls in data.items():
                for k, recall in recalls.items():
                    recall = recall.item()
                    epoch_data[epoch][k]['recall'] = recall
                    if max_local_recall[k][1] < recall:
                        max_local_recall[k] = (epoch, recall)

        elif i == 2:  # ndcg
            max_local_ndcg = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, ndcgs in data.items():
                for k, ndcg in ndcgs.items():
                    ndcg = ndcg.item()
                    epoch_data[epoch][k]['ndcg'] = ndcg
                    if max_local_ndcg[k][1] < ndcg:
                        max_local_ndcg[k] = (epoch, ndcg)
        elif i == 3:  # map
            max_local_map = {1: (0, 0.), 5: (0, 0.), 10: (0, 0.)}
            for epoch, maps in data.items():
                for k, map in maps.items():
                    map = map.item()
                    epoch_data[epoch][k]['map'] = map
                    if max_local_map[k][1] < map:
                        max_local_map[k] = (epoch, map)

    outfile.write(f"{log_name}\n")
    outfile.write(f"recall: {max_local_recall}\n")
    outfile.write(f"ndcg: {max_local_ndcg}\n")
    outfile.write(f"map: {max_local_map}\n")
    outfile.write('--------------\n\n')

    for epoch, data in epoch_data.items():
        outfile.write(f"epoch: {epoch};\n")
        for k, value in data.items():
            outfile.write(f"Recall@{k}: {value['recall']}, NDCG@{k}: {value['ndcg']}, MAP@{k}: {value['map']}\n")
    outfile.write("=============================")

    outfile.close()
    file.close()

    max_performance = max_local_recall[5][1] + max_local_recall[10][1] + max_local_ndcg[5][1] + max_local_ndcg[10][1]

    max_from_different_epochs = [max_local_recall[5][1], max_local_recall[10][1], max_local_ndcg[5][1],
                                 max_local_ndcg[10][1], max_performance, run_name]

    # append the performance to the end of the csv file
    csv_path = f"./results/{output_file_name}.csv"
    if run_num == 1:  # 第一次运行完成时创建针对当前任务的 csv 文件（run_num 从 1 开始计数）
        shutil.copyfile(csv_template_path, csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as f:
        f.write('\n')
        f.write(','.join([str(x) for x in max_from_different_epochs]))

    if settings.enable_cal_epoch_metrics:
        print_max_single_epoch_to_csv(output_file_name, run_num)
