import datetime
import random
import numpy as np

import torch
import pickle
import time
import os

import settings
from TransRec import TransRec
from results.data_reader import print_output_to_file, calculate_average, clear_log_meta_model

device = settings.gpuId if torch.cuda.is_available() else 'cpu'
city = settings.city


def train_TransRec(train_set, test_set, h_params, vocab_size, device, run_name):
    model_path = f"./results/{run_name}_model"
    log_path = f"./results/{run_name}_log"
    meta_path = f"./results/{run_name}_meta"

    # 将 parameters 记录在 log 中
    print("parameters:", h_params)

    if os.path.isfile(f'./results/{run_name}_model'):
        try:
            os.remove(f"./results/{run_name}_meta")
            os.remove(f"./results/{run_name}_model")
            os.remove(f"./results/{run_name}_log")
        except OSError:
            pass
    file = open(log_path, 'wb')
    pickle.dump(h_params, file)
    file.close()

    # construct model
    rec_model = TransRec(
        vocab_size=vocab_size,
        f_embed_size=h_params['embed_size'],
        num_encoder_layers=h_params['tfp_layer_num'],
        num_lstm_layers=h_params['lstm_layer_num'],
        num_heads=h_params['head_num'],
        forward_expansion=h_params['expansion'],
        dropout_p=h_params['dropout']
    )

    rec_model = rec_model.to(device)

    # continue with previous training
    start_epoch = 0
    if os.path.isfile(model_path):
        rec_model.load_state_dict(torch.load(model_path))
        rec_model.train()

        # load training epoch
        meta_file = open(meta_path, "rb")
        start_epoch = pickle.load(meta_file) + 1
        meta_file.close()

    params = list(rec_model.parameters())

    optimizer = torch.optim.Adam(params, lr=h_params['lr'])

    loss_dict, recalls, ndcgs, maps = {}, {}, {}, {}

    if settings.enable_curriculum_learning:
        update_count = 0
        len_train_set = settings.curriculum_num * len(train_set)  # 在训练完多个 epoch 后 CL_weight 达到最大值

    for epoch in range(start_epoch, h_params['epoch']):
        begin_time = time.time()
        total_loss = 0.
        j = 0
        for sample in train_set:
            j += 1
            if settings.enable_curriculum_learning:
                update_count += 1
                settings.CL_weight = min(settings.CL_max_weight, settings.CL_max_weight * update_count / len_train_set)

            sample_to_device = []
            # 数据格式 [(seq1)[((features)[poi_seq],[cat_seq],[user_seq],[hour_seq],[day_seq])],[(seq2)],...]
            for seq in sample:
                features = torch.tensor(seq).to(device)
                sample_to_device.append(features)

            loss, _ = rec_model(sample_to_device)
            total_loss += loss.detach().cpu()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if city == 'PHO':
                if j % 100 == 0:
                    rec_model.print_parameters(epoch=epoch)
            else:
                if j % 1000 == 0:
                    rec_model.print_parameters(epoch=epoch)

        # test
        # if i%10==0:
        recall, ndcg, map = test_TransRec(test_set, rec_model)
        recalls[epoch] = recall
        ndcgs[epoch] = ndcg
        maps[epoch] = map

        # record avg loss
        avg_loss = total_loss / len(train_set)
        loss_dict[epoch] = avg_loss
        print(f"epoch: {epoch}; average loss: {avg_loss}, time taken: {int(time.time() - begin_time)}s")
        # save model
        torch.save(rec_model.state_dict(), model_path)
        # save last epoch
        meta_file = open(meta_path, 'wb')
        pickle.dump(epoch, meta_file)
        meta_file.close()

        # early stop
        past_10_loss = list(loss_dict.values())[-11:-1]
        if len(past_10_loss) > 10 and abs(total_loss - np.mean(past_10_loss)) < h_params['loss_delta']:
            print(f"***Early stop at epoch {epoch}***")
            break

        file = open(log_path, 'wb')
        pickle.dump(loss_dict, file)
        pickle.dump(recalls, file)
        pickle.dump(ndcgs, file)
        pickle.dump(maps, file)
        file.close()

    print("============================")


def test_TransRec(test_set, rec_model, ks=[1, 5, 10]):
    def calc_recall(labels, preds, k):
        return torch.sum(torch.sum(labels == preds[:, :k], dim=1)) / labels.shape[0]

    def calc_ndcg(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        ndcg = 1 / torch.log2(exist_pos + 1)
        return torch.sum(ndcg) / labels.shape[0]

    def calc_map(labels, preds, k):
        exist_pos = (preds[:, :k] == labels).nonzero()[:, 1] + 1
        map = 1 / exist_pos
        return torch.sum(map) / labels.shape[0]

    preds, labels = [], []
    total_test_loss = 0.
    for sample in test_set:
        sample_to_device = []
        for seq in sample:
            features = torch.tensor(seq).to(device)
            sample_to_device.append(features)

        pred, label, test_loss = rec_model.predict(sample_to_device)
        total_test_loss += test_loss.detach().cpu()
        preds.append(pred.detach())
        labels.append(label.detach())
    preds = torch.stack(preds, dim=0)
    labels = torch.unsqueeze(torch.stack(labels, dim=0), 1)

    avg_test_loss = total_test_loss / len(test_set)
    print(f"average test loss: {avg_test_loss}")

    recalls, NDCGs, MAPs = {}, {}, {}
    for k in ks:
        recalls[k] = calc_recall(labels, preds, k)
        NDCGs[k] = calc_ndcg(labels, preds, k)
        MAPs[k] = calc_map(labels, preds, k)
        print(f"Recall @{k} : {recalls[k]},\tNDCG@{k} : {NDCGs[k]},\tMAP@{k} : {MAPs[k]}")

    return recalls, NDCGs, MAPs


if __name__ == '__main__':
    # 输出当前时间
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print("Datetime of now：", now_str)

    # get parameters
    h_params = {
        'expansion': 4,
        'lr': settings.lr,
        'epoch': settings.epoch,
        'loss_delta': 1e-3}

    # read training data
    file = open(f"./processed_data/{city}_train", 'rb')
    train_set = pickle.load(file)
    file = open(f"./processed_data/{city}_valid", 'rb')
    valid_set = pickle.load(file)

    # read meta data
    file = open(f"./processed_data/{city}_meta", 'rb')
    meta = pickle.load(file)
    file.close()

    vocab_size = {"POI": torch.tensor(len(meta["POI"])).to(device),
                  "cat": torch.tensor(len(meta["cat"])).to(device),
                  "user": torch.tensor(len(meta["user"])).to(device),
                  "hour": torch.tensor(len(meta["hour"])).to(device),
                  "day": torch.tensor(len(meta["day"])).to(device)}

    print(f'current+{city}+_+{vocab_size["cat"]}')
    # adjust specific parameters for each city
    if city == 'SIN':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = settings.tfp_layer_num
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1
    elif city == 'NYC':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = settings.tfp_layer_num
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.1
        h_params['head_num'] = 1
    elif city == 'PHO':
        h_params['embed_size'] = settings.embed_size
        h_params['tfp_layer_num'] = 4
        h_params['lstm_layer_num'] = 2
        h_params['dropout'] = 0.2
        h_params['head_num'] = 1

    # create output folder
    if not os.path.isdir('./results'):
        os.mkdir("./results")

    print(f'Current GPU {settings.gpuId}')
    for run_num in range(1, 1 + settings.run_times):
        run_name = f'{settings.output_file_name} {run_num}'
        print(run_name)  # 正式训练前打印一遍 name，提高 log 可读性

        train_TransRec(train_set, valid_set, h_params, vocab_size, device, run_name=run_name)
        print_output_to_file(settings.output_file_name, run_num)

        t = random.randint(1, 9)
        print(f"sleep {t} seconds")
        time.sleep(t)

        clear_log_meta_model(settings.output_file_name, run_num)
    calculate_average(settings.output_file_name, settings.run_times)
