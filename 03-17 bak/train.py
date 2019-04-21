#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from config import opt
from my_class import DataLoaderPool, FaceRecognitionNetPool
from evaluation import test


def show_plot(train_iter_index, train_loss_list, val_iter_index, val_loss_list, val_acc_list):
    plt.figure('Loss and Accuracy', figsize=(12, 6.75))
    plt.subplot(1, 2, 1)
    plt.plot(train_iter_index, train_loss_list)
    plt.plot(val_iter_index, val_loss_list)
    plt.subplot(1, 2, 2)
    plt.plot(val_iter_index, val_acc_list)

    time_str = time.strftime('%m%d%H%M%S')
    save_name = '%s_Loss and Accuracy.png' % time_str
    fig_save_path = os.path.join(opt.log_dir, save_name)
    plt.savefig(fig_save_path)
    plt.ioff()
    plt.show()


def cal_best_m(data_num, AP_avg, AN_avg, distance_AP_list, distance_AN_list):
    n = 100
    d = (AN_avg - AP_avg) / n
    right_list = []
    for i in range(n):
        m = AP_avg + i * d
        right_num = 0
        for distance_AP in distance_AP_list:
            if distance_AP < m:
                right_num += 1
        for distance_AN in distance_AN_list:
            if distance_AN > m:
                right_num += 1
        acc = right_num / (data_num * 2)
        right_list.append(acc)

    best_acc = max(right_list)
    best_index = right_list.index(best_acc)
    best_m = AP_avg + best_index * d
    return best_m, best_acc


def show_distance(distance_AP_list, distance_AN_list, show_plot_epoch, epoch):
    data_num = len(distance_AP_list)
    x = range(data_num)
    AP_avg = np.average(distance_AP_list)
    AN_avg = np.average(distance_AN_list)
    best_m, best_acc = cal_best_m(data_num, AP_avg, AN_avg, distance_AP_list, distance_AN_list)

    plt.figure('distance distribute', figsize=(12, 6.75))
    # 距离均值
    plt.scatter(x, distance_AP_list, c='b', s=6, alpha=0.5)  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.scatter(x, distance_AN_list, c='r', s=6, alpha=0.5)  # s为size，按每个点的坐标绘制，alpha为透明度
    plt.plot((0, data_num - 1), (AP_avg, AP_avg), 'b')
    plt.text(data_num / 2, AP_avg, format(AP_avg, '0.2f'), fontsize=9)
    plt.plot((0, data_num - 1), (AN_avg, AN_avg), 'r')
    plt.text(data_num / 2, AN_avg, format(AN_avg, '0.2f'), fontsize=9)
    # 最好区分度
    plt.plot((0, data_num), (best_m, best_m), 'g')
    plt.text(0, best_m, 'best m:%s, best_acc:%s' % (format(best_m, '0.2f'), format(best_acc, '0.2f')),
             fontsize=9)
    # 保存图
    time_str = time.strftime('%m%d%H%M%S')
    save_name = '%s_distance_distribute_%s.png' % (time_str, str(epoch))
    fig_save_path = os.path.join(opt.log_dir, save_name)
    plt.savefig(fig_save_path)
    if show_plot_epoch:
        plt.ioff()
        plt.show()
    plt.close()


def triplet_select(net, inputs0, inputs1, inputs2):
    # 准备变量
    ori_inputs0, ori_inputs1, ori_inputs2 = inputs0, inputs1, inputs2
    tar_inputs2 = ori_inputs2
    # 计算当前网络编码
    net.eval()
    inputs0, inputs1, inputs2 = Variable(inputs0), Variable(inputs1), Variable(inputs2)
    outputs0, outputs1, outputs2 = net(inputs0), net(inputs1), net(inputs2)
    # 统计
    input_num = len(inputs0)
    # 编码结果脱离
    encodingA_list = outputs0.data.numpy()
    encodingP_list = outputs1.data.numpy()
    encodingN_list = outputs2.data.numpy()

    # for i in range(input_num):
    #     print('本来的负样本', ori_inputs2[i])

    # 更改负样本
    for i in range(input_num):
        encodingA = encodingA_list[i]
        encodingP = encodingP_list[i]
        distance_AN_list = []
        distance_AP = np.linalg.norm(encodingA - encodingP)
        for j in range(input_num):
            distance_AN = np.linalg.norm(encodingA - encodingN_list[j])
            distance_AN_list.append(distance_AN)
        # print(i, 'AP', distance_AP, 'AN', distance_AN_list)
        sorted_AN_list = sorted(distance_AN_list)
        # print(i, 'AP', distance_AP, 'AN', sorted_AN_list)
        select_AN = sorted_AN_list[-1]
        for AN in sorted_AN_list:
            if AN > distance_AP:
                select_AN = AN
                break
        AN_index = distance_AN_list.index(select_AN)
        # print(i, 'AP', distance_AP, 'AN', select_AN, 'AN_index', AN_index)
        tar_inputs2[i] = ori_inputs2[AN_index]
        # print('满足semi-hard的负样本', ori_inputs2[AN_index])
        # print('现在的负样本', tar_inputs2[i])

    return ori_inputs0, ori_inputs1, tar_inputs2  # 把原始负样本替换成了满足semi-hard的负样本


def vector_normalization(vector):
    vector_shape = np.shape(vector)
    zero_np = np.zeros(vector_shape)
    zero_vector = torch.from_numpy(zero_np).type_as(vector)
    new_vector = torch.from_numpy(zero_np).type_as(vector)
    vector_length = F.pairwise_distance(vector, zero_vector)

    for i in range(vector_shape[0]):
        length = vector_length[i]
        new_vector[i] = vector[i] / length
    return new_vector


def train():
    # ------------------------------------  显示程序设定参数  -------------------------------------
    opt._print_opt()

    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(opt)
    train_loader = data_pool.select_dataloader(data_type='train')
    valid_loader = data_pool.select_dataloader(data_type='valid')
    print('load data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------

    net_name = opt.train_net_name
    net_pool = FaceRecognitionNetPool(opt)  # 模型选择类
    net = net_pool.select_model(net_name)  # 调用网络
    print('load net done !')

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

    criterion = nn.TripletMarginLoss(margin=opt.margin)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2,
                                                     verbose=True, threshold=0.005, threshold_mode='rel',
                                                     cooldown=0, min_lr=0, eps=1e-08)  # 设置学习率下降策略

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    print('train start ------------------------------------------------')
    time_str = time.strftime('%H时%M分%S秒')
    print(time_str)
    train_iter_index = []
    train_loss_list = []
    val_iter_index = []
    val_loss_list = []
    val_acc_list = []
    iteration_number = 0
    best_val_loss = 100
    best_val_acc = 0
    now_best_net_save_path = None

    for epoch in range(opt.max_epoch):
        loss_sigma = 0
        net.train()  # 训练模式

        for i, data in tqdm((enumerate(train_loader))):

            # 获取图片和标签
            inputs0, inputs1, inputs2 = data

            inputs0, inputs1, inputs2 = Variable(inputs0), Variable(inputs1), Variable(inputs2)
            outputs0, outputs1, outputs2 = net(inputs0), net(inputs1), net(inputs2)

            if opt.train_harder:

                # online triplet select
                # Choose the hard negatives
                d_p = F.pairwise_distance(outputs0, outputs1, 2)
                d_n = F.pairwise_distance(outputs0, outputs2, 2)
                hard_negatives = (d_n - d_p < opt.margin).data.numpy().flatten()
                hard_triplets = np.where(hard_negatives == 1)
                if len(hard_triplets[0]) == 0:
                    continue
                out_selected_a = Variable(torch.from_numpy(outputs0.data.numpy()[hard_triplets]), requires_grad=True)
                out_selected_p = Variable(torch.from_numpy(outputs1.data.numpy()[hard_triplets]), requires_grad=True)
                out_selected_n = Variable(torch.from_numpy(outputs2.data.numpy()[hard_triplets]), requires_grad=True)
                outputs0 = out_selected_a
                outputs1 = out_selected_p
                outputs2 = out_selected_n

            loss = criterion(outputs0, outputs1, outputs2)

            # forward, backward, update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iteration_number += 1
            loss_sigma += loss.item()

            if i % 10 == 0:
                # print("Epoch:{},  Current loss {}".format(epoch, loss.item()))
                train_iter_index.append(iteration_number)
                train_loss_list.append(loss.item())

        # 每个epoch的 Loss, accuracy, learning rate
        lr_now = [group['lr'] for group in optimizer.param_groups][0]
        loss_avg_epoch = loss_sigma / len(train_loader)
        print(
            "Training: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f}       Lr: {:.8f}".format(
                epoch + 1, opt.max_epoch, loss_avg_epoch, lr_now))

        scheduler.step(loss_avg_epoch)  # 更新学习率

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 1 == 0:
            # print('eval start ')
            loss_sigma = 0
            distance_AP_list = []
            distance_AN_list = []
            predicted = []
            net.eval()  # 测试模式
            for i, data in tqdm((enumerate(valid_loader))):
                # 获取图片和标签
                inputs0, inputs1, inputs2 = data
                inputs0, inputs1, inputs2 = Variable(inputs0), Variable(inputs1), Variable(inputs2)

                # forward
                outputs0, outputs1, outputs2 = net(inputs0), net(inputs1), net(inputs2)

                loss = criterion(outputs0, outputs1, outputs2)

                loss_sigma += loss.item()

                # 统计
                distance_AP = F.pairwise_distance(outputs0, outputs1, 2)
                distance_AN = F.pairwise_distance(outputs0, outputs2, 2)
                # print(euclidean_distance.data)
                for tt in range(len(distance_AP)):
                    distance_AP_list.append(distance_AP[tt].item())
                    distance_AN_list.append(distance_AN[tt].item())
                    distance_N_P = distance_AN - distance_AP
                    if distance_N_P[tt] > 0:
                        predicted.append(0)  # positive pairs
                    else:
                        predicted.append(1)  # negative pairs
                # print(predicted, labels.data)

            # print(conf_mat)
            val_acc_avg = (len(predicted) - sum(predicted)) / len(predicted)
            val_loss_avg = loss_sigma / len(valid_loader)
            val_iter_index.append(iteration_number)
            val_loss_list.append(val_loss_avg)
            val_acc_list.append(val_acc_avg)
            print(
                "Validating: Epoch[{:0>3}/{:0>3}] Loss_Avg_Epoch: {:.4f} Accuracy:{:.4f}".format(
                    epoch + 1, opt.max_epoch, val_loss_avg, val_acc_avg))
            # print(euclidean_distance_list)
            show_distance(distance_AP_list, distance_AN_list, opt.show_plot_epoch, epoch)
            # 每次验证完，根据验证数据判断是否存储当前网络数据
            if (val_loss_avg < best_val_loss) or (val_acc_avg > best_val_acc):
                best_val_loss = np.min((val_loss_avg, best_val_loss))
                best_val_acc = np.max((val_acc_avg, best_val_acc))
                # 储存权重
                time_str = time.strftime('%m%d%H%M%S')
                save_name = '%s_%s_%s_%s_net_params.pkl' % (time_str, best_val_acc, '{:.4f}'.format(best_val_loss),
                                                            net_name)
                now_best_net_save_path = os.path.join(opt.log_dir, save_name)
                torch.save(net.state_dict(), now_best_net_save_path)

    print('Finished Training')
    time_str = time.strftime('%H时%M分%S秒')
    print(time_str)
    # ------------------------------------ step5: 加载最好模型 并且在测试集上评估 ------------------------------------

    # evaluation(now_best_net_save_path, net=net)

    show_plot(train_iter_index, train_loss_list, val_iter_index, val_loss_list, val_acc_list)
    # show_distance(distance_AP_list, distance_AN_list, opt.show_plot_epoch, epoch)
    pass


if __name__ == '__main__':
    train()
    # net_path = 'checkpoints/02201722_0.944_resnet18_net_params.pkl'
    # evaluation(net_path, net=None)
