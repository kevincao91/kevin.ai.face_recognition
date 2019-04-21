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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from config import cuda_opt
from my_class import DataLoaderPool, FaceRecognitionNetPool
from evaluation import test_cuda, test_cuda_dir, valid_cuda


def show_plot(train_iter_index, train_loss_list, val_iter_index, val_acc_list, test_iter_index, test_acc_list):
    plt.figure('Loss and Accuracy', figsize=(12, 6.75))
    plt.subplot(1, 2, 1)
    plt.plot(train_iter_index, train_loss_list)
    plt.subplot(1, 2, 2)
    plt.plot(val_iter_index, val_acc_list)
    plt.plot(test_iter_index, test_acc_list)

    time_str = time.strftime('%m%d%H%M%S')
    save_name = '%s_Loss and Accuracy.png' % time_str
    fig_save_path = os.path.join(cuda_opt.log_dir, save_name)
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
    plt.ylim((0, 2))
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
    fig_save_path = os.path.join(cuda_opt.log_dir, save_name)
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
    inputs0, inputs1, inputs2 = Variable(inputs0).cuda(), Variable(inputs1).cuda(), Variable(inputs2).cuda()
    outputs0, outputs1, outputs2 = net(inputs0), net(inputs1), net(inputs2)
    # 统计
    input_num = len(inputs0)
    # 编码结果脱离
    outputs0, outputs1, outputs2 = outputs0.cpu(), outputs1.cpu(), outputs2.cpu()
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

    return ori_inputs0, ori_inputs1, tar_inputs2        # 把原始负样本替换成了满足semi-hard的负样本


def train():

    # 显示系统信息
    print('\n')
    flag = torch.cuda.is_available()
    print('cuda is available on this system ?', flag)
    if not flag:
        return
    # 返回True代表支持，False代表不支持

    # ------------------------------------  显示程序设定参数  -------------------------------------

    cuda_opt.log_dir = 'checkpoints/' + '%s_%d_%d_%s_%s/' % (cuda_opt.train_net_name,
                                                             cuda_opt.max_epoch,
                                                             cuda_opt.train_bs,
                                                             cuda_opt.face_data,
                                                             time.strftime('%m%d%H%M%S'))
    os.makedirs(cuda_opt.log_dir)
    cuda_opt._print_opt()

    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(cuda_opt)
    train_loader = data_pool.select_dataloader(data_type='train')
    valid_loader = data_pool.select_dataloader(data_type='valid')
    test_loader = data_pool.select_dataloader(data_type='test')
    print('load data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------

    net_name = cuda_opt.train_net_name
    net_pool = FaceRecognitionNetPool(cuda_opt)     # 模型选择类
    net = net_pool.select_model(net_name)           # 调用网络
    net = net.cuda()
    print('load net done !')

    # ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

    criterion_triplet = nn.TripletMarginLoss(margin=cuda_opt.margin).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                     verbose=True, threshold=0.005, threshold_mode='rel',
                                                     cooldown=0, min_lr=0, eps=1e-08)  # 设置学习率下降策略
    # criterion_cross = nn.CrossEntropyLoss().cuda()

    # ------------------------------------ step 4/5 : 训练 --------------------------------------------------
    print('train start ------------------------------------------------')
    time_str = time.strftime('%H时%M分%S秒')
    print(time_str)
    iteration_number = 0
    train_iter_index = []
    val_iter_index = []
    test_iter_index = []
    train_loss_list = []
    val_acc_list = []
    test_acc_list = []
    best_val_acc = 0
    best_test_acc = 0
    now_best_net_save_path = None

    for epoch in range(cuda_opt.max_epoch):
        loss_sigma = 0
        net.train()  # 训练模式

        pbar = tqdm((enumerate(train_loader)))
        for i, data in pbar:

            pbar.set_description('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, i * cuda_opt.train_bs, len(train_loader.dataset), 100. * i / len(train_loader)))

            # 获取图片和标签
            inputs0, inputs1, inputs2, label1, label2 = data
            inputs0, inputs1, inputs2, label1, label2 = Variable(inputs0).cuda(),\
                                                        Variable(inputs1).cuda(),\
                                                        Variable(inputs2).cuda(),\
                                                        Variable(label1).cuda(), Variable(label2).cuda()
            outputs0, outputs1, outputs2 = net(inputs0), net(inputs1), net(inputs2)

            # online triplet select
            # Choose the hard negatives
            d_p = F.pairwise_distance(outputs0, outputs1, 2)
            d_n = F.pairwise_distance(outputs0, outputs2, 2)
            hard_negatives = (d_n - d_p < cuda_opt.margin).data.cpu().numpy().flatten()
            hard_triplets = np.where(hard_negatives == 1)
            if len(hard_triplets[0]) == 0:
                continue
            outputs0 = outputs0[hard_triplets]
            outputs1 = outputs1[hard_triplets]
            outputs2 = outputs2[hard_triplets]

            '''
            inputs0 = inputs0[hard_triplets]
            inputs1 = inputs1[hard_triplets]
            inputs2 = inputs2[hard_triplets]
            label1 = label1[hard_triplets]
            label2 = label2[hard_triplets]

            cls_a = net.forward_classifier(inputs0)
            cls_p = net.forward_classifier(inputs1)
            cls_n = net.forward_classifier(inputs2)

            loss_cross_a = criterion_cross(cls_a, label1)
            loss_cross_p = criterion_cross(cls_p, label1)
            loss_cross_n = criterion_cross(cls_n, label2)
            loss_cross = loss_cross_a + loss_cross_p + loss_cross_n

            loss_triplet = 10.0 * criterion_triplet(outputs0, outputs1, outputs2)

            loss = loss_cross + loss_triplet
            '''
            loss = criterion_triplet(outputs0, outputs1, outputs2)
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
            "\33[34m\t\tTrain_Loss_Avg: {:.4f}\33[0m\t\t\t\t\t\t\33[35mLr: {:.8f}\33[0m".format(loss_avg_epoch, lr_now))

        scheduler.step(loss_avg_epoch)  # 更新学习率

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        if epoch % 2 == 0:
            # print('eval start ')
            valid_acc = valid_cuda(valid_loader, net, epoch)

            val_iter_index.append(iteration_number)
            val_acc_list.append(valid_acc)
            # print("\33[34m\t\t\t\tValid Accuracy:{:.4f}\33[0m".format(valid_acc))
            # print(euclidean_distance_list)
            # show_distance(distance_AP_list, distance_AN_list, opt.show_plot_epoch, epoch)
            # 每次验证完，根据验证数据判断是否存储当前网络数据
            if valid_acc > best_val_acc:
                best_val_acc = valid_acc
                # 储存权重
                time_str = time.strftime('%m%d%H%M%S')
                save_name = '%s_validacc_%s_%s_net_params.pkl' % (time_str, '{:.4f}'.format(best_val_acc), net_name)
                now_best_net_save_path = os.path.join(cuda_opt.log_dir, save_name)
                torch.save(net.state_dict(), now_best_net_save_path)

            # ------------------------------------ 观察模型在测试集上的表现 ------------------------------------
        if (epoch + 1) % 5 == 0:
            # print('eval start ')
            test_acc = test_cuda(test_loader, net, epoch)

            test_iter_index.append(iteration_number)
            test_acc_list.append(test_acc)
            # 每次验证完，根据验证数据判断是否存储当前网络数据
            if np.mean(test_acc) > best_test_acc:
                best_test_acc = np.mean(test_acc)
                # 储存权重
                time_str = time.strftime('%m%d%H%M%S')
                save_name = '%s_testacc_%s_%s_net_params.pkl' % (time_str, '{:.4f}'.format(best_test_acc), net_name)
                now_best_net_save_path = os.path.join(cuda_opt.log_dir, save_name)
                torch.save(net.state_dict(), now_best_net_save_path)

    print('Finished Training')
    time_str = time.strftime('%H时%M分%S秒')
    print(time_str)
    # ------------------------------------ step5: 加载最好模型 并且在测试集上评估 ------------------------------------

    show_plot(train_iter_index, train_loss_list, val_iter_index, val_acc_list, test_iter_index, test_acc_list)
    # show_distance(distance_AP_list, distance_AN_list, opt.show_plot_epoch, epoch)

    # evaluation(now_best_net_save_path, net=net)
    # test_cuda_dir(cuda_opt.log_dir)


if __name__ == '__main__':
    train()
