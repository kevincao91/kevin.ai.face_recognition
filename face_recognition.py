#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import os
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet34, resnet50
from torchvision import transforms
import matplotlib.pyplot as plt


from config import opt
from my_class import MyDataset, FaceRecoModel


def load_data():
    # 数据预处理设置
    mean_data_path = 'picture/mean_data.txt'
    with open(mean_data_path, 'r') as f:
        lines = f.readlines()

    normMean = [float(i) for i in lines[0].split()]
    normStd = [float(i) for i in lines[1].split()]

    normTransform = transforms.Normalize(normMean, normStd)

    testTransform = transforms.Compose([
        transforms.Resize(100),
        transforms.ToTensor(),
        normTransform
    ])
    # print('load data')

    # 构建MyDataset实例
    import_data = MyDataset(split='PersonImageData', transform=testTransform)
    test_data = MyDataset(split='PersonTestImageData', transform=testTransform)
    import_data_num = len(import_data)
    test_data_num = len(test_data)
    print(import_data_num)
    print(test_data_num)

    # 构建DataLoder
    import_loader = DataLoader(dataset=import_data, batch_size=import_data_num)
    test_loader = DataLoader(dataset=test_data, batch_size=test_data_num)

    return import_loader, test_loader


def create_person_data(net, import_loader):
    database = {}
    net.eval()
    it = iter(import_loader)
    images, labels = it.next()
    images = Variable(images)
    outputs = net(images)

    for i in range(len(labels)):
        database[labels[i]] = outputs[i].data.numpy()
        # print(labels[i], outputs[i].data.numpy())

    # print(database)
    return database


def who_is_it(database, net, test_loader):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.

    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras

    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """

    # START CODE HERE #

    # Step 1: Compute the target "encoding" for the image.
    it = iter(test_loader)
    images, labels = it.next()
    images = Variable(images)
    net.eval()

    # Step 1: Compute the encoding for the image.
    encodings = net(images)
    encodings = encodings.data.numpy()

    # Step 2: Find the closest encoding #
    identity_list = []
    min_dist_list = []
    for i in range(len(images)):
        # Initialize "min_dist" to a large value, say 100 (≈1 line)
        min_dist = 100
        identity = 'None'
        encoding = encodings[i]
        # Loop over the database dictionary's names and encodings.
        for (name, db_enc) in database.items():

            # Compute L2 distance between the target "encoding" and the current "emb" from the database. (≈ 1 line)
            dist = np.linalg.norm(encoding - db_enc)

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist < min_dist:
                min_dist = dist
                identity = name
        identity_list.append(identity)
        min_dist_list.append(min_dist)
    # END CODE HERE #

    for i in range(len(images)):
        min_dist = min_dist_list[i]
        identity = identity_list[i]
        if min_dist > 5.0:
            print("Not in the database.")
            identity_list[i] = 'None'
        else:
            print("it's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist_list, identity_list


def show_result(dist_list, string_list):

    # 读取import图片路径
    txt_path = 'picture/PersonImageData.txt'
    gt_imgs = []
    gt_labels = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        words = line.split()
        img_file_path = os.path.join('picture', words[1])
        gt_imgs.append(img_file_path)
        gt_labels.append(words[0])

    # 读取test图片路径
    txt_path = 'picture/PersonTestImageData.txt'
    imgs = []
    labels = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        line = line.rstrip()
        words = line.split()
        img_file_path = os.path.join('picture', words[1])
        imgs.append(img_file_path)
        labels.append(words[0])

    # 展示图片和结果
    plt.figure('result', figsize=(13, 6))
    for i in range(len(dist_list)):
        plt.subplot(2, 4, i+1)
        if labels[i] == 'kevin':
            gt_img = plt.imread(gt_imgs[0])
        elif labels[i] == 'yiyi':
            gt_img = plt.imread(gt_imgs[1])
        img = plt.imread(imgs[i])
        two_img = np.hstack((gt_img, img))
        plt.imshow(two_img)
        plt.axis("off")
        text = 'd=' + format(dist_list[i], '0.2f')
        w = np.shape(img)[0]
        plt.text(w + 5, -60, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})
        text = 'it\'s ' + string_list[i]
        plt.text(w+5, -20, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2})
    plt.show()


if __name__ == '__main__':

    import_loader, test_loader = load_data()
    print('load data done !')

    # 调用已有模型
    model = resnet18(pretrained=False)
    # 提取fc层中固定的参数
    fc_features = model.fc.in_features
    # 修改类别为128
    model.fc = nn.Linear(fc_features, 128)
    net = model  # 调用修改后的已有网络
    pretrained_weight_path = 'checkpoints/02161821_0.8_resnet18_net_params.pkl'
    net.load_state_dict(torch.load(pretrained_weight_path))
    net.eval()
    print('load net done !')

    database = create_person_data(net, import_loader)
    print('load person data done !')
    print(database)

    min_dist_list, identity_list = who_is_it(database, net, test_loader)

    show_result(min_dist_list, identity_list)
