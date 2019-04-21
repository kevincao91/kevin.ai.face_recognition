# coding: utf-8
"""
    将原始数据集进行划分成训练集、验证集和测试集或整体成为测试集
"""
from tqdm import tqdm
import numpy as np
import os
import glob
from config import opt
import random


lfw_img_num = 18982
lfw_idx_num = 5749

lfw_data_dir = opt.lfw_data_dir + '/lfw/'
indices_all_file_path = 'dataset/lfw_all_indices.txt'
indices_trainval_file_path = 'dataset/lfw_trainval_indices.txt'
indices_test_file_path = 'dataset/lfw_test_indices.txt'
train_file_path = 'dataset/lfw_train_triplet.txt'
valid_file_path = 'dataset/lfw_valid_pairs.txt'
test_file_path = 'dataset/lfw_test_pairs.txt'
all_test_file_path = 'dataset/lfw_all_test_pairs.txt'


train_percent = 0.5
valid_percent = 0.2
test_percent = 0.3


def make_indices_file():
    with open(indices_all_file_path, 'w') as f:
        for root, dirs, files in os.walk(lfw_data_dir):
            idx = 0
            for sDir in dirs:
                imgs_path_list = glob.glob(os.path.join(root, sDir) + '/*.jpg')
                imgs_num = len(imgs_path_list)
                file_string = ''
                for i in range(1, imgs_num + 1):
                    file_string += '\t'
                    file_string += str(i)
                line = str(idx) + '\t' + sDir + '\t' + file_string + '\n'
                f.write(line)
                idx += 1
    print('load %d people file' % lfw_idx_num)


def split_indices_file():
    trainval_num = round(lfw_idx_num * (train_percent + valid_percent))
    test_num = round(lfw_idx_num * test_percent)
    print('trainval set load %d people file' % trainval_num)
    print('test set load %d people file' % test_num)

    with open(indices_all_file_path, 'r') as f:
        lines = f.readlines()

    trainval_file = open(indices_trainval_file_path, 'w')
    test_file = open(indices_test_file_path, 'w')

    random.shuffle(lines)
    idx = 1
    for line in lines:
        if idx <= trainval_num:
            trainval_file.write(line)
        else:
            test_file.write(line)
        idx += 1
    trainval_file.close()
    test_file.close()


def make_train_file(num_triplets):
    with open(indices_trainval_file_path, 'r') as f:
        lines = f.readlines()

    name_list = []
    inds = dict()
    idx = 0    # 重新排序号
    for line in lines:
        words = line.split()
        name = words[1]
        files = words[2:]
        if name not in inds:
            inds[idx] = []
            name_list.append(name)
        for file in files:
            inds[idx].append(file)
        idx += 1

    n_classes = len(inds)
    print('[train set] num_triplets: ', num_triplets, 'n_classes: ', n_classes)

    triplets = []
    # Indices = array of labels and each label is an array of indices
    indices_dic = inds

    with open(train_file_path, 'w') as f:

        f.write(str(num_triplets) + '\t' + str(n_classes) + '\n')

        for x in range(num_triplets):
            c1 = np.random.randint(0, n_classes - 1)
            c2 = np.random.randint(0, n_classes - 1)
            while len(indices_dic[c1]) < 2:
                c1 = np.random.randint(0, n_classes - 1)

            while c1 == c2:
                c2 = np.random.randint(0, n_classes - 1)
            if len(indices_dic[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                n2 = np.random.randint(0, len(indices_dic[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices_dic[c1]) - 1)
            if len(indices_dic[c2]) == 1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices_dic[c2]) - 1)

            triplets.append(
                [name_list[c1], indices_dic[c1][n1], indices_dic[c1][n2], name_list[c2], indices_dic[c2][n3], c1, c2])

            line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + indices_dic[c1][n2] + '\t' + \
                   name_list[c2] + '\t' + indices_dic[c2][n3] + '\t' + str(c1) + '\t' + str(c2) + '\n'

            f.write(line)


def make_valid_file(num_pairs):
    with open(indices_trainval_file_path, 'r') as f:
        lines = f.readlines()

    name_list = []
    inds = dict()
    idx = 0    # 重新排序号
    for line in lines:
        words = line.split()
        name = words[1]
        files = words[2:]
        if name not in inds:
            inds[idx] = []
            name_list.append(name)
        for file in files:
            inds[idx].append(file)
        idx += 1

    n_classes = len(inds)
    print('[valid set] num_pairs: ', num_pairs, 'n_classes: ', n_classes)

    pairs = []
    # Indices = array of labels and each label is an array of indices
    indices_dic = inds

    with open(valid_file_path, 'w') as f:

        f.write(str(num_pairs) + '\t' + str(n_classes) + '\n')

        for x in range(num_pairs):
            if x < num_pairs/2:  # positive pairs
                c1 = np.random.randint(0, n_classes - 1)
                while len(indices_dic[c1]) < 2:
                    c1 = np.random.randint(0, n_classes - 1)

                if len(indices_dic[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    n2 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices_dic[c1]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], indices_dic[c1][n2], c1])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + indices_dic[c1][n2] + '\t' + str(c1) + '\n'
            else:   # negative pairs
                c1 = np.random.randint(0, n_classes - 1)
                c2 = np.random.randint(0, n_classes - 1)

                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                if len(indices_dic[c1]) == 1:  # hack to speed up process
                    n1 = 0
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                if len(indices_dic[c2]) == 1:  # hack to speed up process
                    n2 = 0
                else:
                    n2 = np.random.randint(0, len(indices_dic[c2]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], name_list[c2], indices_dic[c2][n2], c1, c2])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + \
                       name_list[c2] + '\t' + indices_dic[c2][n2] + '\t' + str(c1) + '\t' + str(c2) + '\n'

            f.write(line)


def make_test_file(num_pairs):
    with open(indices_test_file_path, 'r') as f:
        lines = f.readlines()

    name_list = []
    inds = dict()
    idx = 0    # 重新排序号
    for line in lines:
        words = line.split()
        name = words[1]
        files = words[2:]
        if name not in inds:
            inds[idx] = []
            name_list.append(name)
        for file in files:
            inds[idx].append(file)
        idx += 1

    n_classes = len(inds)
    print('[test set] num_pairs: ', num_pairs, 'n_classes: ', n_classes)

    pairs = []
    # Indices = array of labels and each label is an array of indices
    indices_dic = inds

    with open(test_file_path, 'w') as f:

        f.write(str(num_pairs) + '\t' + str(n_classes) + '\n')

        for x in range(num_pairs):
            if x < num_pairs/2:  # positive pairs
                c1 = np.random.randint(0, n_classes - 1)
                while len(indices_dic[c1]) < 2:
                    c1 = np.random.randint(0, n_classes - 1)

                if len(indices_dic[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    n2 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices_dic[c1]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], indices_dic[c1][n2], c1])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + indices_dic[c1][n2] + '\t' + str(c1) + '\n'
            else:   # negative pairs
                c1 = np.random.randint(0, n_classes - 1)
                c2 = np.random.randint(0, n_classes - 1)

                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                if len(indices_dic[c1]) == 1:  # hack to speed up process
                    n1 = 0
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                if len(indices_dic[c2]) == 1:  # hack to speed up process
                    n2 = 0
                else:
                    n2 = np.random.randint(0, len(indices_dic[c2]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], name_list[c2], indices_dic[c2][n2], c1, c2])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + \
                       name_list[c2] + '\t' + indices_dic[c2][n2] + '\t' + str(c1) + '\t' + str(c2) + '\n'

            f.write(line)


def make_all_test_file(num_pairs):
    with open(indices_all_file_path, 'r') as f:
        lines = f.readlines()

    name_list = []
    inds = dict()
    for line in lines:
        words = line.split()
        idx = int(words[0])  # 获取原始编号
        name = words[1]
        files = words[2:]
        if name not in inds:
            inds[idx] = []
            name_list.append(name)
        for file in files:
            inds[idx].append(file)

    n_classes = len(inds)
    print('[all test set] num_pairs: ', num_pairs, 'n_classes: ', n_classes)

    pairs = []
    # Indices = array of labels and each label is an array of indices
    indices_dic = inds

    with open(all_test_file_path, 'w') as f:

        f.write(str(num_pairs) + '\t' + str(n_classes) + '\n')

        for x in range(num_pairs):
            if x < num_pairs/2:  # positive pairs
                c1 = np.random.randint(0, n_classes - 1)
                while len(indices_dic[c1]) < 2:
                    c1 = np.random.randint(0, n_classes - 1)

                if len(indices_dic[c1]) == 2:  # hack to speed up process
                    n1, n2 = 0, 1
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    n2 = np.random.randint(0, len(indices_dic[c1]) - 1)
                    while n1 == n2:
                        n2 = np.random.randint(0, len(indices_dic[c1]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], indices_dic[c1][n2], c1])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + indices_dic[c1][n2] + '\t' + str(c1) + '\n'
            else:   # negative pairs
                c1 = np.random.randint(0, n_classes - 1)
                c2 = np.random.randint(0, n_classes - 1)

                while c1 == c2:
                    c2 = np.random.randint(0, n_classes - 1)
                if len(indices_dic[c1]) == 1:  # hack to speed up process
                    n1 = 0
                else:
                    n1 = np.random.randint(0, len(indices_dic[c1]) - 1)
                if len(indices_dic[c2]) == 1:  # hack to speed up process
                    n2 = 0
                else:
                    n2 = np.random.randint(0, len(indices_dic[c2]) - 1)

                pairs.append(
                    [name_list[c1], indices_dic[c1][n1], name_list[c2], indices_dic[c2][n2], c1, c2])

                line = name_list[c1] + '\t' + indices_dic[c1][n1] + '\t' + \
                       name_list[c2] + '\t' + indices_dic[c2][n2] + '\t' + str(c1) + '\t' + str(c2) + '\n'

            f.write(line)


if __name__ == '__main__':
    make_indices_file()
    split_indices_file()
    make_train_file(8000)
    make_valid_file(2000)
    make_test_file(3000)
    make_all_test_file(100)
