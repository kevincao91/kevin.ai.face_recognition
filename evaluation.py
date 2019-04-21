import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import glob
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from eval_metrics import evaluate, valid_evaluate

from utils import PairwiseDistance, display_triplet_distance, display_triplet_distance_test

from config import opt, cuda_opt
from my_class import DataLoaderPool, FaceRecognitionNetPool

np.random.seed(68)

l2_dist = PairwiseDistance(2)


def plot_roc(fpr, tpr, figure_name="roc.png"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    roc_auc = auc(fpr, tpr)
    fig = plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    fig.savefig(figure_name, dpi=fig.dpi)


def valid(valid_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(valid_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        with torch.no_grad():
            data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            # euclidean distance
            dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            for tt in range(len(dists)):
                distances.append(dists[tt].item())
                labels.append(label[tt].item())

        pbar.set_description('Valid Epoch: {} [{}/{} ({:.0f}%)]'.format(
            epoch, batch_idx * len(data_a), len(valid_loader.dataset), 100. * batch_idx / len(valid_loader)))

    best_thr, best_acc = valid_evaluate(distances, labels)
    print('\33[91mvalid set: Accuracy: {:.8f}\n\33[0m'.format(best_acc))
    # logger.log_value('Test Accuracy', np.mean(accuracy))

    # figure_name = os.path.join(opt.log_dir, "roc_test_epoch_{}.png".format(epoch))
    # plot_roc(fpr, tpr, figure_name=figure_name)
    return best_acc


def valid_cuda(valid_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(valid_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        with torch.no_grad():
            data_a, data_p, label = Variable(data_a).cuda(), Variable(data_p).cuda(), Variable(label).cuda()

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            # euclidean distance
            dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            for tt in range(len(dists)):
                distances.append(dists[tt].cpu().item())
                labels.append(label[tt].cpu().item())

        pbar.set_description('Valid Epoch: {} [{}/{} ({:.0f}%)]'.format(
            epoch, batch_idx * len(data_a), len(valid_loader.dataset), 100. * batch_idx / len(valid_loader)))

    best_thr, best_acc = valid_evaluate(distances, labels)
    print('\33[91mvalid set: Accuracy: {:.8f}\n\33[0m'.format(best_acc))
    # logger.log_value('Test Accuracy', np.mean(accuracy))

    # figure_name = os.path.join(opt.log_dir, "roc_test_epoch_{}.png".format(epoch))
    # plot_roc(fpr, tpr, figure_name=figure_name)
    return best_acc


def test(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        with torch.no_grad():
            data_a, data_p, label = Variable(data_a), Variable(data_p), Variable(label)

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            # euclidean distance
            dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            for tt in range(len(dists)):
                distances.append(dists[tt].item())
                labels.append(label[tt].item())

        pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
            epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    # logger.log_value('Test Accuracy', np.mean(accuracy))

    figure_name = os.path.join(opt.log_dir, "roc_test_epoch_{}.png".format(epoch))
    plot_roc(fpr, tpr, figure_name=figure_name)
    return np.mean(accuracy)


def test_cuda(test_loader, model, epoch):
    # switch to evaluate mode
    model.eval()

    labels, distances = [], []

    pbar = tqdm(enumerate(test_loader))
    for batch_idx, (data_a, data_p, label) in pbar:

        with torch.no_grad():
            data_a, data_p, label = Variable(data_a).cuda(), Variable(data_p).cuda(), Variable(label).cuda()

            # compute output
            out_a, out_p = model(data_a), model(data_p)
            # euclidean distance
            dists = l2_dist.forward(out_a, out_p)  # torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            for tt in range(len(dists)):
                distances.append(dists[tt].data.cpu().item())
                labels.append(label[tt].data.cpu().item())

        pbar.set_description('Test Epoch: {} [{}/{} ({:.0f}%)]'.format(
            epoch, batch_idx * len(data_a), len(test_loader.dataset), 100. * batch_idx / len(test_loader)))

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
    print('\33[91mTest set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    # logger.log_value('Test Accuracy', np.mean(accuracy))

    figure_name = os.path.join(cuda_opt.log_dir, "roc_test_epoch_{}.png".format(epoch))
    plot_roc(fpr, tpr, figure_name=figure_name)
    return np.mean(accuracy)


def test_dir(net_dir):
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(opt)
    test_loader = data_pool.select_dataloader(data_type='test')
    print('load test data done !')

    net_path_list = []
    for root, dirs, files in tqdm(os.walk(net_dir)):
        for file_path in glob.glob(root + '/*.pkl'):
            net_path_list.append(file_path)
        for sDir in dirs:
            for file_path in glob.glob(os.path.join(root, sDir) + '/*.pkl'):
                net_path_list.append(file_path)
    print('find %d pkl files.' % len(net_path_list))

    with open(os.path.join(opt.log_dir, 'test_accuracy.txt'), 'w') as f:
        idx = 1000
        for net_path in net_path_list:
            print(idx, net_path)
            # ------------------------------------ step 2/5 : 定义网络------------------------------------
            net_name = opt.train_net_name
            net_pool = FaceRecognitionNetPool(opt)  # 模型选择类
            model = net_pool.select_model(net_name)  # 调用网络

            # 加载参数
            best_net_dict = torch.load(net_path)
            model.load_state_dict(best_net_dict)
            print('load net done !')

            # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
            epoch = idx
            accuracy = test(test_loader, model, epoch)

            line = str(idx) + '\t' + net_path + '\t' + accuracy + '\n'
            f.write(line)

            idx += 1


def test_cuda_dir(net_dir):
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(cuda_opt)
    test_loader = data_pool.select_dataloader(data_type='test')
    print('load test data done !')

    net_path_list = []
    for root, dirs, files in tqdm(os.walk(net_dir)):
        for file_path in glob.glob(root + '/*.pkl'):
            net_path_list.append(file_path)
        for sDir in dirs:
            for file_path in glob.glob(os.path.join(root, sDir) + '/*.pkl'):
                net_path_list.append(file_path)
    print('find %d pkl files.' % len(net_path_list))

    with open(os.path.join(opt.log_dir, 'test_accuracy.txt'), 'w') as f:
        idx = 1000
        for net_path in net_path_list:
            print(idx, net_path)
            # ------------------------------------ step 2/5 : 定义网络------------------------------------
            net_name = cuda_opt.train_net_name
            net_pool = FaceRecognitionNetPool(cuda_opt)  # 模型选择类
            model = net_pool.select_model(net_name)  # 调用网络

            # 加载参数
            best_net_dict = torch.load(net_path)
            model.load_state_dict(best_net_dict)
            model = model.cuda()
            print('load net done !')

            # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
            epoch = idx
            accuracy = test_cuda(test_loader, model, epoch)

            line = str(idx) + '\t' + net_path + '\t' + accuracy + '\n'
            f.write(line)

            idx += 1


def test_one(net_path):
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(opt)
    test_loader = data_pool.select_dataloader(data_type='test')
    print('load test data done !')

    # ------------------------------------ step 2/5 : 定义网络------------------------------------

    net_name = opt.train_net_name
    net_pool = FaceRecognitionNetPool(opt)  # 模型选择类
    model = net_pool.select_model(net_name)  # 调用网络

    # 加载参数
    best_net_dict = torch.load(net_path)
    model.load_state_dict(best_net_dict)

    print('load net done !')

    # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
    epoch = 0
    print(epoch, net_path)
    test(test_loader, model, epoch)


def test_more(net_path_list):
    # ------------------------------------ step 1/5 : 加载数据------------------------------------

    data_pool = DataLoaderPool(opt)
    test_loader = data_pool.select_dataloader(data_type='test')
    print('load test data done !')

    idx = 0
    for net_path in net_path_list:
        print(idx, net_path)
        # ------------------------------------ step 2/5 : 定义网络------------------------------------
        net_name = opt.train_net_name
        net_pool = FaceRecognitionNetPool(opt)  # 模型选择类
        model = net_pool.select_model(net_name)  # 调用网络

        # 加载参数
        best_net_dict = torch.load(net_path)
        model.load_state_dict(best_net_dict)

        print('load net done !')

        # ------------------------------------ 观察模型在验证集上的表现 ------------------------------------
        epoch = idx
        test(test_loader, model, epoch)
        idx += 1


if __name__ == '__main__':
    test_dir('/home/kevin/PycharmProjects/deeplearning.ai.face_recognition/checkpoints/')

    # net_path = 'checkpoints/0317054746_0.9330667851325337_0.0955_face_net_params.pkl'
    # test_one(net_path)

    # net_path_list = ['checkpoints/0313133934_0.958_0.1149_resnet18_net_params.pkl',
    #                  'checkpoints/02201722_0.944_resnet18_net_params.pkl',
    #                  'checkpoints/02192106_0.904_resnet18_net_params.pkl',
    #                  'checkpoints/0221115802_0.964_0.1062_resnet18_net_params.pkl',
    #                  'checkpoints/0304121811_0.954_0.1320_resnet18_net_params.pkl',
    #                  ]
    # test_more(net_path_list)
    '''
    result
    0 checkpoints/0313133934_0.958_0.1149_resnet18_net_params.pkl
    train best threshold: 7.740000
    Test set: Accuracy: 0.81785714
    
    1 checkpoints/02201722_0.944_resnet18_net_params.pkl
    交叉验证[1/10]: train best threshold: 6.910000, test acc: 0.857143
    Test set: Accuracy: 0.82500000
    
    2 checkpoints/02192106_0.904_resnet18_net_params.pkl
    train best threshold: 4.730000
    Test set: Accuracy: 0.80714286
    
    3 checkpoints/0221115802_0.964_0.1062_resnet18_net_params.pkl
    train best threshold: 8.380000
    Test set: Accuracy: 0.87500000
    
    4 checkpoints/0304121811_0.954_0.1320_resnet18_net_params.pkl
    train best threshold: 9.430000
    Test set: Accuracy: 0.86428571
    
    '''
