from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18, resnet34, resnet50, vgg16


# 数据集类 ======================================
class LWFDataset(Dataset):
    def __init__(self, data_dir, img_dir, split='tripletTrain', transform=None, target_transform=None):
        txt_path = os.path.join(data_dir, '{0}.txt'.format(split))
        imgsA = []
        imgsP = []
        imgsN = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        num_line = lines[0]
        triplet_num = int(num_line)
        triplet_lines = lines[1:]

        for line in triplet_lines:
            # print(line)
            line = line.rstrip()
            words = line.split()
            imgA_file_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
            # print(imgA_file_path)
            imgsA.append(imgA_file_path)
            imgP_file_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[2])))
            # print(imgP_file_path)
            imgsP.append(imgP_file_path)
            imgN_file_path = os.path.join(img_dir, words[3], words[3] + "_%0*d.jpg" % (4, int(words[4])))
            # print(imgN_file_path)
            imgsN.append(imgN_file_path)

        self.imgsA = imgsA  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsP = imgsP  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsN = imgsN
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgsA[index]
        fn1 = self.imgsP[index]
        fn2 = self.imgsN[index]
        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img2 = Image.open(fn2).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等
            img2 = self.transform(img2)  # 在这里做transform，转为tensor等等

        return img0, img1, img2

    def __len__(self):
        return len(self.imgsA)


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_dir, split='tripletTrain', transform=None, target_transform=None):
        txt_path = os.path.join(data_dir, '{0}.txt'.format(split))
        imgsA = []
        imgsP = []
        imgsN = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        num_line = lines[0]
        triplet_num = int(num_line)
        triplet_lines = lines[1:]

        for line in triplet_lines:
            # print(line)
            line = line.rstrip()
            words = line.split()
            imgA_file_path = os.path.join(img_dir, words[1])
            # print(imgA_file_path)
            imgsA.append(imgA_file_path)
            imgP_file_path = os.path.join(img_dir, words[2])
            # print(imgP_file_path)
            imgsP.append(imgP_file_path)
            imgN_file_path = os.path.join(img_dir, words[4])
            # print(imgN_file_path)
            imgsN.append(imgN_file_path)

        self.imgsA = imgsA  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsP = imgsP  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsN = imgsN
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgsA[index]
        fn1 = self.imgsP[index]
        fn2 = self.imgsN[index]
        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img2 = Image.open(fn2).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等
            img2 = self.transform(img2)  # 在这里做transform，转为tensor等等

        return img0, img1, img2

    def __len__(self):
        return len(self.imgsA)


class MyDataset(Dataset):
    def __init__(self, split='PersonImageData', transform=None, target_transform=None):
        img_dir = 'picture/'
        txt_path = os.path.join('picture/', '{0}.txt'.format(split))
        fh = open(txt_path, 'r')
        imgs = []
        labels = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            # print(line)
            line = line.rstrip()
            words = line.split()
            img_file_path = os.path.join(img_dir, words[1])
            # print(imgA_file_path)
            imgs.append(img_file_path)
            labels.append(words[0])
            # print(words[0])

        self.imgs = imgs  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        label = self.labels[index]
        img = Image.open(fn).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)  # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


class DataLoaderPool:
    def __init__(self, opt_use):
        self.opt = opt_use
        self.trainTransform = None
        self.validTransform = None
        self.testTransform = None
        self.get_transform()

    def get_transform(self):
        # 数据预处理设置
        mean_data_path = os.path.join(self.opt.imgset_data_dir, 'mean_data.txt')
        with open(mean_data_path, 'r') as f:
            lines = f.readlines()

        normMean = [float(i) for i in lines[0].split()]
        normStd = [float(i) for i in lines[1].split()]

        normTransform = transforms.Normalize(normMean, normStd)

        transforms_list = []
        if self.opt.crop_size != 0:
            transforms_list.append(transforms.CenterCrop(self.opt.crop_size))
        if self.opt.resize_size != 0:
            transforms_list.append(transforms.Resize(self.opt.resize_size))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(normTransform)

        self.trainTransform = transforms.Compose(transforms_list)
        self.validTransform = transforms.Compose(transforms_list)
        self.testTransform = transforms.Compose(transforms_list)

    def select_dataloader(self, data_type='train'):
        if data_type not in ['train', 'valid', 'trainval', 'test']:
            return None
        # 选择Dataset参数
        if data_type == 'train':
            split = 'tripletTrain'
            transform = self.trainTransform
            batch_size = self.opt.train_bs
            shuffle = True
        elif data_type == 'valid':
            split = 'tripletValid'
            transform = self.validTransform
            batch_size = self.opt.valid_bs
            shuffle = False
        elif data_type == 'trainval':
            split = 'tripletTrainval'
            transform = self.trainTransform
            batch_size = self.opt.train_bs
            shuffle = True
        else:
            split = 'tripletTest'
            transform = self.testTransform
            batch_size = self.opt.test_bs
            shuffle = False
        # 构建Data
        if self.opt.face_data == 'lfw':
            data = LWFDataset(data_dir=self.opt.imgset_data_dir, img_dir=self.opt.img_data_dir,
                              split=split, transform=transform)
            # 构建DataLoder
            data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
        elif self.opt.face_data == 'celeba':
            data = CelebADataset(data_dir=self.opt.imgset_data_dir, img_dir=self.opt.img_data_dir,
                                 split=split, transform=transform)
            # 构建DataLoder
            data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
        else:
            print('No Face Data !')
            return None
        print(self.opt.face_data + ' ' + data_type + ' data number : ', len(data))
        return data_loader


# 模型类 ========================================
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(kernel_size=6, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, 10)

    def forward(self, inputs):
        network = self.pre_layers(inputs)
        network = self.a3(network)
        network = self.b3(network)
        network = self.maxpool(network)
        network = self.a4(network)
        network = self.b4(network)
        network = self.c4(network)
        network = self.d4(network)
        network = self.e4(network)
        network = self.maxpool(network)
        network = self.a5(network)
        network = self.b5(network)
        # print(np.shape(network))
        network = self.avgpool(network)
        # print(np.shape(network))
        network = network.view(network.size(0), -1)
        # print(np.shape(network))
        network = self.dropout(network)
        out = self.fc(network)
        return out


class FaceRecognitionNetPool:
    def __init__(self, opt_use):
        self.opt = opt_use
        self.input_img_size = 0
        self.cal_input_size()

    def cal_input_size(self):
        if self.opt.crop_size == 0 and self.opt.resize_size == 0:
            self.input_img_size = self.opt.input_img_size
        elif self.opt.resize_size == 0:
            self.input_img_size = self.opt.crop_size
        else:
            self.input_img_size = self.opt.resize_size
        print('input_img_size', self.input_img_size)

    def select_model(self, net_name):
        if net_name == '' or net_name is None:
            return None
        if net_name == 'resnet18':
            net = resnet18(pretrained=True)
        elif net_name == 'resnet34':
            net = resnet34(pretrained=True)
        elif net_name == 'resnet50':
            net = resnet50(pretrained=True)
        elif net_name == 'googlenet':
            net = GoogLeNet()
        elif net_name == 'vgg16':
            net = vgg16(pretrained=True)
        elif net_name == 'face':
            net = FaceModel()
        else:
            return None
        # 更改avgpool
        # avg_pool_size = int(self.input_img_size / 32)
        # net.avgpool = nn.AvgPool2d(kernel_size=avg_pool_size, stride=1)
        # 修改fc
        if net_name == 'vgg16':
            fc_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(fc_features, 128)
        elif net_name == 'face':
            pass
        else:
            fc_features = net.fc.in_features
            net.fc = nn.Linear(fc_features, 128)

        return net


class FaceModel(nn.Module):
    def __init__(self):
        super(FaceModel, self).__init__()
        self.features = None
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512*7*7, 128)
        # self.model.classifier = nn.Linear(self.embedding_size, num_classes)

    def l2_norm(self, x):
        input_size = x.size()
        buffer = torch.pow(x, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(x, norm.view(-1, 1).expand_as(x))
        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        self.features = self.l2_norm(x)
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res

