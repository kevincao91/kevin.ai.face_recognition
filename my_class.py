from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image
import os
from torchvision.models import resnet18, resnet34, resnet50, vgg16


# 数据集类 ======================================
class LWFTripletDataset(Dataset):
    def __init__(self, dataset_dir, img_dir, split='lfw_train_triplet', transform=None, target_transform=None):
        txt_path = os.path.join(dataset_dir, '{0}.txt'.format(split))
        imgsA = []
        imgsP = []
        imgsN = []
        labelsP = []
        labelsN = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
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
            labelsP.append(int(words[5]))
            labelsN.append(int(words[6]))

        self.imgsA = imgsA  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsP = imgsP  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsN = imgsN
        self.labelsP = labelsP
        self.labelsN = labelsN
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgsA[index]
        fn1 = self.imgsP[index]
        fn2 = self.imgsN[index]
        lp = torch.from_numpy(np.array(self.labelsP[index]))
        ln = torch.from_numpy(np.array(self.labelsN[index]))

        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img2 = Image.open(fn2).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等
            img2 = self.transform(img2)  # 在这里做transform，转为tensor等等

        return img0, img1, img2, lp, ln

    def __len__(self):
        return len(self.imgsA)


class LWFPairsDataset(Dataset):
    def __init__(self, dataset_dir, img_dir, split='lfw_valid_pairs', transform=None, target_transform=None):
        txt_path = os.path.join(dataset_dir, '{0}.txt'.format(split))
        imgsA = []
        imgsB = []
        labels = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines[1:]:
            # print(line)
            line = line.rstrip()
            words = line.split()
            if len(words) == 4:  # positive pairs
                imgA_file_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
                # print(imgA_file_path)
                imgsA.append(imgA_file_path)
                imgB_file_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[2])))
                # print(imgP_file_path)
                imgsB.append(imgB_file_path)
                labels.append(1)    # is same
            else:  # negative pairs
                imgA_file_path = os.path.join(img_dir, words[0], words[0] + "_%0*d.jpg" % (4, int(words[1])))
                # print(imgA_file_path)
                imgsA.append(imgA_file_path)
                imgB_file_path = os.path.join(img_dir, words[2], words[2] + "_%0*d.jpg" % (4, int(words[3])))
                # print(imgP_file_path)
                imgsB.append(imgB_file_path)
                labels.append(0)    # is not same

        self.imgsA = imgsA  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsB = imgsB  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgsA[index]
        fn1 = self.imgsB[index]
        label = torch.from_numpy(np.array(self.labels[index]))

        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等

        return img0, img1, label

    def __len__(self):
        return len(self.imgsA)


class CelebADataset(Dataset):
    def __init__(self, data_dir, img_dir, split='tripletTrain', transform=None, target_transform=None):
        txt_path = os.path.join(data_dir, '{0}.txt'.format(split))
        imgsA = []
        imgsP = []
        imgsN = []
        labelsP = []
        labelsN = []
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
            labelsP.append(int(words[0]) - 1)
            labelsN.append(int(words[3]) - 1)

        self.imgsA = imgsA  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsP = imgsP  # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.imgsN = imgsN
        self.labelsP = labelsP
        self.labelsN = labelsN
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn0 = self.imgsA[index]
        fn1 = self.imgsP[index]
        fn2 = self.imgsN[index]
        lp = torch.from_numpy(np.array(self.labelsP[index]))
        ln = torch.from_numpy(np.array(self.labelsN[index]))

        img0 = Image.open(fn0).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img1 = Image.open(fn1).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img2 = Image.open(fn2).convert('RGB')  # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img0 = self.transform(img0)  # 在这里做transform，转为tensor等等
            img1 = self.transform(img1)  # 在这里做transform，转为tensor等等
            img2 = self.transform(img2)  # 在这里做transform，转为tensor等等

        return img0, img1, img2, lp, ln

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
        self.trainvalTransform = None
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

        trainval_transforms_list = []
        if self.opt.crop_size != 0:
            trainval_transforms_list.append(transforms.CenterCrop(self.opt.crop_size))
        if self.opt.resize_size != 0:
            trainval_transforms_list.append(transforms.Resize(self.opt.resize_size))
        trainval_transforms_list.append(transforms.ToTensor())
        trainval_transforms_list.append(normTransform)

        self.trainvalTransform = transforms.Compose(trainval_transforms_list)

        test_transforms_list = trainval_transforms_list[:-1]
        test_transforms_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

        self.testTransform = transforms.Compose(test_transforms_list)

    def select_dataloader(self, data_type='train'):
        if data_type not in ['train', 'valid', 'test']:
            return None
        # 构建Data
        if self.opt.face_data == 'lfw':
            # 选择Dataset参数
            if data_type == 'train':
                split = 'lfw_train_triplet'
                transform = self.trainvalTransform
                batch_size = self.opt.train_bs
                shuffle = True
                # 构建Data
                data = LWFTripletDataset(dataset_dir=self.opt.imgset_data_dir, img_dir=self.opt.img_data_dir,
                                         split=split, transform=transform)
            elif data_type == 'valid':
                split = 'lfw_valid_pairs'
                transform = self.trainvalTransform
                batch_size = self.opt.valid_bs
                shuffle = False
                # 构建Data
                data = LWFPairsDataset(dataset_dir=self.opt.imgset_data_dir, img_dir=self.opt.img_data_dir,
                                       split=split, transform=transform)
            else:
                split = 'lfw_test_pairs'
                transform = self.testTransform
                batch_size = self.opt.test_bs
                shuffle = False
                # 构建Data
                data = LWFPairsDataset(dataset_dir=self.opt.imgset_data_dir, img_dir=self.opt.img_data_dir,
                                       split=split, transform=transform)
            # 构建DataLoder
            data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
        elif self.opt.face_data == 'celeba':
            pass
            return None
        else:
            print('No Face Data !')
            return None
        print(self.opt.face_data + ' ' + data_type + ' data number : ', len(data))
        return data_loader


# 模型类 ========================================

class FaceRecognitionNetPool:
    def __init__(self, opt_use):
        self.opt = opt_use
        self.input_net_size = self.opt.input_net_size
        self.n_class = self.opt.n_people

    def select_model(self, net_name):
        if net_name == '' or net_name is None:
            return None
        if net_name == 'resnet18':
            net = resnet18(pretrained=True)
        elif net_name == 'resnet34':
            net = resnet34(pretrained=True)
        elif net_name == 'resnet50':
            net = resnet50(pretrained=True)
        elif net_name == 'facenet':
            net = FaceModel(self.n_class, self.opt.alpha)
        else:
            return None
        # 更改avgpool
        # avg_pool_size = int(self.input_img_size / 32)
        # net.avgpool = nn.AvgPool2d(kernel_size=avg_pool_size, stride=1)
        # 修改fc
        if net_name == 'vgg16':
            fc_features = net.classifier[6].in_features
            net.classifier[6] = nn.Linear(fc_features, 128)
        elif net_name == 'facenet':
            pass
        else:
            fc_features = net.fc.in_features
            net.fc = nn.Linear(fc_features, 128)

        return net


class FaceModel(nn.Module):
    def __init__(self, n_class, alpha):
        super(FaceModel, self).__init__()
        self.features = None
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(512 * 7 * 7, 128)
        self.model.classifier = nn.Linear(128, n_class)
        print('net n_class: ', n_class)
        self.alpha = alpha

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
        self.features *= self.alpha
        return self.features

    def forward_classifier(self, x):
        features = self.forward(x)
        res = self.model.classifier(features)
        return res
