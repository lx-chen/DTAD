import numpy as np
import os, sys
from datasets.base_dataset import BaseADDataset
from PIL import Image
from torchvision import transforms
from datasets.cutmix import CutMix
import random
from utils import t2np
import torch
import random

    
class MVTecAD(BaseADDataset):

    def __init__(self, args, train_type, train = True):
        super(MVTecAD).__init__()
        self.args = args
        self.train = train
        self.train_type = train_type
        self.classname = self.args.classname
        self.know_class = self.args.know_class
        self.pollution_rate = self.args.cont_rate
        if self.args.test_threshold == 0 and self.args.test_rate == 0:
            self.test_threshold = self.args.nAnomaly
        else:
            self.test_threshold = self.args.test_threshold

        self.root = os.path.join(self.args.dataset_root, self.classname)
        self.transform = self.transform_train() if self.train else self.transform_test()
        # self.transform_pseudo = self.transform_pseudo()

        # 添加数据集根目录
        normal_data = list()
        split = 'train'
        normal_files = os.listdir(os.path.join(self.root, split, 'good'))
        for file in normal_files:
            if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                normal_data.append(split + '/good/' + file)

        self.nPollution = int((len(normal_data)/(1-self.pollution_rate)) * self.pollution_rate)
        if self.test_threshold==0 and self.args.test_rate>0:
            self.test_threshold = int((len(normal_data)/(1-self.args.test_rate)) * self.args.test_rate) + self.args.nAnomaly

        self.ood_data = self.get_ood_data()

        if self.train is False:
            normal_data = list()
            split = 'test'
            normal_files = os.listdir(os.path.join(self.root, split, 'good'))
            for file in normal_files:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    normal_data.append(split + '/good/' + file)

        # ref image loader，随机选10个正常图片作为参考,并transform之后取平均
        ref_data = list()
        ref_data = random.sample(normal_data, 10)
        # ref_data = normal_data
        ref_images = []  # 存储10个参考图片的张量
        for i in range(10):
            ref_image = self.load_image(os.path.join(self.root, ref_data[i]))
            transformed_ref_image = self.transform(ref_image)
            ref_images.append(transformed_ref_image)
        # 将张量列表堆叠起来，并计算平均值
        stacked_ref_images = torch.stack(ref_images, dim=0)
        self.ref_image = torch.mean(stacked_ref_images, dim=0)

        outlier_data, pollution_data = self.split_outlier()
        outlier_data.sort()

        normal_data = normal_data + pollution_data

        normal_label = np.zeros(len(normal_data)).tolist()
        outlier_label = np.ones(len(outlier_data)).tolist()


        if self.train_type  == "normal_set":
            self.images = normal_data       # 正常的图片
            self.labels = np.array(normal_label)
        else:
            self.images = normal_data + outlier_data     # 训练或者测试图片
            self.labels = np.array(normal_label + outlier_label)    

        self.normal_idx = np.argwhere(self.labels == 0).flatten()
        self.outlier_idx = np.argwhere(self.labels == 1).flatten()


    def get_ood_data(self):
        ood_data = list()
        if self.args.outlier_root is None:
            return None
        dataset_classes = os.listdir(self.args.outlier_root)
        for cl in dataset_classes:
            if cl == self.args.classname:
                continue
            cl_root = os.path.join(self.args.outlier_root, cl, 'train', 'good')
            ood_file = os.listdir(cl_root)
            for file in ood_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    ood_data.append(os.path.join(cl_root, file))
        return ood_data

    def split_outlier(self):
        outlier_data_dir = os.path.join(self.root, 'test')
        outlier_classes = os.listdir(outlier_data_dir)
        if self.know_class in outlier_classes:
            print("Know outlier class: " + self.know_class)
            outlier_data = list()
            know_class_data = list()
            for cl in outlier_classes:
                if cl == 'good':
                    continue
                outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
                for file in outlier_file:
                    if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                        if cl == self.know_class:
                            know_class_data.append('test/' + cl + '/' + file)
                        else:
                            outlier_data.append('test/' + cl + '/' + file)
            np.random.RandomState(self.args.ramdn_seed).shuffle(know_class_data)
            know_outlier = know_class_data[0:self.args.nAnomaly]
            unknow_outlier = outlier_data
            if self.train:
                return know_outlier, list()
            else:
                return unknow_outlier, list()


        outlier_data = list()
        for cl in outlier_classes:
            if cl == 'good':
                continue
            outlier_file = os.listdir(os.path.join(outlier_data_dir, cl))
            for file in outlier_file:
                if 'png' in file[-3:] or 'PNG' in file[-3:] or 'jpg' in file[-3:] or 'npy' in file[-3:]:
                    outlier_data.append('test/' + cl + '/' + file)
        np.random.RandomState(self.args.ramdn_seed).shuffle(outlier_data)
        if self.train:
            # 训练模式：在这种情况下，函数返回 outlier_data 列表中前 self.args.nAnomaly 个数据作为已知异常数据，
            # 以及从 self.args.nAnomaly 位置开始直到 self.args.nAnomaly + self.nPollution 位置的数据作为污染数据
            return outlier_data[0:self.args.nAnomaly], outlier_data[self.args.nAnomaly:self.args.nAnomaly + self.nPollution]
        else:
            # 测试模式：在这种情况下，函数返回 outlier_data 列表中从 self.test_threshold 位置开始到列表末尾的数据作为未知异常数据，
            # 并返回一个空列表作为污染数据，表示在测试时不使用污染数据。
            return outlier_data[self.test_threshold:], list()

    def load_image(self, path):
        if 'npy' in path[-3:]:
            img = np.load(path).astype(np.uint8)
            img = img[:, :, :3]
            return Image.fromarray(img)
        return Image.open(path).convert('RGB')

    def transform_train(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size,self.args.img_size)),
            transforms.RandomRotation(180),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms


    def transform_test(self):
        composed_transforms = transforms.Compose([
            transforms.Resize((self.args.img_size, self.args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        return composed_transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):

        file_name = self.images[index]    
        image = self.load_image(os.path.join(self.root, self.images[index]))
        ref_image = self.ref_image # 去平均后的ref_image
        transform = self.transform
        label = self.labels[index]
        sample = {'image': transform(image), 'ref_image': ref_image, 'label': label, 'file_name': file_name}
        return sample
    


 