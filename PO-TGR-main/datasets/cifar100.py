import random
import numpy as np
import os
import torchvision
from torch.utils.data import Dataset as TorchDataset


CLASSNAME_CFIAR100 = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
    'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard',
    'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree',
    'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea',
    'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
    'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
    'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm'
]


class CIFAR100(TorchDataset):
    def __init__(self, tfm, task_id, mode, shot=5, class_per_task=5, B=8) -> None:
        self.class_per_task = class_per_task
        self.task_id = task_id
        self.mode = mode
        self.B = B
        self.shot = shot

        # 构建任务划分（base: 0-59, 新类: 60-99）
        task_split = [[] for _ in range(9)]
        for i in range(60):
            task_split[0].append(i)
        for i in range(1, 9):
            for j in range(5):
                task_split[i].append(60 + (i - 1) * 5 + j)

        self.select_class_id = task_split[task_id]
        self.end_class_id = self.select_class_id[-1]

        if mode == 'train':
            cifar100 = torchvision.datasets.CIFAR100(
                root=os.path.expanduser("~/.cache"),
                train=True,
                download=False,
                transform=tfm
            )
        else:
            cifar100 = torchvision.datasets.CIFAR100(
                root=os.path.expanduser("~/.cache"),
                train=False,
                download=False,
                transform=tfm
            )

        self.class_idx_dict = {x: [] for x in self.select_class_id}
        for i in range(len(cifar100)):
            image, label = cifar100[i]
            if label in self.class_idx_dict:
                self.class_idx_dict[label].append(i)

        self.data = []
        for c in self.select_class_id:
            idx_list = self.class_idx_dict[c]
            if mode == 'train' and c >= 60:
                idx_list = random.sample(idx_list, shot)
            for idx in idx_list:
                self.data.append(cifar100[idx])

        self.len = len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]

        # 在训练阶段（非task0）返回伪label列表（用于生成图像）
        if self.mode == 'train' and self.task_id > 0:
            pseudo_label = []
            for _ in range(self.B):
                cls = random.randint(0, self.end_class_id - self.class_per_task)
                pseudo_label.append(cls)
            pseudo_label = np.array(pseudo_label)
            return image, label, pseudo_label

        return image, label

    def __len__(self):
        return self.len
