import random
import os.path as osp
import numpy as np
import os
from torch.utils.data import Dataset as TorchDataset
from PIL import Image


CLASSNAME_miniImageNet = [
    'house_finch', 'robin', 'triceratops', 'green_mamba', 'harvestman', 'toucan', 'goose', 'jellyfish', 'nematode', 'king_crab',
    'dugong', 'Walker_hound', 'Ibizan_hound', 'Saluki', 'golden_retriever', 'Gordon_setter', 'komondor', 'boxer', 'Tibetan_mastiff', 'French_bulldog',
    'malamute', 'dalmatian', 'Newfoundland', 'miniature_poodle', 'white_wolf', 'African_hunting_dog', 'Arctic_fox', 'lion', 'meerkat', 'ladybug',
    'rhinoceros_beetle', 'ant', 'black-footed_ferret', 'three-toed_sloth', 'rock_beauty', 'aircraft_carrier', 'ashcan', 'barrel', 'beer_bottle', 'bookshop',
    'cannon', 'carousel', 'carton', 'catamaran', 'chime', 'clog', 'cocktail_shaker', 'combination_lock', 'crate', 'cuirass',
    'dishrag', 'dome', 'electric_guitar', 'file', 'fire_screen', 'frying_pan', 'garbage_truck', 'hair_slide', 'holster', 'horizontal_bar',
    'hourglass', 'iPod', 'lipstick', 'miniskirt', 'missile', 'mixing_bowl', 'oboe', 'organ', 'parallel_bars', 'pencil_box',
    'photocopier', 'poncho', 'prayer_rug', 'reel', 'school_bus', 'scoreboard', 'slot', 'snorkel', 'solar_dish', 'spider_web',
    'stage', 'tank', 'theater_curtain', 'tile_roof', 'tobacco_shop', 'unicycle', 'upright', 'vase', 'wok', 'worm_fence',
    'yawl', 'street_sign', 'consomme', 'trifle', 'hotdog', 'orange', 'cliff', 'coral_reef', 'bolete', 'ear'
]

class MiniImageNet(TorchDataset):
    def __init__(self, data_root, tfm, task_id, mode, class_per_task=5, B=5):
        root = os.path.join(data_root, 'miniimagenet')
        self.IMAGE_PATH = os.path.join(root, 'images')
        self.SPLIT_PATH = os.path.join(root, 'split')
        self.index_list = os.path.join(root, "index_list/mini_imagenet")
        self.tfm = tfm
        self.class_per_task = class_per_task
        self.B = B
        self.mode = mode
        self.task_id = task_id

        task_split = {0: 60, 1: 65, 2: 70, 3: 75, 4: 80, 5: 85, 6: 90, 7: 95, 8: 100}
        self.end_class_id = task_split[task_id] - 1

        csv_path = osp.join(self.SPLIT_PATH, mode + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        self.data = []
        self.targets = []
        self.data2label = {}
        lb = -1
        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            path = osp.join(self.IMAGE_PATH, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            self.data.append(path)
            self.targets.append(lb)
            self.data2label[path] = lb

        if mode == 'train':
            if task_id == 0:
                self.data, self.targets = self.SelectfromClasses(self.data, self.targets, np.arange(task_split[0]))
            else:
                index_path = os.path.join(self.index_list, f'session_{task_id + 1}.txt')
                self.data, self.targets = self.SelectfromTxt(self.data2label, index_path)
        else:
            self.data, self.targets = self.SelectfromClasses(self.data, self.targets, np.arange(task_split[task_id]))

    def SelectfromTxt(self, data2label, index_path):
        lines = [x.strip() for x in open(index_path, 'r').readlines()]
        data_tmp, targets_tmp = [], []
        for line in lines:
            img_name = line.split('/')[3]
            img_path = os.path.join(self.IMAGE_PATH, img_name)
            data_tmp.append(img_path)
            targets_tmp.append(data2label[img_path])
        return data_tmp, targets_tmp

    def SelectfromClasses(self, data, targets, index):
        data_tmp, targets_tmp = [], []
        for i in index:
            indices = np.where(i == targets)[0]
            for j in indices:
                data_tmp.append(data[j])
                targets_tmp.append(targets[j])
        return data_tmp, targets_tmp

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, target = self.data[i], self.targets[i]
        image = self.tfm(Image.open(path).convert('RGB'))

        if self.mode == 'train' and self.task_id > 0:
            pseudo_label = []
            for _ in range(self.B):
                cls = random.randint(0, self.end_class_id - self.class_per_task)
                pseudo_label.append(cls)
            pseudo_label = np.array(pseudo_label)
            return image, target, pseudo_label

        return image, target
