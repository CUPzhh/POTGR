import random
import numpy as np
import os
from torch.utils.data import Dataset as TorchDataset
from PIL import Image


CLASSNAME_CUB200 = ['Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet', 'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird', 'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting', 'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow', 'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird', 'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo', 'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher', 'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher', 'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch', 'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe', 'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak', 'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull', 'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird', 'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay', 'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher', 'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark', 'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird', 'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole', 'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit', 'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx', 'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow', 'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow', 'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow', 'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow', 'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow', 'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern', 'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo', 'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo', 'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler', 'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler', 'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler', 'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler', 'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler', 'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing', 'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker', 'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren', 'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat']


class CUB200(TorchDataset):
    def __init__(self, data_root, tfm, task_id, mode, shot=5, class_per_task=10, B=7):
        self.tfm = tfm
        self.mode = mode
        self.task_id = task_id
        self.shot = shot
        self.class_per_task = class_per_task
        self.B = B

        cub_root = os.path.join(data_root, 'CUB_200_2011')
        image_data_txt = os.path.join(cub_root, 'images.txt')
        image_root = os.path.join(cub_root, 'images')
        label_txt = os.path.join(cub_root, 'image_class_labels.txt')
        train_test_split = os.path.join(cub_root, 'train_test_split.txt')

        # 创建映射关系
        image_id_split, image_id_path_dict, image_id_label_dict = {}, {}, {}

        with open(train_test_split, 'r') as f:
            for line in f:
                image_id, is_train = line.strip().split(" ")
                image_id_split[image_id] = int(is_train)

        with open(image_data_txt, 'r') as f:
            for line in f:
                image_id, path = line.strip().split(" ")
                image_id_path_dict[image_id] = os.path.join(image_root, path)

        with open(label_txt, 'r') as f:
            for line in f:
                image_id, label = line.strip().split(" ")
                image_id_label_dict[image_id] = int(label) - 1  # labels 从 0 开始

        # 生成 task 分配
        self.novel_task_len = int(100 / class_per_task)
        self.task_len = self.novel_task_len + 1
        task_split = [[] for _ in range(self.task_len)]

        for i in range(100):
            task_split[0].append(i)
        for i in range(1, self.task_len):
            for j in range(class_per_task):
                task_split[i].append(100 + (i - 1) * class_per_task + j)

        self.select_class_id = task_split[task_id]
        self.end_class_id = self.select_class_id[-1]

        self.images_list, self.labeled_list = [], []

        if mode == 'train':
            self.class_idx_dict = {x: [] for x in self.select_class_id}
            for key in image_id_path_dict:
                if image_id_split[key] == 1:
                    label = image_id_label_dict[key]
                    if label in self.class_idx_dict:
                        self.class_idx_dict[label].append(key)

            for c in self.select_class_id:
                idx_list = self.class_idx_dict[c]
                if c >= 100:
                    idx_list = random.sample(idx_list, shot)
                for id in idx_list:
                    self.images_list.append(image_id_path_dict[id])
                    self.labeled_list.append(image_id_label_dict[id])

        else:
            task_to_id_end = {0: 100}
            start = 100 + class_per_task
            for i in range(1, self.task_len):
                task_to_id_end[i] = start
                start += class_per_task

            valid_class_ids = list(range(task_to_id_end[task_id]))

            for key in image_id_path_dict:
                if image_id_split[key] == 0:
                    label = image_id_label_dict[key]
                    if label in valid_class_ids:
                        self.images_list.append(image_id_path_dict[key])
                        self.labeled_list.append(label)

        self.len = len(self.images_list)

    def __getitem__(self, idx):
        image_path = self.images_list[idx]
        label = self.labeled_list[idx]
        image = Image.open(image_path).convert('RGB')
        if self.tfm:
            image = self.tfm(image)

        if self.mode == 'train' and self.task_id > 0:
            pseudo_label = [random.randint(0, self.end_class_id - self.class_per_task) for _ in range(self.B)]
            pseudo_label = np.array(pseudo_label)
            return image, label, pseudo_label

        return image, label

    def __len__(self):
        return self.len
