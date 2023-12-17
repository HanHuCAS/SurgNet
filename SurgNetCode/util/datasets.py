# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms
from torchvision.datasets.folder import default_loader

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import numpy as np
from torch.utils.data import Dataset
import lmdb
import cv2
import glob
from torchvision.utils import save_image

class Dataset2(Dataset):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class ImageDataset_LMY2(Dataset2):
    def __init__(self, root, transform):
        self.database_offset_dict = {}
        self.frame_offset_dict = {}
        self.transform = transform
        lmy_dataset_box = glob.glob(root + '/*/')
        self.sum_num = 0
        self.lmy_dataset_box = []
        self.frame_box = []
        for path in lmy_dataset_box:
            is_close = 1
            env = lmdb.open(path, map_size=1099511627776 // 8, readonly=True, lock=False)
            with env.begin() as txn:
                num_img = txn.get(("num_img").encode())
                print(path)
                print(num_img)
                if not num_img is None:
                    self.sum_num += int(num_img)
                    self.lmy_dataset_box.append(env)
                    self.frame_box.append(int(num_img))
                    is_close = 0
            if is_close == 1:        
                env.close()
        print(self.sum_num)  
        print("load")
        for index in range(self.sum_num):
            index_ = index
            for i in range(len(self.lmy_dataset_box)):
                if index < self.frame_box[i]:
                    frame_offset = index
                    database_offset = i
                    break
                else:
                    index -= self.frame_box[i]
            self.database_offset_dict[index_] = database_offset
            self.frame_offset_dict[index_] = frame_offset
        print('OK')
        self.pool = []
    def __getitem__(self, index):
        if len(self.pool):
            return self.pool.pop(index % len(self.pool))
        else:
            for i in range(64):
                frame_offset = 0
                database_offset = 0
                database_offset = self.database_offset_dict[index]
                frame_offset = self.frame_offset_dict[index]
                env = self.lmy_dataset_box[database_offset]
                    
                with env.begin() as txn:
                    value = txn.get(("image:"+str((frame_offset+i)%self.frame_box[database_offset])).encode())
                    image_buf = np.asarray(bytearray(value), dtype="uint8")
                    img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = PIL.Image.fromarray(img)
                    img = self.transform(img)
                self.pool.append(img)   
            return self.pool.pop(index % len(self.pool))
    def __len__(self):
        return self.sum_num


class ImageDataset_LMY(Dataset2):
    def __init__(self, root, transform):
        self.database_offset_dict = {}
        self.frame_offset_dict = {}
        self.transform = transform
        lmy_dataset_box = glob.glob(root + '/*/')
        self.sum_num = 0
        self.lmy_dataset_box = []
        self.frame_box = []
        for path in lmy_dataset_box:
            is_close = 1
            env = lmdb.open(path, map_size=1099511627776 // 4, readonly=True, lock=False)
            with env.begin() as txn:
                num_img = txn.get(("num_img").encode())
                print(path)
                print(num_img)
                if not num_img is None:
                    self.sum_num += int(num_img)
                    self.lmy_dataset_box.append(env)
                    self.frame_box.append(int(num_img))
                    is_close = 0
            if is_close == 1:        
                env.close()
        print(self.sum_num)  
        print("load")
        for index in range(self.sum_num):
            index_ = index
            for i in range(len(self.lmy_dataset_box)):
                if index < self.frame_box[i]:
                    frame_offset = index
                    database_offset = i
                    break
                else:
                    index -= self.frame_box[i]
            self.database_offset_dict[index_] = database_offset
            self.frame_offset_dict[index_] = frame_offset
        print('OK')
    def __getitem__(self, index):
        frame_offset = 0
        database_offset = 0
        database_offset = self.database_offset_dict[index]
        frame_offset = self.frame_offset_dict[index]
        env = self.lmy_dataset_box[database_offset]
            
        with env.begin() as txn:
            value = txn.get(("image:"+str(frame_offset)).encode())
            image_buf = np.asarray(bytearray(value), dtype="uint8")
            img = cv2.imdecode(image_buf, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(img)

        #env.close()
        return self.transform(img)
    def __len__(self):
        return self.sum_num

class ImageListFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None,
                 ann_file=None, loader=default_loader):
        self.root = root
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.nb_classes = 1000

        assert ann_file is not None
        print('load info from', ann_file)

        self.samples = []
        ann = open(ann_file)
        for elem in ann.readlines():
            cut = elem.split(' ')
            path_current = os.path.join(root, cut[0])
            target_current = int(cut[1])
            self.samples.append((path_current, target_current))
        ann.close()

        print('load finish')


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # TODO modify your own dataset here
    folder = os.path.join(args.data_path, 'train' if is_train else 'val')
    ann_file = os.path.join(args.data_path, 'train.txt' if is_train else 'val.txt')
    dataset = ImageListFolder(folder, transform=transform, ann_file=ann_file)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
