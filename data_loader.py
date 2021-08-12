# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
from pathlib import Path

from feature_extraction import resnet_transform
import h5py
import numpy as np

from util.file_process import Logger, read_json, write_json


class VideoData(Dataset):
    def __init__(self, root, preprocessed=True, transform=resnet_transform, with_name=False):
        self.root = root
        self.preprocessed = preprocessed
        self.transform = transform
        self.with_name = with_name
        # self.video_list = list(self.root.iterdir())
        self.splits = read_json("dataset_tvsum/splits.json")
        self.split = self.splits[0]
        self.train_keys = self.split["train_keys"]
        self.test_keys = self.split["test_keys"]

    def __len__(self):
        # return len(self.video_list)
        if self.with_name:
            return len(self.test_keys)
        else:
            return len(self.train_keys)


    def __getitem__(self, index):
        if self.preprocessed:
            # image_path = self.video_list[index]
            key = self.train_keys[index]

            # with h5py.File(image_path, 'r') as f:
            with h5py.File("dataset_tvsum/data.h5", 'r') as dataset:
                if self.with_name:
                    # return torch.Tensor(np.array(f['pool5'])), image_path.name[:-5]
                    try:
                        print("GOT FEATURES!!!!!!!!!!!!!!!!!")
                        print(key)
                        return torch.Tensor(np.array(dataset[key]['features'])), dataset[key]['video_name']
                    except Exception:
                        print("FAILED TO GET FEATURES!!!!!!!!!!!!!!!!!")
                        print(key)
                        return None
                else:
                    # return torch.Tensor(np.array(f['pool5']))
                    try:
                        print("GOT FEATURES11111111!!!!!!!!!!!!!!!!!")
                        print(np.array(dataset[key]['video_name']))
                        print(key)
                        return torch.Tensor(np.array(dataset[key]['features']))
                    except Exception:
                        print("FAILED TO GET FEATURES111111111111!!!!!!!!!!!!!!!!!")
                        print(np.array(dataset[key]['video_name']))
                        print(key)
                        return None


        else:
            images = []
            for img_path in Path(self.video_list[index]).glob('*.jpg'):
                img = default_loader(img_path)
                img_tensor = self.transform(img)
                images.append(img_tensor)

            return torch.stack(images), img_path.parent.name[4:]


def get_loader(root, mode):
    if mode.lower() == 'train':
        return DataLoader(VideoData(root), batch_size=1)
    else:
        return VideoData(root, with_name=True)


if __name__ == '__main__':
    pass
