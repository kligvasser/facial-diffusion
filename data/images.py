import torch
import os
import numpy as np

from PIL import Image
from torchvision import transforms

import data.misc
import utils.misc


class ImagesLibary(torch.utils.data.Dataset):
    def __init__(self, root='', image_size=None, max_size=None):
        self.root = root
        self.image_size = image_size
        self.max_size = max_size

        self._init()

    def _init(self):
        self.paths = data.misc.get_path_list(self.root, self.max_size)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_image(self, path):
        image = Image.open(path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)
        return image

    def __getitem__(self, index):
        example = dict()

        path = self.paths[index]
        example['image'] = self.preprocess_image(path)
        example['path'] = path

        return example

    def __len__(self):
        return len(self.paths)


class ImageLandmarks(torch.utils.data.Dataset):
    def __init__(
        self,
        df_path='',
        keys=['arcface', 'dense'],
        image_size=None,
        max_size=None,
    ):
        self.df_path = df_path
        self.keys = keys
        self.image_size = image_size
        self.max_size = max_size
        self.dir_name = os.path.dirname(df_path)

        self._init()

    def _init(self):
        self.df = utils.misc.load_df(self.df_path)

        if self.max_size:
            self.df = self.df.sample(min(len(self.df), self.max_size)).reset_index(drop=True)

        self.preprocess = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def preprocess_image(self, path):
        image = Image.open(path)
        if not image.mode == 'RGB':
            image = image.convert('RGB')
        image = self.preprocess(image)
        return image

    def __getitem__(self, index):
        example = dict()
        df_row = self.df.iloc[index]
        path = os.path.join(self.dir_name, df_row['relative'])

        example['image'] = self.preprocess_image(path)
        example['path'] = df_row['path']
        example['human_label'] = df_row['basename']

        for key in self.keys:
            example[key] = df_row[key].astype(np.float32)

        return example

    def __len__(self):
        return len(self.df)
