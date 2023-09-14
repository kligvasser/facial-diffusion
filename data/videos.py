import torch
import numpy as np
import random

import data.misc
import utils.misc

from einops import rearrange

FRAMES_PER_SECONDS = 25.0


class VideoLMDB(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path='',
        num_frames=8,
        frame_skip=3,
        p_random_crop=0.5,
        size=256,
        min_resolution=256,
        max_size=None,
    ):
        self.csv_path = csv_path
        self.num_frames = num_frames
        self.frame_skip = frame_skip
        self.p_random_crop = p_random_crop
        self.size = size
        self.min_resolution = min_resolution
        self.max_size = max_size

        self._init()

    def _init(self):
        def is_hd(resolution):
            return min(resolution) > self.min_resolution

        def is_long(valid):
            return len(valid) > self.num_frames * self.frame_skip

        df = utils.misc.load_df(self.csv_path)
        df['hd'] = df.apply(lambda row: is_hd([int(row['height']), int(row['width'])]), axis=1)
        df['long'] = df.apply(lambda row: is_long(row['valid_frames']), axis=1)
        df = df[df['hd'] == True].reset_index(drop=True)
        df = df[df['long'] == True].reset_index(drop=True)
        self.records = df.to_dict('records')

        if self.max_size:
            self.records = self.records[: self.max_size]

    def _preprocess_images(self, images):
        images = np.concatenate(images, axis=-1)
        images = (images / 127.5 - 1.0).astype(np.float32)

        if images.shape[0] > self.size:
            top_width = random.randint(0, images.shape[1] - self.size - 1)
            top_height = random.randint(0, images.shape[0] - self.size - 1)
            images = images[
                top_height : (top_height + self.size), top_width : (top_width + self.size), :
            ]

        images = rearrange(images, 'h w c -> c h w')

        return images

    def _get_frame_numbers(self, valid):
        start_index = random.randint(0, len(valid) - self.num_frames * self.frame_skip - 1)
        return valid[
            start_index : (start_index + self.num_frames * self.frame_skip) : self.frame_skip
        ]

    def __getitem__(self, index):
        example = dict()

        record = self.records[index]
        frame_numbers = self._get_frame_numbers(record['valid_frames'])
        new_size = (
            self.size
            if random.random() > self.p_random_crop
            else random.randint(self.size, min(int(record['height']), int(record['width'])) - 1)
        )

        frames, _ = data.misc.extract_frames_and_landmarks_from_lmdb(
            record['lmdb_path'], frame_numbers, (new_size, new_size)
        )
        example['image'] = self._preprocess_images(frames)
        example['path'] = record['lmdb_path']
        example['indexes'] = frame_numbers

        return example

    def __len__(self):
        return len(self.records)


class VideoPairedLMDB(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path='',
        keys=['dense'],
        frame_skip=4,
        size=256,
        min_resolution=320,
        p_flip=0.5,
        max_size=None,
    ):
        self.csv_path = csv_path
        self.keys = keys
        self.num_frames = 2
        self.frame_skip = frame_skip
        self.size = size
        self.min_resolution = min_resolution
        self.p_flip = p_flip
        self.max_size = max_size

        self._init()

    def _init(self):
        def is_hd(resolution):
            return min(resolution) > self.min_resolution

        def is_long(valid):
            return len(valid) > self.num_frames * self.frame_skip

        df = utils.misc.load_df(self.csv_path)
        df['hd'] = df.apply(lambda row: is_hd([int(row['height']), int(row['width'])]), axis=1)
        df['long'] = df.apply(lambda row: is_long(row['valid_frames']), axis=1)
        df = df[df['hd'] == True].reset_index(drop=True)
        df = df[df['long'] == True].reset_index(drop=True)
        self.records = df.to_dict('records')

        if self.max_size:
            self.records = self.records[: self.max_size]

    def _preprocess_image(self, image):
        image = (image / 127.5 - 1.0).astype(np.float32)
        image = rearrange(image, 'h w c -> c h w')
        return image

    def _get_frame_numbers(self, valid):
        start_index = random.randint(0, len(valid) - self.num_frames * self.frame_skip - 1)
        return valid[
            start_index : (start_index + self.num_frames * self.frame_skip) : self.frame_skip
        ]

    def __getitem__(self, index):
        example = dict()

        record = self.records[index]
        frame_numbers = self._get_frame_numbers(record['valid_frames'])

        if random.random() > self.p_flip:
            frame_numbers.reverse()

        frames, landmarks = data.misc.extract_frames_and_landmarks_from_lmdb(
            record['lmdb_path'], frame_numbers, (self.size, self.size)
        )
        example['image'] = self._preprocess_image(frames[0])
        example['reference'] = self._preprocess_image(frames[1])
        example['human_label'] = 'human_label'

        for key in self.keys:
            example[key] = landmarks[0][key]

        return example

    def __len__(self):
        return len(self.records)
