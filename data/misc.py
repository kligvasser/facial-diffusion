import os
import lmdb
import pickle
import cv2
import numpy as np


IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']


def get_path_list(path_dir, max_size=None, extentions=IMAGE_EXTENSIONS):
    paths = list()

    for dirpath, _, files in os.walk(path_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(tuple(extentions)):
                paths.append(fname)

    return sorted(paths)[:max_size]


def extract_frames_and_landmarks_from_lmdb(lmdb_file, frame_numbers, resize=None):
    frames, landmarks = list(), list()

    env = lmdb.open(lmdb_file, readonly=True)
    for frame_number in frame_numbers:
        with env.begin() as txn:
            key = str(frame_number).zfill(10)
            value = txn.get(key.encode())

        data_dict = pickle.loads(value)
        frame = data_dict['image']
        frame = cv2.imdecode(np.frombuffer(frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        if resize:
            frame = cv2.resize(frame, resize, interpolation=cv2.INTER_CUBIC)
        frames.append(frame)
        landmarks.append(data_dict['landmarks'])

    env.close()
    return frames, landmarks


def landmarks_to_2d_mask(landmarks, size=256):
    mask = np.zeros((1, size, size))

    x_indices = (np.clip(landmarks[:, 0], 0, 1) * (size - 1)).astype(int)
    y_indices = (np.clip(landmarks[:, 1], 0, 1) * (size - 1)).astype(int)

    mask[:, y_indices, x_indices] = 1

    return mask


def landmarks_to_3d_mask(landmarks, size=256):
    mask = np.zeros((1, size, size))

    x_indices = (np.clip(landmarks[:, 0], 0, 1) * (size - 1)).astype(int)
    y_indices = (np.clip(landmarks[:, 1], 0, 1) * (size - 1)).astype(int)

    values = landmarks[:, 2]
    values[values > 0] += 0.5
    values[values < 0] -= 0.5
    values = values / max(abs(values))

    mask[:, y_indices, x_indices] = values

    return mask
