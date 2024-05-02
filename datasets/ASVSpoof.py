import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import scipy.io
import random


class ASVSpoof(Dataset):

    def __init__(self, root, split="Train", ext='tiff', partition='CQT', sliding_window = 32, frames = 5, img_size = 32, transform=None):

        self.root = root

        self.transform = transform

        self.sliding_window = sliding_window

        self.frames = frames

        self.resize = img_size

        imgs = list(Path(os.path.join(root, split, partition)).rglob("**/*.{}".format(ext)))

        if('Train' in split):

            ap_index = np.array([i for i in range(len(imgs)) if 'spoof' in str(imgs[i])])

            bp_index = [i for i in range(len(imgs)) if 'bonafide' in str(imgs[i])]

            random.shuffle(ap_index)

            ap_index = list(ap_index[:len(bp_index)])

            imgs = [*list(np.asarray(imgs)[ap_index]), *list(np.asarray(imgs)[bp_index])]

        self.total_image, self.labels = [], []

        for c in imgs:

            self.labels.append(1 if 'bonafide' in str(c) else 0)

            self.total_image.append(c)


    def __len__(self):

        return len(self.total_image)


    def __getitem__(self, idx):

        path_cqt = self.total_image[idx]

        mat = scipy.io.loadmat(path_cqt, verify_compressed_data_integrity=False)

        m = mat['features']

        img_cqt = Image.fromarray(m).convert("RGB")

        # _, h = img_cqt.size

        img_cqt = img_cqt.resize((self.sliding_window*self.frames, self.resize))

        narray_cqt = np.array(img_cqt)

        images_cqt = []

        for i in range(self.frames):

            im_tmp = narray_cqt[:, i*self.sliding_window: i*self.sliding_window + self.sliding_window, :]

            images_cqt.append(Image.fromarray(im_tmp))

        if self.transform is not None:

            tensor_image_cqt = self.transform(images_cqt)

        return tensor_image_cqt, self.labels[idx]
