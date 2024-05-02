import os
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import torch
import scipy.io
import random


class ASVSpoofH(Dataset):

    def __init__(self, root, split="Train", ext='tiff', sliding_window = 32, frames = 5, img_size = 32, transform=None):

        self.root = root

        self.transform = transform

        self.sliding_window = sliding_window

        self.frames = frames

        self.resize = img_size

        cqt_imgs = list(Path(os.path.join(root, split, 'CQT')).rglob("**/*.{}".format(ext)))

        stft_imgs = list(Path(os.path.join(root, split, 'STFT')).rglob("**/*.{}".format(ext)))

        # if('Train' in split):

        #     ap_index = np.array([i for i in range(len(cqt_imgs)) if 'spoof' in str(cqt_imgs[i])])

        #     bp_index = [i for i in range(len(cqt_imgs)) if 'bonafide' in str(cqt_imgs[i])]

        #     random.shuffle(ap_index)

        #     ap_index = list(ap_index[:len(bp_index)])

        #     cqt_imgs = [*list(np.asarray(cqt_imgs)[ap_index]), *list(np.asarray(cqt_imgs)[bp_index])]

        #     stft_imgs = [*list(np.asarray(stft_imgs)[ap_index]), *list(np.asarray(stft_imgs)[bp_index])]

        self.total_image, self.labels = [], [] #self.weights = [], [], []

        for c, s in zip(cqt_imgs, stft_imgs):

            self.labels.append(1 if 'bonafide' in str(c) else 0)

            # self.weights.append(0.88 if 'bonafide' in str(c) else 0.12)

            self.total_image.append((c, s))


    def __len__(self):

        return len(self.total_image)


    def __getitem__(self, idx):

        path_cqt, path_stft = self.total_image[idx]

        mat = scipy.io.loadmat(path_cqt, verify_compressed_data_integrity=False)

        m = mat['features']

        img_cqt = Image.fromarray(m).convert("RGB")

        # _, h = img_cqt.size

        img_cqt = img_cqt.resize((self.sliding_window*self.frames, self.resize))

        narray_cqt = np.array(img_cqt)

        #loading stft images
        mat = scipy.io.loadmat(path_stft, verify_compressed_data_integrity=False)

        m = mat['features']

        img_stft = Image.fromarray(m).convert("RGB")

        # _, h = img_stft.size

        img_stft = img_stft.resize((self.sliding_window*self.frames, self.resize))

        narray_stft = np.array(img_stft)

        images_cqt, images_stft = [], []

        for i in range(self.frames):

            im_tmp = narray_cqt[:, i*self.sliding_window: i*self.sliding_window + self.sliding_window, :]

            images_cqt.append(Image.fromarray(im_tmp))

            #adding sftp frame
            im_tmp = narray_stft[:, i*self.sliding_window: i*self.sliding_window + self.sliding_window, :]

            images_stft.append(Image.fromarray(im_tmp))


        if self.transform is not None:

            tensor_image_cqt = self.transform(images_cqt)

            tensor_image_stft = self.transform(images_stft)


        # return (tensor_image_cqt, tensor_image_stft, self.weights[idx]), self.labels[idx]

        return (tensor_image_cqt, tensor_image_stft), self.labels[idx]
