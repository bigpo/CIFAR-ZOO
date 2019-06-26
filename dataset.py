import os.path as osp
from PIL import Image
import numpy as np
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self,
                 ann_file,
                 root,
                 transform=None,
                 train=True):
        # prefix of images path
        self.img_prefix = root

        # load annotations (and proposals)
        self.img_infos = self.load_annotations(ann_file)

        self.is_train = train
        self.transform = transform

    def __len__(self):
        return len(self.img_infos)

    def load_annotations(self, ann_file):
        with open(ann_file) as fp:
            img_infos = [line.rstrip('\n') for line in fp]
        return [{'filename': line.split(' ')[0],
                 'label': line.split(' ')[1]} for line in img_infos]

    def get_ann_info(self, idx):
        return self.img_infos[idx]['label']

    def _rand_another(self):
        pool = range(len(self.img_infos))
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if not self.is_train:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another()
                continue
            return data

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]

        img = Image.open(osp.join(self.img_prefix, img_info['filename']))
        img = self.transform(img)
        label = self.get_ann_info(idx)
        return img, int(label)

    def prepare_test_img(self, idx):
        """Prepare an image for testing (multi-scale and flipping)"""
        img_info = self.img_infos[idx]
        img = Image.open(osp.join(self.img_prefix, img_info['filename']))
        img = self.transform(img)
        label = self.get_ann_info(idx)
        return img, int(label)
