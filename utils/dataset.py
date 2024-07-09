# -*- coding: utf-8 -*-

# import libraries
import numpy as np
import os
import torch.utils.data as data
from PIL import Image


"""# Load Dataset"""

class Dataset(data.Dataset):

    def __init__(self, root='./', load_set='train', transform=None, include_path=False):
        self.include_path = include_path
        self.root = root#os.path.expanduser(root)
        self.transform = transform
        self.load_set = load_set  # 'train','val','test'
        print(self.load_set)
        print(os.path.abspath(os.path.join(root, 'images-%s.npy'%self.load_set)))
        self.images = np.load(os.path.join(root, 'images-%s.npy'%self.load_set))
        self.points2d = np.load(os.path.join(root, 'points2d-%s.npy'%self.load_set))
        self.points3d = np.load(os.path.join(root, 'points3d-%s.npy'%self.load_set))
        print("------------------")
        print(self.images[0])
        
        #if shuffle:
        #    random.shuffle(data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, points2D, points3D).
        """
        image_path = self.images[index]
        image = Image.open(self.images[index])
        point2d = self.points2d[index]
        point3d = self.points3d[index]

        if self.transform is not None:
            image = self.transform(image)

        if self.include_path:
            return image, point2d, point3d, image_path
        else:
            return image, point2d, point3d

    def __len__(self):
        return len(self.images)
