import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset class for loading numpy images
class cryo_np_Dataset(Dataset):
    def __init__(self, data,labels, train=True, transform=None):
        self.transform = transform
        self.data = data
        self.labels = labels
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


