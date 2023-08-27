import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# Custom Dataset class for loading numpy images
class cryo_np_Dataset(Dataset):
    def __init__(self, outliers_file_path, particles_file_path, train=True, train_ratio=0.8, transform=None):
        self.outliers_data = np.load(outliers_file_path)
        self.particles_data = np.load(particles_file_path)
        self.train = train
        self.transform = transform
        self.data = np.concatenate((self.outliers_data, self.particles_data), axis=0)
        self.labels = np.concatenate((np.ones(len(self.outliers_data)), np.zeros(len(self.particles_data))))

        # Split the data into training and validation
        self.total_samples = len(self.data)
        self.train_samples = int(train_ratio * self.total_samples)

        if self.train:
            self.data = self.data[:self.train_samples]
            self.labels = self.labels[:self.train_samples]
        else:
            # Use a portion of the training data for validation
            self.data = self.data[self.train_samples:]
            self.labels = self.labels[self.train_samples:]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


