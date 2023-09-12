import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np

# Define the directory path
save_dir = "/data/yoavharlap/10028_classification/"

# Create the directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Define a data transform to convert the images to NumPy arrays
transform = transforms.Compose([transforms.ToTensor()])

# Download the MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)

# Filter the dataset for zeros and ones
zero_indices = torch.where((train_dataset.targets == 0))[0]
one_indices = torch.where((train_dataset.targets == 1))[0]

zero_images = []
one_images = []

# Convert and store zero images
for idx in zero_indices:
    zero_images.append(train_dataset[idx][0].numpy().squeeze())  # Remove the single-dimensional entry

# Convert and store one images
for idx in one_indices:
    one_images.append(train_dataset[idx][0].numpy().squeeze())  # Remove the single-dimensional entry

zero_images = np.array(zero_images)
one_images = np.array(one_images)

# Save the zero and one images
zero_file_path = os.path.join(save_dir, "zero_images.npy")
one_file_path = os.path.join(save_dir, "one_images.npy")

np.save(zero_file_path, zero_images)
np.save(one_file_path, one_images)

print("Zero images saved as zero_images.npy.")
print("One images saved as one_images.npy.")
