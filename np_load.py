import numpy as np
import matplotlib.pyplot as plt

# Load all_images_array from the saved .npy file
output_npy_file = "/data/yoavharlap/10028_classification/outliers_images.npy"
# output_npy_file = "/data/yoavharlap/10028_classification/particles_images.npy"
output_npy_file = "/data/yoavharlap/10028_classification/zero_images.npy"
output_npy_file = "/data/yoavharlap/10028_classification/one_images.npy"


all_images_array = np.load(output_npy_file)

# Get the number of images and their dimensions
num_images, height, width = all_images_array.shape
num_images, height, width = all_images_array.shape
print(f"Number of Images: {num_images}\nHeight: {height}\nWidth: {width}")

# Plot 10 random images
num_images_to_show = 10
random_indices = np.random.choice(num_images, num_images_to_show, replace=False)

plt.figure(figsize=(12, 6))
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(all_images_array[idx], cmap='gray')
    plt.title(f"Image {idx}")
    plt.axis('off')

plt.tight_layout()
plt.show()


