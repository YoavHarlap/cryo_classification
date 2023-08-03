import mrcfile
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
def normalize_image(image):
    """
    Normalize the input image to have pixel values between 0 and 1.

    Parameters:
        image (numpy.ndarray): The input image as a NumPy array.

    Returns:
        numpy.ndarray: The normalized image.
    """
    # Convert the data type of the image to float32 to ensure precision during calculations
    image = image.astype('float32')

    # Compute the minimum and maximum pixel values in the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Normalize the image to have pixel values between 0 and 1
    normalized_image = (image - min_val) / (max_val - min_val)

    return 255*normalized_image

mrc_file_path = "/data/yoavharlap/10028_small/micrographs/061.mrc"
with mrcfile.open(mrc_file_path) as mrc:
    # Access the 3D volumetric data as a numpy array
    image_data = mrc.data
# image_data = normalize_image(image_data)
print(image_data.min())
plt.imshow(image_data, cmap='gray')

# Display the image



plt.show()
