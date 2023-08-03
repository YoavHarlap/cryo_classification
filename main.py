import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pandas as pd

"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"

def create_points_list_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Print the DataFrame's head to verify the column names and data
    print(df.head())

    # Create an empty list to store the points
    points_list = []

    # Iterate throu6733gh the DataFrame rows and extract X and Y coordinates
    for index, row in df.iterrows():
        x_coordinate = row['X-Coordinate']
        y_coordinate = row['Y-Coordinate']
        point = (x_coordinate, y_coordinate)
        points_list.append(point)

    return points_list


def reflect_points(points_list, mrc_len, axis='x'):
    reflected_points = []
    for point in points_list:
        x, y = point
        if axis == 'x':
            reflected_point = (x, mrc_len - y)
        elif axis == 'y':
            reflected_point = (mrc_len - x, y)
        else:
            raise ValueError("Invalid axis. Use 'x' or 'y'.")

        reflected_points.append(reflected_point)

    return reflected_points


def get_img_from_mrc(mrc_file_path):
    with mrcfile.open(mrc_file_path) as mrc:
        image = mrc.data
    return image


#
# def plot_mrc_file(mrc_file_path):
#     image = get_img_from_mrc(mrc_file_path)
#     # Define the standard deviation (sigma) for the Gaussian filter
#     sigma = 10
#
#     # Apply the Gaussian filter to the image
#     filtered_image = gaussian_filter(image, sigma=sigma)
#
#     # Plot the original and filtered images for comparison
#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')
#
#     plt.subplot(1, 2, 2)
#     plt.imshow(filtered_image, cmap='gray')
#     plt.title(f'Filtered Image (sigma={sigma})')
#     plt.axis('off')
#
#     plt.show()
#
#
# def plot_circles_on_image(image_data, points_list,radius = 250 / 2):
#     # Create a new figure and axis
#     fig, ax = plt.subplots()
#
#     # Display the image
#     ax.imshow(image_data, cmap='gray')
#
#     for point in points_list:
#         # Unpack the point information (center_x, center_y)
#         center_x, center_y = point
#
#         # Create a circle object
#         circle = plt.Circle((center_x, center_y), radius, color='red', fill=False)
#
#         # Add the circle to the plot
#         ax.add_patch(circle)
#
#     # Show the plot
#     plt.show()


def plot_mrc_file_with_circles(mrc_file_path, points_list, radius=250 / 2):
    # Load the image data from the MRC file (replace this with your actual function to read MRC files)
    image = get_img_from_mrc(mrc_file_path)

    # Define the standard deviation (sigma) for the Gaussian filter
    sigma = 10

    # Apply the Gaussian filter to the image
    filtered_image = gaussian_filter(image, sigma=sigma)

    # Create a new figure with a 2x2 grid for the plots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the original image
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    # Plot the filtered image
    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title(f'Filtered Image (sigma={sigma}) and coordinate')
    ax[1].axis('off')

    # Plot circles on the filtered image
    for point in points_list:
        # Unpack the point information (center_x, center_y)
        center_x, center_y = point

        # Create a circle object
        circle = plt.Circle((center_x, center_y), radius, color='red', fill=False)

        # Add the circle to the plot
        ax[1].add_patch(circle)

    # Show the plot
    plt.show()


def crop_images_from_points(image_data, points_list, square_size):
    # Calculate half the square size to get the cropping boundary
    half_size = square_size // 2

    # Create an empty list to store the cropped images
    cropped_images = []

    for point in points_list:
        # Unpack the point information (center_x, center_y)
        center_x, center_y = point

        # Calculate the cropping boundary
        left = max(int(center_x - half_size), 0)
        right = min(int(center_x + half_size), image_data.shape[1])
        top = max(int(center_y - half_size), 0)
        bottom = min(int(center_y + half_size), image_data.shape[0])

        # Crop the region around the point
        cropped_image = image_data[top:bottom, left:right]

        # Append the cropped image to the list
        cropped_images.append(cropped_image)

    # Return the array of cropped images
    return np.array(cropped_images)


if __name__ == "__main__":
    mrc_file_path = "/data/yoavharlap/10028_small/micrographs/061.mrc"
    image = get_img_from_mrc(mrc_file_path)
    # plot_mrc_file(mrc_file_path)
    coordinates_csv_path = "/data/yoavharlap/10028_small/ground_truth/particle_coordinates/061.csv"
    # coordinates_csv_path = "/data/yoavharlap/10028_small/ground_truth/false_positives/061_false_positives.csv"

    points = create_points_list_from_csv(coordinates_csv_path)
    print("coordinates:", points)
    points = reflect_points(points, len(image))
    plot_mrc_file_with_circles(mrc_file_path, points)

    # image_data = np.array(image)
    #
    # points_list = points
    # square_size = 250  # Replace this with the size of the squares you want to crop
    #
    # cropped_images_array = crop_images_from_points(image_data, points_list, square_size)
    # print(cropped_images_array.shape)  # (Number_of_points, square_size, square_size)
    #
    # # Show the first ten cropped images
    # num_images_to_show = 10
    # for i in range(min(num_images_to_show, cropped_images_array.shape[0])):
    #     plt.subplot(2, 5, i + 1)  # Create a 2x5 grid for the images
    #     plt.imshow(cropped_images_array[i], cmap='gray')
    #     plt.title(f"image {i + 1}")
    #     plt.axis('off')
    #
    # plt.tight_layout()  # Adjust the layout for better spacing
    # plt.show()
