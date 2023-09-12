import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import mrcfile

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import pandas as pd
import pandas as pd


def create_points_list_from_csv(file_path):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Print the DataFrame's head to verify the column names and data
    # print(df.head())

    # Create an empty list to store the points
    points_list = []

    # Iterate throu67343gh the DataFrame rows and extract X and Y coordinates
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
# def plot_circles_on_image(image_data, points_list,radius = 256 / 2):
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

def plot_mrc_file_with_circles(mrc_file_path, points_list, radius=256 / 2):
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
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False)

        # Add the circle to the plot
        ax[1].add_patch(circle)

    # Show the plot
    plt.show()


def plot_mrc_file_with_circles_2(mrc_file_path, particles_points_list, outliers_points_list, radius=256 / 2):
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
    ax[1].set_title(f'Filtered Image (sigma={sigma}) and coordinate(red for outliers)')
    ax[1].axis('off')

    # Plot circles on the filtered image
    for point in outliers_points_list:
        # Unpack the point information (center_x, center_y)
        center_x, center_y = point

        # Create a circle object
        circle = plt.Circle((center_x, center_y), radius, color='red', fill=False)

        # Add the circle to the plot
        ax[1].add_patch(circle)

    for point in particles_points_list:
        # Unpack the point information (center_x, center_y)
        center_x, center_y = point

        # Create a circle object
        circle = plt.Circle((center_x, center_y), radius, color='blue', fill=False)

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


def process_files(mrc_file_path, coordinates_csv_path, square_size):
    image = get_img_from_mrc(mrc_file_path)
    points = create_points_list_from_csv(coordinates_csv_path)
    points = reflect_points(points, len(image))
    image_data = np.array(image)
    cropped_images_array = crop_images_from_points(image_data, points, square_size)
    return cropped_images_array


import os
import numpy as np
import matplotlib.pyplot as plt


def process_files(mrc_file_path, coordinates_csv_path, square_size):
    image = get_img_from_mrc(mrc_file_path)
    points = create_points_list_from_csv(coordinates_csv_path)
    points = reflect_points(points, len(image))
    image_data = np.array(image)
    cropped_images_array = crop_images_from_points(image_data, points, square_size)
    return cropped_images_array


def cropping():
    input_folder = "/data/yoavharlap/10028/micrographs/"
    coordinates_folder = "/data/yoavharlap/10028/ground_truth/false_positives/"
    output_npy_file = "/data/yoavharlap/10028_classification/outliers_images.npy"

    square_size = 256
    all_images = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".mrc"):
            mrc_file_path = os.path.join(input_folder, filename)
            basename = os.path.splitext(filename)[0]
            coordinates_csv_path = os.path.join(coordinates_folder, f"{basename}_false_positives.csv")
            # coordinates_csv_path = os.path.join(coordinates_folder, f"{basename}.csv")

            if os.path.exists(coordinates_csv_path):
                cropped_images_array = process_files(mrc_file_path, coordinates_csv_path, square_size)
                num_processed_images = cropped_images_array.shape[0]
                all_images.append(cropped_images_array)
                print(f"Processed {filename} and added {num_processed_images} images to the collection")
            else:
                print(f"CSV file not found for {filename}")

    if all_images:
        all_images_array = np.concatenate(all_images, axis=0)  # Combine all images into one array

        np.save(output_npy_file, all_images_array)
        print(f"Saved all processed images to {output_npy_file}")
    else:
        print("No images were processed.")


def circles_1_class():
    input_folder = "/data/yoavharlap/10028/micrographs/"

    # for outliers
    coordinates_folder = "/data/yoavharlap/10028/ground_truth/false_positives/"
    # for particles
    coordinates_folder = "/data/yoavharlap/10028/ground_truth/particle_coordinates"

    square_size = 256
    all_images = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".mrc"):
            mrc_file_path = os.path.join(input_folder, filename)
            basename = os.path.splitext(filename)[0]

            # for outliers
            # coordinates_csv_path = os.path.join(coordinates_folder, f"{basename}_false_positives.csv")
            # for particles
            coordinates_csv_path = os.path.join(coordinates_folder, f"{basename}.csv")

            if os.path.exists(coordinates_csv_path):
                points = create_points_list_from_csv(coordinates_csv_path)
                points_list = reflect_points(points, 4096)
                plot_mrc_file_with_circles(mrc_file_path, points_list, radius=256 / 2)

            else:
                print(f"CSV file: {coordinates_csv_path} not found for: {filename}")


def circles_2_classes():
    input_folder = "/data/yoavharlap/10028/micrographs/"

    # for outliers
    outliers_coordinates_folder = "/data/yoavharlap/10028/ground_truth/false_positives/"
    # for particles
    particles_coordinates_folder = "/data/yoavharlap/10028/ground_truth/particle_coordinates"

    square_size = 256
    all_images = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".mrc"):
            mrc_file_path = os.path.join(input_folder, filename)
            basename = os.path.splitext(filename)[0]

            # for outliers
            outliers_coordinates_csv_path = os.path.join(outliers_coordinates_folder, f"{basename}_false_positives.csv")
            # for particles
            particles_coordinates_csv_path = os.path.join(particles_coordinates_folder, f"{basename}.csv")

            if os.path.exists(particles_coordinates_csv_path):
                points = create_points_list_from_csv(particles_coordinates_csv_path)
                particles_points_list = reflect_points(points, 4096)
            else:
                print(f"CSV file: {particles_coordinates_csv_path} not found for: {filename}")

            if os.path.exists(outliers_coordinates_csv_path):
                points = create_points_list_from_csv(outliers_coordinates_csv_path)
                outliers_points_list = reflect_points(points, 4096)
            else:
                print(f"CSV file: {outliers_coordinates_csv_path} not found for: {filename}")

            plot_mrc_file_with_circles_2(mrc_file_path, particles_points_list, outliers_points_list, radius=256 / 2)


if __name__ == "__main__":
    # cropping()
    # circles_1_class()
    circles_2_classes()
