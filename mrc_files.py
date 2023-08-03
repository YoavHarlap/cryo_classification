
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
import glob, os
import numpy as np
from PIL import Image, ImageDraw
def draw_circle_on_np_array(image_np, center_x, center_y, radius, color=(255, 0, 0), thickness=10):
    # Create a PIL Image from the NumPy array
    print(image_np[0,0])
    # img = Image.fromarray(image_np)
    img = Image.fromarray(np.uint8(image_np))

    pixel_access = img.load()

    # Access a pixel value
    pixel_value = pixel_access[0,0]
    print(pixel_value)
    img.show()
    # Create a drawing context
    draw = ImageDraw.Draw(img)

    # Calculate the bounding box of the circle
    x1 = center_x - radius
    y1 = center_y - radius
    x2 = center_x + radius
    y2 = center_y + radius

    # Convert RGB color tuple to integer
    color = int((color[0]<<16) + (color[1]<<8) + color[2])

    # Draw the circle
    draw.ellipse([x1, y1, x2, y2], outline=color, width=thickness)
    img.show()
    # Convert back to NumPy array
    modified_image_np = np.array(img)
    print(modified_image_np[0, 0])

    return modified_image_np


images = []

mrc_path = "/data/yoavharlap/10028_small/micrographs/"
# mrc_path = "/data/yoelsh/datasets/10028/data/Micrographs/Micrographs_part1/"
# mrc_path = "/data/yoavharlap/eman_particles/good"
os.chdir(mrc_path)
i=0
for file in glob.glob("*.mrc"):
    print(file)
    mrc = mrcfile.open(file, 'r')
    images.append(np.array(mrc.data))
    mrc.close()
    plt.imshow(images[i], cmap='gray')
    plt.show()
    # Example usage:
    # Assuming you have the image as a NumPy array named "image_np"
    center_x = 3091

    center_y = 2726
    radius = 250/2
    modified_image_np = draw_circle_on_np_array(images[i], center_x, center_y, radius)
    plt.imshow(modified_image_np, cmap='gray')
    plt.show()
    i=i+1

    # Now you can use the modified_image_np array for further processing or displaying.

# for i in range(len(images)):
#     plt.imshow(images[i],cmap='gray')
#     plt.show()