import os
from PIL import Image
import glob

from sklearn.model_selection import train_test_split

def stitch_and_save_images(source_folder_a, target_folder_b, train_folder, test_folder):

    if not os.path.exists(train_folder):
        os.makedirs(train_folder)

    if not os.path.exists(test_folder):
        os.makedirs(test_folder)

    # List all images in folder A
    images_a = sorted(glob.glob(f"{source_folder_a}/*.png"))
    images_b = sorted(glob.glob(f"{target_folder_b}/*.png"))

    # Split the images into training and testing sets
    images_a_train, images_a_test, images_b_train, images_b_test = train_test_split(images_a, images_b, test_size=0.2, random_state=42)

    # Stitch and save the training images
    stitch(images_a_train, images_b_train, train_folder)
    stitch(images_a_test, images_b_test, test_folder)


def stitch(source_arr, target_arr, target_folder):

    for i in range(len(source_arr)):
        img_path_a = source_arr[i]
        img_path_b = target_arr[i]
        filename = os.path.basename(img_path_a)

        # Open both images
        img_a = Image.open(img_path_a).convert('RGB')
        img_b = Image.open(img_path_b).convert('RGB')

        # Stitch images side-by-side
        stitched_image = Image.new('RGB', (img_a.width + img_b.width, img_a.height))
        stitched_image.paste(img_a, (0, 0))
        stitched_image.paste(img_b, (img_a.width, 0))

        # Save the stitched image to the target folder
        stitched_image.save(os.path.join(target_folder, filename))


if __name__ == "__main__":
    source_folder_X = 'images/with_mask'
    source_folder_y = 'images/no_mask'
    train_folder = 'images/train'
    test_folder = 'images/test'

    stitch_and_save_images(source_folder_X, source_folder_y, train_folder, test_folder)