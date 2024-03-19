import os
from PIL import Image
import glob


def stitch_and_save_images(source_folder_a, source_folder_b, target_folder):
    # Create the target folder if it does not exist
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    # List all images in folder A
    images_a = sorted(glob.glob(f"{source_folder_a}/*.png"))

    # Iterate through each image in folder A
    for img_path_a in images_a:
        filename = os.path.basename(img_path_a)

        # Construct the corresponding image path in folder B
        img_path_b = os.path.join(source_folder_b, filename)

        # Check if the corresponding image exists in folder B
        if os.path.exists(img_path_b):
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
    source_folder_train = 'mask/outputs'
    source_folder_test = 'mask/inputs'
    target_folder = 'mask/train'

    stitch_and_save_images(source_folder_train,source_folder_test,target_folder)