import os
import glob

from PIL import Image


folders = ['outputs/Pix2Pix_10', 'outputs/Pix2Pix_50', 'outputs/Pix2Pix_100', 'outputs/Pix2Pix_200', 'outputs/Pix2Pix_300', 'outputs/Pix2Pix_400', 'outputs/Pix2Pix_500']
image_num = len(glob.glob(f"outputs/Pix2Pix_10/*.png"))

output_folder = 'outputs/epoch_comparison'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(image_num)

for i in range(0, image_num):
    stitched_image = Image.new('RGB', (256 * (len(folders) + 2), 256))
    for j, folder in enumerate(folders):
        img_path = f"{folder}/{i}.png"
        img = Image.open(img_path).convert('RGB')
        W, H = img.size
        cW = W//3
        masked = img.crop((0, 0, cW, H))
        generated = img.crop((cW, 0, cW * 2, H))
        target = img.crop((cW * 2, 0, W, H))
        stitched_image.paste(generated, (256 * (j + 1), 0))
    stitched_image.paste(masked, (0, 0))
    stitched_image.paste(target, (256 * (len(folders) + 1), 0))
    stitched_image.save(os.path.join(output_folder, f"{i}.png"))