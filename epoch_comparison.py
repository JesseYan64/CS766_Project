import os
import glob
from dataset import Mask
from dataset import transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from PIL import Image

# Set the path to the output folders, it has to be the prefix of the epochs folders
prefix = 'outputs/cegan/cegan_'

# test if masked, test_r if random covering.
path_to_test = './data/test'

# Set the epochs to compare
epochs = [10, 50]

# In the end the output would be masked, generated images and the real image, stitched together

folders = [f'{prefix}{i}' for i in epochs]
image_num = len(glob.glob(f"{prefix}10/*.png"))

# You can find the output in the epoch_comparison folder, which locates in the same directory as epochs folders
output_folder = f'{prefix}epoch_comparison'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print(image_num)

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

dataset = Mask(path=path_to_test, transform=transforms, mode='test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

i = 0
for m, real in dataloader:
    test_image_to_plot = m[0].numpy().transpose(1, 2, 0)
    real_image_to_plot = real[0].numpy().transpose(1, 2, 0)
    test_image_to_plot = (test_image_to_plot - test_image_to_plot.min()) / (test_image_to_plot.max() - test_image_to_plot.min())
    real_image_to_plot = (real_image_to_plot - real_image_to_plot.min()) / (real_image_to_plot.max() - real_image_to_plot.min())

    masked = Image.fromarray((test_image_to_plot * 255).astype('uint8'))
    target = Image.fromarray((real_image_to_plot * 255).astype('uint8'))

    stitched_image = Image.new('RGB', (256 * (len(folders) + 2), 256))

    for j, folder in enumerate(folders):
        img_path = f"{folder}/{i}.png"
        generated = Image.open(img_path).convert('RGB')
        stitched_image.paste(generated, (256 * (j + 1), 0))

    stitched_image.paste(masked, (0, 0))
    stitched_image.paste(target, (256 * (len(folders) + 1), 0))
    stitched_image.save(os.path.join(output_folder, f"{i}.png"))
    i += 1