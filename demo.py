import os
import glob
from matplotlib import pyplot as plt
from dataset import Mask
from dataset import transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from PIL import Image

# You can find the output in the epoch_comparison folder, which locates in the same directory as epochs folders
output_folder = f'outputs/demo_comparison'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

transforms = T.Compose([T.Resize((256,256)),
                        T.ToTensor(),
                        T.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])])

fontsize = 10

f, axarr = plt.subplots(1, 2)

# Test image
img = Image.open('./data/test/001162.png').convert('RGB')
W, H = img.size
cW = W//2
imgA = img.crop((0, 0, cW, H))
imgB = img.crop((cW, 0, W, H))

imgA, imgB = transforms(imgA, imgB)

test_image_to_plot = imgA.numpy().transpose(1, 2, 0)
real_image_to_plot = imgB.numpy().transpose(1, 2, 0)
test_image_to_plot = (test_image_to_plot - test_image_to_plot.min()) / (test_image_to_plot.max() - test_image_to_plot.min())
real_image_to_plot = (real_image_to_plot - real_image_to_plot.min()) / (real_image_to_plot.max() - real_image_to_plot.min())

masked = Image.fromarray((test_image_to_plot * 255).astype('uint8'))
target = Image.fromarray((real_image_to_plot * 255).astype('uint8'))

# Generated image
img_path_1 = f"outputs/cegan/cegan_1000/353.png"
generated = Image.open(img_path_1).convert('RGB')
axarr[1].imshow(generated)
axarr[1].axis('off')
axarr[1].set_title(f'Generated', fontsize=fontsize)

axarr[0].imshow(masked)
axarr[0].axis('off')
axarr[0].set_title('Random', fontsize=fontsize)

plt.savefig(os.path.join(output_folder, f"demo4.png"), bbox_inches='tight', dpi=350)