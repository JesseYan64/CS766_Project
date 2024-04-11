import os
import glob

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from dataset import Mask
from dataset import transforms as T
from gan.generator import UnetGenerator

def test(output_folder):
    generator = UnetGenerator()
    generator.load_state_dict(torch.load("./runs/generator.pt"))
    generator.eval()

    transforms = T.Compose([T.Resize((256,256)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])])
    
    path_to_test = './images/test'

    dataset = Mask(path=path_to_test, transform=transforms, mode='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    count = 0
    for m, real in dataloader:
        # Convert the tensor to a numpy array and adjust the dimension order for matplotlib
        # The 'm' tensor is expected to be in the format [B, C, H, W], where
        # B = Batch size, C = Channels, H = Height, W = Width
        test_image_to_plot = m[0].numpy().transpose(1, 2, 0)
        real_image_to_plot = real[0].numpy().transpose(1, 2, 0)
        # Normalize the image data to [0, 1] for correct visualization
        test_image_to_plot = (test_image_to_plot - test_image_to_plot.min()) / (test_image_to_plot.max() - test_image_to_plot.min())
        real_image_to_plot = (real_image_to_plot - real_image_to_plot.min()) / (real_image_to_plot.max() - real_image_to_plot.min())

        test_image = Image.fromarray((test_image_to_plot * 255).astype('uint8'))
        real_image = Image.fromarray((real_image_to_plot * 255).astype('uint8'))

        test_width = test_image.width
        stitched_image = Image.new('RGB', (test_width + test_width + test_width, test_image.height))
        stitched_image.paste(test_image, (0, 0))

        # Prepare your test image
        transform = Compose([
            Resize((256, 256)),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        test_image = transform(test_image).unsqueeze(0)  # Add a batch dimension

        # Generate the output
        with torch.no_grad():  # No need to track gradients for the test image
            generated_image = generator(test_image).squeeze(0)  # Remove the batch dimension

        # Convert the output tensor to an image
        generated_image = generated_image.mul(0.5).add(0.5)  # Denormalize
        generated_image = generated_image.clamp(0, 1)  # Clamp the values to ensure they're between 0 and 1
        generated_image = generated_image.permute(1, 2, 0).numpy()  # Change the tensor shape from CxHxW to HxWxC

        # Display the generated image

        # plt.imshow(generated_image.numpy())
        stitched_image.paste(Image.fromarray((generated_image * 255).astype('uint8')), (test_width, 0))
        stitched_image.paste(real_image, (test_width * 2, 0))
        stitched_image.save(os.path.join(output_folder, f"{count}.png"))
        count += 1


def clear_outputs(output_folder):
    for f in glob.glob(f"{output_folder}/*.png"):
        os.remove(f)


if __name__ == "__main__":
    output_folder="./outputs"
    clear_outputs(output_folder)
    test(output_folder)