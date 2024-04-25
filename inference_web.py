import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image

from dataset import transforms as T
from gan.generator import UnetGenerator
from gan.cegan import Generator

def test(output, path_to_image=None, cegan=False, model_path=None):
    if cegan:
        generator = Generator()
    else:
        generator = UnetGenerator()
    generator.load_state_dict(torch.load(model_path))
    generator.eval()

    transforms = T.Compose([T.Resize((256,256)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])])

    img = Image.open(path_to_image).convert('RGB')
    m = transforms(img)
    test_image_to_plot = m.numpy().transpose(1, 2, 0)
    # Normalize the image data to [0, 1] for correct visualization
    test_image_to_plot = (test_image_to_plot - test_image_to_plot.min()) / (test_image_to_plot.max() - test_image_to_plot.min())

    test_image = Image.fromarray((test_image_to_plot * 255).astype('uint8'))

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
    Image.fromarray((generated_image * 255).astype('uint8')).save(output)


if __name__ == "__main__":
    # You have four options here, according to what you have in runs: 'pix2pix', 'cegan', 'pix2pix_r', 'cegan_r'. Note that '_r' means random covering.
    model = 'cegan'

    # test if masked, test_r if random covering.
    output = 'outputs/test.png'
    path_to_image='data/with_mask/000001.png'

    epoch = 1000

    # Specify the epochs you want to use for generating the images
    epochs = [10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

    test(output, path_to_image, cegan=True, model_path=f'./runs/{model}/generator_{epoch}.pt')