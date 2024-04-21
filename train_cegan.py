import torch
from torch import nn
from torch.utils.data import DataLoader
import time
import os
import argparse
from progress.bar import IncrementalBar

from dataset import transforms as T
from gan.cegan import Generator, Discriminator
from gan.criterion import GeneratorLoss, DiscriminatorLoss
from gan.utils import Logger, initialize_weights

from dataset import Mask

def train(args):
    device = ('cuda:0' if torch.cuda.is_available() else 'cpu')

    transforms = T.Compose([T.Resize((256,256)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])])
    # models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    # optimizers
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    # loss functions
    g_criterion = GeneratorLoss(alpha=100)
    d_criterion = DiscriminatorLoss()
    # dataset
    dataset = Mask(path=args.dataset_path, transform=transforms, mode='train')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    sub_dir = args.output

    logger = Logger(exp_name=f'./runs/{sub_dir}', filename=sub_dir)

    total_time = 0
    epoch_start = 0
    # Whether has a checkpoint
    if os.path.exists(f'./runs/{sub_dir}/checkpoint.pt'):
        checkpoint = torch.load(f'./runs/{sub_dir}/checkpoint.pt')
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        epoch_start = checkpoint['epoch']
        total_time = checkpoint['total_time']
        g_loss = checkpoint['g_loss']
        d_loss = checkpoint['d_loss']
        logger.add_scalar('generator_loss', g_loss, epoch_start)
        logger.add_scalar('discriminator_loss', d_loss, epoch_start)
        print(f"Resuming training from epoch {epoch_start}")
    else:
        logger.save_weights(generator.state_dict(), 'generator_0')
        logger.save_weights(discriminator.state_dict(), 'discriminator_0')

    print('Training started')

    for epoch in range(epoch_start + 1, args.epochs + 1):
        ge_loss=0.
        de_loss=0.
        start = time.time()
        bar = IncrementalBar(f'[Epoch {epoch}/{args.epochs}]', max=len(dataloader))
        for x, real in dataloader:
            x = x.to(device)
            real = real.to(device)

            # Generator`s loss
            fake = generator(x)
            fake_pred = discriminator(fake, x)
            g_loss = g_criterion(fake, real, fake_pred)

            # Discriminator`s loss
            fake = generator(x).detach()
            fake_pred = discriminator(fake, x)
            real_pred = discriminator(real, x)
            d_loss = d_criterion(fake_pred, real_pred)

            # Generator`s params update
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # Discriminator`s params update
            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()
            # add batch losses
            ge_loss += g_loss.item()
            de_loss += d_loss.item()
            bar.next()
        bar.finish()
        # obttain per epoch losses
        g_loss = ge_loss/len(dataloader)
        d_loss = de_loss/len(dataloader)
        # count timeframe
        end = time.time()
        tm = (end - start)
        total_time += tm
        logger.save_weights(generator.state_dict(), f'generator')
        logger.save_weights(discriminator.state_dict(), f'discriminator')
        if epoch % 10 == 0:
            logger.save_weights(generator.state_dict(), f'generator_{epoch}')
            logger.save_weights(discriminator.state_dict(), f'discriminator_{epoch}')
        torch.save({'epoch': epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_loss': g_loss,
                    'd_loss': d_loss,
                    'total_time': total_time
                    }, f'./runs/{sub_dir}/checkpoint.pt')
        logger.add_scalar('generator_loss', g_loss, epoch)
        logger.add_scalar('discriminator_loss', d_loss, epoch)
        print("[Epoch %d/%d] [G loss: %.3f] [D loss: %.3f] ETA: %.3fs" % (epoch, args.epochs, g_loss, d_loss, tm))
    print('Training finished')
    logger.add_scalar('total_time', total_time, args.epochs)
    print(f"Total time: {total_time}")
    logger.close()

if __name__ == "__main__":
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(prog ='top', description='Train Pix2Pix')
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs")
    parser.add_argument("--dataset", type=str, default="mask", help="Name of the train dataset")
    parser.add_argument("--dataset-path", type=str, default="./data/train", help="Path to the train dataset")
    parser.add_argument("--output", type=str, default="cegan", help="Output")
    parser.add_argument("--batch-size", type=int, default=64, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="Adams learning rate")
    args = parser.parse_args()

    # Now you can access the arguments just like you would with args parsed from command-line
    print(f"Train {args.epochs} epochs")
    print(f"Using {args.dataset} dataset")
    print(f"Dataset path is {args.dataset_path}")
    print("Batch size is", args.batch_size)
    print("Learning rate is", args.lr)

    train(args)