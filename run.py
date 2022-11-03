# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.utils.data
import torchvision
from model import DenoiseDiffusion
import argparse
from tqdm import tqdm
from utils import dir_check


def get_params():
    parser = argparse.ArgumentParser(description='Entry point of the code')

    # dataset information
    parser.add_argument('--image_channels', type=int, default=1, help='Number of channels in the image. $3$ for RGB.')
    parser.add_argument('--image_size', type=int, default=32, help='Image size.')
    parser.add_argument('--n_channels', type=int, default=64, help='Number of channels in the initial feature map.')
    # The list of channel numbers at each resolution., The number of channels is `channel_multipliers[i] * n_channels`
    # The number of channels is `channel_multipliers[i] * n_channels`
    parser.add_argument('--channel_multipliers', type=list, default=[1, 2, 2, 4], help='The list of channel numbers at each resolution.')
    parser.add_argument('--is_attention', type=list, default=[False, False, False, True], help='The list of booleans that indicate whether to use attention at each resolution.')

    # model parameters
    parser.add_argument('--T', type=int, default=1000, help='Number of time steps')

    # Training parameters
    parser.add_argument('--is_test', type=bool, default=True, help='Whether test the code')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate.')
    parser.add_argument('--epoch', type=int, default=5, help='Number of training epochs')

    # sample parameters
    parser.add_argument('--n_samples', type=int, default=16, help='Number of samples to generate')
    parser.add_argument('--file_samples', type=str, default='./samples/', help='File of saving the samples')

    args, _ = parser.parse_known_args()
    return args


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """
    def __init__(self, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])
        super().__init__('./data/', train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


def main(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    params['device']  = device
    diffusion = DenoiseDiffusion(params)

    # Load training dataset
    train_dataset = MNISTDataset(params['image_size'])
    train_loader = torch.utils.data.DataLoader(train_dataset, params['batch_size'], shuffle=True, pin_memory=True)

    # Create optimizer
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=params['learning_rate'])

    # Train and sample the data
    for epoch in range(params['epoch']):
        postfix = {"epoch": epoch,  "current_loss": 0.0}

        # train the diffusion model
        with tqdm(train_loader, total=len(train_loader), postfix=postfix) as bar:
            for i, batch in enumerate(bar):
                if i > 700 and params['is_test']: break
                data = batch.to(device)
                optimizer.zero_grad()
                loss = diffusion.loss(data)
                loss.backward()
                optimizer.step()
                postfix["current_loss"] = loss.item()
                bar.set_postfix(**postfix)

        # sample images
        print('Begin to generate samples')
        with torch.no_grad():
            x = torch.randn([params['n_samples'], params['image_channels'],  params['image_size'], params['image_size']], device=device)
            # Remove noise for $T$ steps
            T = params['T']
            for t in range(T, 0, -1): # in paper, t should start from T, and end at 1
                t = t - 1 # in code, t is index, so t should minus 1
                x = diffusion.p_sample(x, x.new_full((params['n_samples'], ),t, dtype=torch.long))

            img_path = f'./samples/{epoch}.png'
            data2img(x.cpu().numpy(), img_path)


def data2img(x, fout):
    # draw the samples
    # x: (16, 1, 32, 32)
    dir_check(fout)
    import matplotlib.pyplot as plt
    imgs = x.clip(0, 1)
    fig, ax = plt.subplots(4,4)
    for i in range(4):
        for j in range(4):
            idx = 4 * i + j
            img = imgs[idx][0] # only take the first channel to draw the image
            ax[i][j].imshow(img)
    plt.savefig(fout)



if __name__ == '__main__':
    params = vars(get_params())
    print(params)
    main(params)




