import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
from torchvision import datasets
from torchvision import transforms

import matplotlib.pyplot as plt


class Config(object):
    train_path ='./' # there is a ./MNIST data folder
    img_size = 784
    h_dim = 400
    z_dim = 20
    download = True
    batch_size = 100
    shuffle = True
    num_workers = 2
    lr = 1e-3
    use_gpu = True
    epoch = 20


class VAE(nn.Module):
    def __init__(self, cfg):
        super(VAE, self).__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.img_size, cfg.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(cfg.h_dim, cfg.z_dim * 2))
            # no Sigmoid here, it should produce log_var
            # whose range is [-infty, +infty]

        self.decoder = nn.Sequential(
            nn.Linear(cfg.z_dim, cfg.h_dim),
            nn.ReLU(),
            nn.Linear(cfg.h_dim, cfg.img_size),
            nn.Sigmoid())

    def reparameterize(self, mean, log_var):
        samples = Variable(torch.randn(mean.size(0), mean.size(1)))
        if self.cfg.use_gpu:
            samples = samples.cuda()
        z = mean + samples * torch.exp(log_var / 2)
        return z

    def forward(self, x):
        h = self.encoder(x)
        mean, log_var = torch.chunk(h, 2, dim=1)
        z = self.reparameterize(mean, log_var)
        out = self.decoder(z)
        return out, mean, log_var


def train(**kwargs):
    cfg = Config()
    for k, v in kwargs.items():
        setattr(cfg, k, v)

    dataset = datasets.MNIST(root=cfg.train_path,
                             train=True,
                             transform=transforms.ToTensor(),
                             download=cfg.download)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=cfg.batch_size,
                                              shuffle=cfg.shuffle,
                                              num_workers=cfg.num_workers)
    vae = VAE(cfg)
    if cfg.use_gpu:
        print('use GPU!')
        vae = vae.cuda()

    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.lr)
    data_iter = iter(data_loader)
    fixed_x, _ = next(data_iter)
    torchvision.utils.save_image(fixed_x.cpu(), './data/real_images.png')
    fixed_x = Variable(fixed_x.view(fixed_x.size(0), -1))
    if cfg.use_gpu:
        fixed_x = fixed_x.cuda()

    plt.ion()
    for epoch in range(cfg.epoch):
        print('epoch', epoch)
        for i, (images, _) in enumerate(data_loader):

            images = Variable(images.view(images.size(0), -1))
            if cfg.use_gpu:
                images = images.cuda()
            out, mean, log_var = vae(images)
            reconst_loss = F.binary_cross_entropy(out, images, reduction='sum')
            kl_divergence = torch.sum(0.5 * (mean ** 2 + torch.exp(log_var) - log_var - 1))

            total_loss = reconst_loss + kl_divergence
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if i % 100 == 0:
                plt.cla()

                plt.subplot(1, 2, 1)
                plt.imshow(images.data[0].view(28, 28).cpu().numpy(), cmap="gray")
                plt.subplot(1, 2, 2)
                plt.imshow(out.data[0].view(28, 28).cpu().numpy(), cmap="gray")

                plt.draw()
                plt.pause(0.1)

        reconst_images, _, _ = vae(fixed_x)
        reconst_images = reconst_images.view(reconst_images.size(0), 1, 28, 28)
        torchvision.utils.save_image(reconst_images.data.cpu(),
                                     './data/reconst_images_%d.png' % (epoch + 1))
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    import fire
    fire.Fire()
