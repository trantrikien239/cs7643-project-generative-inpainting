import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    def __init__(self, hidden_size=64):
        super(Generator, self).__init__()

        self.hidden_size=hidden_size

        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.enc2 = self.contract(hidden_size, hidden_size*2, 4, 2, 1, False)
        self.enc3 = self.contract(hidden_size*2, hidden_size*4, 4, 2, 1, False)
        self.enc4 = self.contract(hidden_size*4, hidden_size*8, 4, 2, 1, False)
        self.enc5 = self.contract(hidden_size*8, hidden_size*8, 4, 2, 1, False)
        self.enc6 = self.contract(hidden_size*8, hidden_size*8, 4, 2, 1, False)

        self.dec6 = self.expand(hidden_size*8, hidden_size*8, 4, 2, 1, False)
        self.dec5 = self.expand(hidden_size*8, hidden_size*8, 4, 2, 1, False)
        self.dec4 = self.expand(hidden_size*8, hidden_size*4, 4, 2, 1, False)
        self.dec3 = self.expand(hidden_size*4, hidden_size*2, 4, 2, 1, False)
        self.dec2 = self.expand(hidden_size*2, hidden_size, 4, 2, 1, False)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=hidden_size, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def contract(self, ic, oc, ks, s, p, b):
      return nn.Sequential(
                nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=s, padding=p, bias=b),
                nn.BatchNorm2d(oc),
                nn.LeakyReLU(0.2, inplace=True)
          )
      
    def expand(self, ic, oc, ks, s, p, b):
      return nn.Sequential(
                nn.ConvTranspose2d(in_channels=ic, out_channels=oc, kernel_size=ks, stride=s, padding=p, bias=b),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True)
          )

    def forward(self, input):
        conv1 = self.enc1(input)
        conv2 = self.enc2(conv1)
        conv3 = self.enc3(conv2)
        conv4 = self.enc4(conv3)
        conv5 = self.enc5(conv4)
        conv6 = self.enc6(conv5)

        upconv6 = self.dec6(conv6)
        upconv5 = self.dec5(upconv6 + conv5)
        upconv4 = self.dec4(upconv5 + conv4)
        upconv3 = self.dec3(upconv4 + conv3)
        upconv2 = self.dec2(upconv3 + conv2)
        upconv1 = self.dec1(upconv2 + conv1)

        return upconv1

    
class Discriminator(nn.Module):
    def __init__(self, hidden_size=64):
        super(Discriminator, self).__init__()

        self.hidden_size = hidden_size

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size*2, out_channels=hidden_size*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size*4, out_channels=hidden_size*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size*8, out_channels=hidden_size*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size*8, out_channels=hidden_size*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_size*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=hidden_size*8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.conv(input)


class DCGAN:
    def __init__(self, hidden_size_g=64, hidden_size_d=64, device=None):

        self.generator = Generator(hidden_size=hidden_size_g).to(device)
        self.discriminator = Discriminator(hidden_size=hidden_size_d).to(device)

        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.g_losses = []
        self.d_losses = []
        self.d_real_losses = []
        self.d_fake_losses = []
        self.rec_losses = []
        

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def train(self, dataloader_train, batch_size, num_images, device, epochs=1, lr=0.0002, beta1=0.5):

        real_label = 1
        fake_label = 0

        num_batches = int(np.ceil(num_images / batch_size))

        adversarial_loss = nn.BCELoss()
        reconstruction_loss = nn.L1Loss()

        optimizerG = optim.Adam(self.generator.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerD = optim.Adam(self.discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

        for e in range(epochs):

            for i, sample in enumerate(dataloader_train):

                images = torch.permute(sample['images'], (0, 3, 1, 2)).type(torch.FloatTensor).to(device)
                masked_images = torch.permute(sample['masked_images'], (0, 3, 1, 2)).type(torch.FloatTensor).to(device)

                self.discriminator.zero_grad()

                # Real batch
                labels = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

                out_d_real = self.discriminator(images).view(-1)

                loss_d_real = adversarial_loss(out_d_real, labels)
                loss_d_real.backward()

                # Fake batch
                out_g = self.generator(masked_images)
                labels.fill_(fake_label)

                out_d_fake = self.discriminator(out_g.detach()).view(-1)

                loss_d_fake = adversarial_loss(out_d_fake, labels)
                loss_d_fake.backward()

                loss_d = loss_d_real + loss_d_fake

                optimizerD.step()

                # Generator

                self.generator.zero_grad()

                labels.fill_(real_label)

                out_d = self.discriminator(out_g).view(-1)

                adv_loss = adversarial_loss(out_d, labels)
                rec_loss = reconstruction_loss(out_g, images)

                loss_g = adv_loss + rec_loss
                loss_g.backward()

                optimizerG.step()

                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D_Real: %.4f\tLoss_D_Fake: %.4f\tLoss_G: %.4f\tLoss_G_rec: %.4f' 
                    % (e+1, epochs, i, num_batches, loss_d_real.item(), 
                    loss_d_fake.item(), loss_g.item(), rec_loss.item()))

                self.d_losses.append(loss_d.item())
                self.g_losses.append(loss_g.item())
                self.d_real_losses.append(loss_d_real.item())
                self.d_fake_losses.append(loss_d_fake.item())
                self.rec_losses.append(rec_loss.item())

                if (i * batch_size) > num_images:
                  break
























