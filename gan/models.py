import torch
from gan.spectral_normalization import SpectralNorm
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = SpectralNorm(nn.Conv2d(3, 128, 4, stride=2, padding=1))
        #self.conv1 = nn.Conv2d(3, 128, 4, stride=2, padding=1)
        self.conv2 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=1))
        #self.conv2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(256)
        self.conv3 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=1))
        #self.conv3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(512)
        self.conv4 = SpectralNorm(nn.Conv2d(512, 1024, 4, stride=2, padding=1))
        #self.conv4 = nn.Conv2d(512, 1024, 4, stride=2, padding=1)
        self.conv4_bn = nn.BatchNorm2d(1024)
        self.conv5 = SpectralNorm(nn.Conv2d(1024, 1, 4, stride=1, padding=0))
        #self.conv5 = nn.Conv2d(1024, 1, 4, stride=1, padding=0)
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################


        ##########       END      ##########

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x),0.2).cuda()
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)),0.2).cuda()
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)),0.2).cuda()
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)),0.2).cuda()
        x = F.leaky_relu(self.conv5(x),0.2).cuda()
        x = x.view(-1,1).cuda()
        ####################################
        #          YOUR CODE HERE          #
        ####################################


        ##########       END      ##########

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.conv1 = nn.ConvTranspose2d(self.noise_dim, 1024, 4, stride=1,padding=0)
        self.conv1_bn = nn.BatchNorm2d(1024)
        self.conv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2,padding=1)
        self.conv2_bn = nn.BatchNorm2d(512)
        self.conv3 = nn.ConvTranspose2d(512, 256, 4, stride=2,padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv4 = nn.ConvTranspose2d(256, 128, 4, stride=2,padding=1)
        self.conv4_bn = nn.BatchNorm2d(128)
        self.conv5 = nn.ConvTranspose2d(128, 3, 4, stride=2,padding=1)
        ####################################
        #          YOUR CODE HERE          #
        ####################################


        ##########       END      ##########

    def forward(self, x):
        x=x.view(-1,self.noise_dim,1,1).cuda()
        x = F.relu(self.conv1_bn(self.conv1(x))).cuda()
        x = F.relu(self.conv2_bn(self.conv2(x))).cuda()
        x = F.relu(self.conv3_bn(self.conv3(x))).cuda()
        x = F.relu(self.conv4_bn(self.conv4(x))).cuda()
        x = F.tanh(self.conv5(x))
        ####################################
        #          YOUR CODE HERE          #
        ####################################


        ##########       END      ##########

        return x
