from torch import nn

class generator(nn.Module):

  def __init__(self, seed_size=128):

    super(generator, self).__init__()

    self.conv_transpose_block_1 = nn.Sequential(
        nn.ConvTranspose2d(seed_size, 1024, 4, 1, 0, bias=False),
        nn.BatchNorm2d(1024),
        nn.ReLU()) #shape = (1024,4,4)
    
    self.conv_transpose_block_2 = nn.Sequential(
        nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU()) #shape = (512,8,8)
    
    self.conv_transpose_block_3 = nn.Sequential(
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU()) #shape = (256,16,16)

    self.conv_transpose_block_4 = nn.Sequential(
        nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.ReLU()) #shape = (128,32,32)

    self.conv_transpose_block_5 = nn.Sequential(
        nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
        nn.Tanh()) #shape = (3,64,64), outputs are normalized to (-1,1)

  def forward(self, input):
    input = self.conv_transpose_block_1(input)
    input = self.conv_transpose_block_2(input)
    input = self.conv_transpose_block_3(input)
    input = self.conv_transpose_block_4(input)
    input = self.conv_transpose_block_5(input)
    return input
  
  
class discriminator(nn.Module):

  def __init__(self, seed_size):

    super(discriminator, self).__init__()
    self.block = nn.Sequential(

        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(128, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(256, 512, 4, 2, 1, bias=False),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(0.2, inplace=True),

        nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )

  def forward(self,input):
    return self.block(input)
