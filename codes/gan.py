import torch
from torch import nn
from components import generator, discriminator
import numpy as np
import matplotlib.pyplot as plt

def plotter(data,
            rows=8,
            columns=8,
           ):
  data = np.moveaxis(np.array(data*127.5+127.5,dtype=int),1,-1)
  fig,ax = plt.subplots(rows,columns,figsize=(10,10))
  for i in range(rows):
    for j in range(columns):
      ax[i,j].imshow(data[i+8*j])
      ax[i,j].axis('off')
  plt.show()

def weight_init(model):
  std = 0.02
  for layer in model.modules():
    name = layer.__class__.__name__
    if 'Conv' in name: 
      #for all convolution and transpose convolution layers
      #init all weights to have a mean of 0 and std of 0.02
      mean = 0
      nn.init.normal_(layer.weight.data, mean, std)
    elif 'BatchNorm' in name: 
      #for batchnorms, weight and bias refer to the gamma and beta values used in affine transformation
      mean = 1
      nn.init.normal_(layer.weight.data, mean, std)
      nn.init.constant_(layer.bias.data, 0)
    else:
      continue

class GAN:
  
  def __init__(self,
               seed_size = 128
              ):
    self.seed_size = 128
    self.gen = generator(seed_size)
    weight_init(self.gen)
    self.dis = discriminator(seed_size)
    weight_init(self.dis)
    
  def train(self,
            dataset,
            dis_optimizer = torch.optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999)),
            gen_optimizer = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999)),
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), #for GPU support
            num_epochs = 100,
            batch_size = 128,
            plot = True,
           ):
    assert type(dataset)==torch.utils.data.dataloader.DataLoader, "Require PyTorch's DataLoader for your dataset"
    loss = nn.BCELoss()
    self.dis.to(device)
    self.gen.to(device)
    
    if plot:
      fixed_seed = torch.randn(64, self.seed_size, 1, 1, device=device, dtype=torch.float)
      
    for epoch in range(num_epochs):
      for i, data in enumerate(dataset):
        b = data.shape[0]
        
        '''
        First step, we update the discriminator's weights
        1. Have the generator generate fake images
        2. Input real images from trainloader
        3. Predict both fake and real images
        4. Create labels accordingly (fake = 0.1 and real = 0.9, setting 0 and 1 would be too rigid)
        5. Compute loss and train
        '''
        self.dis.zero_grad()

        #step 1
        noise = torch.randn(b, seed_size, 1, 1, device=device, dtype=torch.float)
        fake_img = self.gen(noise)

        #step 2
        real_img = data.to(device)   

        #step 3
        fake_dis_pred = self.dis(fake_img.detach()) #we detach the fake images so that the generator isn't updated
        real_dis_pred = self.dis(real_img)

        #step 4
        fake_dis_lbl = torch.full((b,), 0.1, device=device, dtype=torch.float)
        real_dis_lbl = torch.full((b,), 0.9, device=device, dtype=torch.float)

        #step 5
        fake_dis_loss = loss(fake_dis_pred.view(-1), fake_dis_lbl)
        real_dis_loss = loss(real_dis_pred.view(-1), real_dis_lbl)

        real_dis_loss.backward()
        fake_dis_loss.backward()
        dis_loss = fake_dis_loss + real_dis_loss #loss for final report
        dis_optimizer.step()

        '''
        Then, we update the generator's weights
        1. Have the generator generate fake images
        2. Have the trained discriminator predict the fake images
        3. Train the generator by tricking the discriminator into predicting the images into the "real" class
        '''
        self.gen.zero_grad()

        #step 1 is already done (using "fake_img")

        #step 2
        fake_gen_pred = self.dis(fake_img)

        #step 3
        gen_loss = loss(fake_gen_pred.view(-1), real_dis_lbl)
        gen_loss.backward()
        gen_optimizer.step()

        '''
        Lastly, we visualize
        '''

        dis_loss_show = '{:.4f}'.format(dis_loss.item())
        gen_loss_show = '{:.4f}'.format(gen_loss.item())
        
        if i%25==0:
          print(f'Iteration {i}\t[Epoch {epoch}/{num_epochs}]\tLosses:\t L_discriminator = {dis_loss_show}\t L_generator = {gen_loss_show}')
          #the only reason I output it this way is to make it look nice, nothing more
          
      if plot: #end of epoch, plot the fixed seed
        with torch.no_grad():
          pred = self.gen(visualize_noise)
          plotter(pred.cpu())
          
  def generate(self,
               num_row=8,
               num_col=8,
               device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               plot=True,
               return_noise=True
              ):
    noise = torch.randn(num_rows*num_cols, self.seed_size, 1, 1, device=device, dtype=torch.float)
    generated = self.gen(noise).cpu()
    
    if plot:
      plotter(generated,
              rows=num_row,
              columns=num_col
             )
      
    if return_noise:
      return generated, noise.cpu()
    else:
      return generated
