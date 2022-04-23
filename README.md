# GANime-PyTorch
A Deep Convolutional GAN created for generation of low-medium (64x64 px) resolution images.

Demonstration can be found in https://konkuad.github.io/gan.html, which is created and run on a google colab notebook.<br>
The dataset used in the demo can be downloaded from https://www.kaggle.com/prasoonkottarathil/gananime-lite.<br>
A 64x64 processed version (used in my google drive) can be found here https://drive.google.com/file/d/1zUOt42VfZ9jaPNtwZhZOV4pf2mCx0uWL/view?usp=sharing.

To install, in terminal

```
git clone https://github.com/konkuad/GANime
cd GANime
pip install .
```

To create a dataloader from an image folder

```python
from torch.utils.data import DataLoader
import torchvision.transforms as T

from GANime.gan import plotter
from GANime.datasets import ImageOnlyDataset

resize_transform = T.Compose([
    T.Resize(64), #resize
    T.ToTensor(), #convert to tensor
    T.Lambda(lambda x: (255*x).int()/127.5-1) #normalize color channels to -1 and 1
])

ds = ImageOnlyDataset('out2', resize_transform)
dl = DataLoader(ds, batch_size=128, shuffle=True)

#plot a few images
it = next(iter(dl))
plotter(it, rows=8, columns=8, renormalize_func = lambda x: (x*127.5+127.5).astype(int))
```

Training with your own dataloader.

```
from GANime.gan import GAN
seed_size = 128
gan_model = GAN(seed_size)
gan_model.train(dl,
                num_epochs = 20,
                batch_size = 128,
                plot = True,)

#save trained model
import torch
torch.save(gan_model.gen.state_dict(), 'gen.pt')
torch.save(gan_model.dis.state_dict(), 'dis.pt')
```

Doing inference with your trained model

```
from GANime.gan import GAN

#initialize new model with random weights
new_gan_model = GAN(seed_size)

#load model trained for 100 epochs
new_gan_model.gen.load_state_dict(torch.load('GANime/example_models/gen.pt'))
new_gan_model.dis.load_state_dict(torch.load('GANime/example_models/dis.pt'))

#infer generation
generated_images = new_gan_model.generate(num_rows=8,
               num_cols=8,
               plot=True,
               device='cpu',
               return_noise=False
              )
```

