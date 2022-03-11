# GANime-PyTorch
A Deep Convolutional GAN created for generation of low-medium (64x64 px) resolution images.

Demonstration can be found in https://konkuad.github.io/gan.html, which is created and run on a google colab notebook.<br>
The dataset used in the demo can be downloaded from https://www.kaggle.com/prasoonkottarathil/gananime-lite.<br>
A 64x64 processed version (used in my google drive) can be found here https://drive.google.com/file/d/1zUOt42VfZ9jaPNtwZhZOV4pf2mCx0uWL/view?usp=sharing.

To create a dataset from the 64x64 .npy version, simply use

```python
import numpy as np
import torch

path = 'path to your file'
arr = np.moveaxis(np.load(path)/127.5-1,-1,1) #(-1,1) normalize and format to C,W,H
dataset = torch.utils.data.DataLoader(torch.tensor(arr,  dtype=torch.float), batch_size=128, num_workers=2, shuffle=True)
```

Example of using the package on your dataset.
```
from gan import GAN

seed_size = 128
gan_model = GAN(seed_size)
gan_model.train(dataset)
```
