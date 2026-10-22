"""
nn.convo2D(in_channels , out_channels ,kernal_size , stride , padding)

# in_channels = how many layers come in 

black & white = 1
colour RGB = 3
[batch,channels,height,width]
[1,1,28,28]

# out_channels
each output channel = 1 pattern detector

# kernal_size = how big we want out batch size

3 - 3x3 ( most common)
5 - 5x5

# stride = how fast the pixels will move
1 = see every pixel
2 = skip every second pixel

# padding = do we need to see at the edges or not

"""


import torch
import torch.nn as nn

image = torch.randn(1,1,28,28)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    stride=1,
    padding=1
)

output = conv(image)

print("Input image",image.shape)
print("Output image",output.shape)


#change the functions and try diff things


# CNN ARCHITECTURE FLOW

"""
image
conv2d (find simple patterns)
reLU (keep only useful signals)
pool(shorten the image size and strengthen it)
conv2d
ReLU
pool
image flattening (convert pattern to list)
linear
output
"""