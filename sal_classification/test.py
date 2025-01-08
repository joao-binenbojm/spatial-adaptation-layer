from sal_classification.simulation import apply_affine
import matplotlib.pyplot as plt
import torch

a = torch.zeros(1, 1, 70, 240)
a[:,:,20:50, 20:220] = 1.0

b = apply_affine(a, 10.0, 10.0, 15*torch.pi/180)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(a.squeeze())
axs[1].imshow(b.squeeze())
plt.savefig('block.jpg')
