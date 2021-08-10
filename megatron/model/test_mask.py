import torch
import numpy as np

mask = torch.tensor([[1, 1, 0], [1, 1, 0]])
mask_b1s = mask.unsqueeze(1)
mask_bs1 = mask.unsqueeze(2)

mask_bss = mask_b1s * mask_bs1

extended_mask = mask_bss.unsqueeze(1)

extended_mask = (extended_mask < 0.5)

print(extended_mask)
