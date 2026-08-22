import torch
import numpy as np
from torch.utils.data import Dataset

class Dataset(torch.utils.data.Dataset):
    """
      INPUT:  [b, c, d, h, w]
    """
    def __init__(self, INPUT: np.ndarray, OUTPUT: np.ndarray, augment: bool = False):
        self.INPUT = INPUT
        self.OUPUT = OUTPUT
        self.augment = augment
    
    def __len__(self):
        return self.INPUT.shape[0]

    def __getitem__(self, idx):
        input = self.INPUT[idx]   # [c,d,h,w]
        output = self.OUPUT[idx]  # [c,d,h,w]

        input = torch.from_numpy(input.astype(np.float32))
        output = torch.from_numpy(output.astype(np.float32))
 
        if self.augment:
            if torch.rand(1) < 0.5: 
                input = torch.flip(input, [1])
                output = torch.flip(output, [1])
            if torch.rand(1) < 0.5:
                input = torch.flip(input, [2])
                output = torch.flip(output, [2])
            if torch.rand(1) < 0.5:
                input = torch.flip(input, [3])
                output = torch.flip(output, [3])

        return input, output

