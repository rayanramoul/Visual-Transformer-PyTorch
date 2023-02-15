import numpy as np 
from logging import getLogger
from tqdm  import tqdm 

import torch 
import torch.nn as nn
from torch.optim  import Adam
from  torch.nn import CrossEntropyLoss
from torcch.utils.data import DataLoader

from torchvision.transforms  import ToTensor
from torchvision.datasets import MNIST

logger = getLogger('vit')

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device {device}')
    
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion  = CrossEntropyLoss()