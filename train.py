import os
from logging import getLogger

import torch
from torch.optim  import Adam
from  torch.nn import CrossEntropyLoss

from src.engine import train_loop
from src.dataloaders import read_dataset
from src.models import ViT


logger = getLogger('vit')

N_EPOCHS = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'weights/vit.pth'

# create directory if not exists
if not os.path.exists(os.path.dirname(SAVE_PATH)):
    os.makedirs(os.path.dirname(SAVE_PATH))

def main():
    logger.info(f'Using device {device}')
    train_loader, test_loader = read_dataset()
    model = ViT((1, 28, 28), number_patches=7, number_blocks=2, hidden_dimension=8, number_heads=2, output_dimension=10).to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion  = CrossEntropyLoss()
    train_loop(train_loader, model, criterion, optimizer, N_EPOCHS, device)
    torch.save(model.state_dict(), SAVE_PATH)
    
if __name__ == '__main__':
    main()