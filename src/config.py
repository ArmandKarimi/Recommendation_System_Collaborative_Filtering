import torch 
import torch.nn as nn
import torch.optim as optim


DEVICE = (
    "cuda" if torch.cuda.is_available() else 
    "mps" if torch.backends.mps.is_available() else 
    "cpu"
)

FRACTION = 1

EPOCHS = 300
N_FACTORS = 5

