from NEAT import NEAT

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

# MNIST Dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # flatten 28x28 -> 784
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# Initialize genome
device = 'cuda' if torch.cuda.is_available() else 'cpu'
neat = NEAT(P=500, N_max=800, C_max=2000, num_inputs=28*28, num_outputs=10, device=device)

def run():
    for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
        X_input = X_batch.to(device)
        outputs = neat.forward(X_input)
        print(outputs.shape)
        # break

# run()
import cProfile
cProfile.run('run()')