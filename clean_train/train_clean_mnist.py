import sys
sys.path = ["./"] + sys.path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.simplecnn import SimpleCNN


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = "mnist_clean_model.pth"


# ------------------------------------------------------------
# DATA
# ------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
)


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
model = SimpleCNN().to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    model.parameters(),
    lr=LR,
    momentum=0.9,
)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
@torch.no_grad()
def evaluate():
    model.eval()

    correct = 0
    total = 0

    for x, y in test_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x).argmax(1)

        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------
for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        pred = model(x)

        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    acc = evaluate()

    print(
        f"Epoch {epoch} | "
        f"Loss: {running_loss / len(train_loader):.4f} | "
        f"Test Accuracy: {acc:.4f}"
    )


# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
torch.save(model.state_dict(), SAVE_PATH)
print(f"Saved model to {SAVE_PATH}")