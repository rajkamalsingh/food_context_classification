import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
PLOT_DIR = OUTPUT_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Params
# -----------------------
BATCH_SIZE = 16
EPOCHS = 10
LR = 1e-4
IMG_SIZE = 224

# -----------------------
# Synthetic transforms
# -----------------------
data_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3)
        ], p=0.35),

        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.15,
                saturation=0.10
            )
        ], p=0.50),

        transforms.RandomApply([
            transforms.RandomGrayscale(p=1.0)
        ], p=0.10),

        transforms.ToTensor(),
    ]),

    "val": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]),

    "test": transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ]),
}

# -----------------------
# Data
# -----------------------
image_datasets = {
    x: datasets.ImageFolder(DATA_DIR / x, transform=data_transforms[x])
    for x in ["train", "val", "test"]
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True)
    for x in ["train", "val", "test"]
}

dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}
class_names = image_datasets["train"].classes

print("Classes:", class_names)

# -----------------------
# Model
# -----------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -----------------------
# Train
# -----------------------
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if phase == "train":
            train_losses.append(epoch_loss)
            train_accs.append(epoch_acc.cpu().item())
        else:
            val_losses.append(epoch_loss)
            val_accs.append(epoch_acc.cpu().item())

            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

# -----------------------
# Save
# -----------------------
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), MODEL_DIR / "synth_resnet50.pth")

print("\nBest Validation Accuracy:", best_acc.item())

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.title("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train")
plt.plot(val_accs, label="Val")
plt.title("Accuracy")
plt.legend()

plt.tight_layout()
plt.savefig(PLOT_DIR / "synth_training_curves.png")
plt.show()