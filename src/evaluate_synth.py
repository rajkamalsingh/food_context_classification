import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -----------------------
# Paths
# -----------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "outputs" / "models" / "synth_resnet50.pth"

CM_DIR = BASE_DIR / "outputs" / "confusion_matrices"
CM_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------
# Parameters
# -----------------------
BATCH_SIZE = 16
IMG_SIZE = 224

# -----------------------
# Transform
# -----------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

# -----------------------
# Dataset
# -----------------------
test_dataset = datasets.ImageFolder(DATA_DIR / "test", transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class_names = test_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)

# -----------------------
# Model
# -----------------------
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# -----------------------
# Predict
# -----------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -----------------------
# Metrics
# -----------------------
acc = accuracy_score(all_labels, all_preds)
print(f"\nTest Accuracy: {acc:.4f}")

print("\nClassification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# -----------------------
# Confusion Matrix
# -----------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(7,6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Oranges",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Synthetic Model Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()
plt.savefig(CM_DIR / "synth_confusion_matrix.png")
plt.show()