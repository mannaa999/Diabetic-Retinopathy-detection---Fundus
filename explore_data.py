import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import models
import torch.optim as optim
import time
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Set path to the data folder
DATA_DIR = Path("/users/km/downloads/diabeticretinopathy/gaussian_filtered_images/gaussian_filtered_images")

# List the class folders
classes = sorted(os.listdir(DATA_DIR))
print("Classes found:", classes)

# Count images per class
for cls in classes:
    cls_path = DATA_DIR / cls
    if cls_path.is_dir():
        n_images = len(os.listdir(cls_path))
        print(f"{cls}: {n_images} images")


# Map each class folder to a binary label
CLASS_TO_BINARY = {
    "No_DR": 0,
    "Mild": 1,
    "Moderate": 1,
    "Severe": 1,
    "Proliferate_DR": 1,
}

# Build a dataframe of filepath -> original class -> binary label
records = []
for cls in classes:
    cls_path = DATA_DIR / cls
    if not cls_path.is_dir():
        continue
    for img_path in cls_path.iterdir():
        if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
            records.append({
                "filepath": str(img_path),
                "original_class": cls,
                "label": CLASS_TO_BINARY[cls],
            })

df = pd.DataFrame(records)
print(df["label"].value_counts())
print(df["original_class"].value_counts())

# Stratify on original_class (not just binary label) so each DR
# subtype — including the rare Severe/Proliferate_DR ones — is
# proportionally represented in both train and val
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["original_class"],
)

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

print("Train shape:", train_df.shape)
print("Val shape:  ", val_df.shape)
print("Train label balance:", train_df["label"].value_counts().to_dict())
print("Val label balance:  ", val_df["label"].value_counts().to_dict())


class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = row["label"]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ImageNet normalization stats — required since we're using a
# pretrained ResNet18 that expects inputs normalized this way
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # safety net even though images are already this size
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

train_dataset = RetinopathyDataset(train_df, transform=train_transform)
val_dataset = RetinopathyDataset(val_df, transform=val_transform)

print("Train dataset size:", len(train_dataset))
print("Val dataset size:  ", len(val_dataset))

# Sanity check — pull one sample and confirm shape/dtype
img, lbl = train_dataset[0]
print("Image shape:", img.shape, "| Label:", lbl.item())


BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,   # keep at 0 for now — bump later once things are stable
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,   # no need to shuffle validation
    num_workers=0,
)

# Sanity check — pull one batch and confirm shapes
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch label shape:", labels.shape)
print("Sample labels:", labels[:10])



device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else
                       "cpu")
print("Using device:", device)

# Load pretrained ResNet18
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# Freeze all layers initially — we'll only train the new final layer first
for param in model.parameters():
    param.requires_grad = False

# Replace the final fully connected layer
# Original: model.fc = Linear(512, 1000)  -- for 1000 ImageNet classes
# New: Linear(512, 2) -- for our binary classification
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# The new fc layer's params are trainable by default (requires_grad=True),
# even though we froze everything else above
model = model.to(device)

print(model.fc)



criterion = nn.CrossEntropyLoss()

# Only the new fc layer has requires_grad=True, so this optimizer
# will only update those parameters — the frozen backbone is untouched
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

NUM_EPOCHS = 10

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

best_val_acc = 0.0
for epoch in range(NUM_EPOCHS):
    start = time.time()

    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    elapsed = time.time() - start
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f}s")



from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Load the best checkpoint back into a fresh model
best_model = models.resnet18(weights=None)  # no need for pretrained weights, we're loading our own
num_features = best_model.fc.in_features
best_model.fc = nn.Linear(num_features, 2)
best_model.load_state_dict(torch.load("best_model.pth", map_location=device))
best_model = best_model.to(device)
best_model.eval()

# Run predictions on the full validation set
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print("                Predicted No_DR   Predicted DR")
print(f"Actual No_DR         {cm[0][0]:>6}          {cm[0][1]:>6}")
print(f"Actual DR            {cm[1][0]:>6}          {cm[1][1]:>6}")

# Precision, recall, F1 for both classes
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=["No_DR", "DR"]))

import torch.nn.functional as F
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.eval()
        output = self.model(input_tensor)

        self.model.zero_grad()
        # Backprop from the score of the target class
        output[0, class_idx].backward()

        # Global-average-pool the gradients -> importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted sum of activation maps, then ReLU (keep only positive influence)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to match input image size (224x224)
        cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to 0-1
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


misclassified_idx = np.where(all_preds != all_labels)[0]
def show_gradcam(model, image_path, true_label, pred_label, target_layer):
    grad_cam = GradCAM(model, target_layer)

    # Load and preprocess image
    image = Image.open(image_path).convert("RGB").resize((224, 224))
    input_tensor = val_transform(image).unsqueeze(0).to(device)
    input_tensor.requires_grad_()

    cam = grad_cam.generate(input_tensor, class_idx=pred_label)

    # Overlay heatmap on original image
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(image)
    overlay = (0.5 * orig_np + 0.5 * heatmap).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(orig_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    label_names = {0: "No_DR", 1: "DR"}
    axes[2].set_title(f"True: {label_names[true_label]} | Pred: {label_names[pred_label]}")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


# Target the last conv block of ResNet18 -- layer4 -- which holds
# the highest-level spatial features before global pooling
target_layer = best_model.layer4[-1]

# Try it on a correctly classified DR image, and a misclassified one
correct_idx = np.where((all_preds == all_labels) & (all_labels == 1))[0][0]
row = val_df.iloc[correct_idx]
fig1 = show_gradcam(best_model, row["filepath"], all_labels[correct_idx], all_preds[correct_idx], target_layer)
fig1.savefig("gradcam_correct.png", dpi=150)
plt.show()

if len(misclassified_idx) > 0:
    idx = misclassified_idx[0]
    row = val_df.iloc[idx]
    fig2 = show_gradcam(best_model, row["filepath"], all_labels[idx], all_preds[idx], target_layer)
    fig2.savefig("gradcam_misclassified.png", dpi=150)
    plt.show()