import copy
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "data_mc"
BATCH_SIZE = 16
EPOCHS = 6
LR = 1e-3
VAL_SPLIT = 0.2
SEED = 42

torch.manual_seed(SEED)

tfm = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

dataset = datasets.ImageFolder(DATA_DIR, transform=tfm)
num_classes = len(dataset.classes)
print("Classes:", dataset.classes)
print("class_to_idx:", dataset.class_to_idx)

n_total = len(dataset)
n_val = int(n_total * VAL_SPLIT)
n_train = n_total - n_val
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(DEVICE)

for p in model.parameters():
    p.requires_grad = False
for p in model.fc.parameters():
    p.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

best_state = copy.deepcopy(model.state_dict())
best_val_acc = 0.0


def batch_acc(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds.eq(labels)).float().mean().item()


for epoch in range(EPOCHS):
    start = time.time()

    model.train()
    tr_loss = 0.0
    tr_acc = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item()
        tr_acc += batch_acc(logits.detach(), labels)

    tr_loss /= max(1, len(train_loader))
    tr_acc /= max(1, len(train_loader))

    model.eval()
    va_loss = 0.0
    va_acc = 0.0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            va_loss += loss.item()
            va_acc += batch_acc(logits, labels)

    va_loss /= max(1, len(val_loader))
    va_acc /= max(1, len(val_loader))

    if va_acc > best_val_acc:
        best_val_acc = va_acc
        best_state = copy.deepcopy(model.state_dict())

    print(
        f"Epoch {epoch+1}/{EPOCHS} "
        f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} "
        f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} "
        f"time={time.time()-start:.1f}s"
    )

model.load_state_dict(best_state)
torch.save(
    {
        "state_dict": model.state_dict(),
        "classes": dataset.classes,
        "class_to_idx": dataset.class_to_idx
    },
    "model_mc.pth"
)
print("Saved model_mc.pth")
