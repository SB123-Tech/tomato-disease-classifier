"""
任务 A：迁移学习基础 — 冻结特征提取
运行方式：在容器中执行 python task_a_transfer_learning.py
或在 Jupyter 中逐段运行
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前设备: {device}")

# ── 1. 数据加载 ──
data_dir = './data/PlantVillage'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

try:
    full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
    print(f"成功加载数据，共 {len(full_dataset)} 张图片")
except FileNotFoundError:
    print(f"数据不存在于 {data_dir}，请查看手动下载说明")
    raise

class_names = full_dataset.classes
print(f"类别: {class_names}")

# ── 2. 划分训练集/验证集 ──
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"训练集: {train_size}, 验证集: {val_size}")

# ── 3. 加载预训练模型 ──
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换分类头
num_classes = len(class_names)
model.fc = nn.Linear(model.fc.in_features, num_classes)

trainable_params = [p for p in model.parameters() if p.requires_grad]
print(f"可训练参数: {sum(p.numel() for p in trainable_params)}")
print(f"总参数: {sum(p.numel() for p in model.parameters())}")

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# ── 4. 训练 ──
epochs = 5
train_losses, val_accs = [], []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model(images).max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    val_acc = 100. * correct / total
    train_losses.append(avg_loss)
    val_accs.append(val_acc)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ── 5. 可视化 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, epochs+1), train_losses, 'bo-', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Feature Extraction)')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, epochs+1), val_accs, 'ro-', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy (%)')
ax2.set_title('Validation Accuracy (Feature Extraction)')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task_a_feature_extraction.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n最佳验证准确率: {max(val_accs):.2f}%")
print("已保存: task_a_feature_extraction.png")
print("\n任务 A 完成！")
