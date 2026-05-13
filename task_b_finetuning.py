"""
任务 B：微调 — 解冻更多层提升准确率
运行方式：在容器中执行 python task_b_finetuning.py
或在 Jupyter 中逐段运行
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

full_dataset = datasets.ImageFolder(root=data_dir, transform=train_transform)
class_names = full_dataset.classes
num_classes = len(class_names)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ── 微调模型：解冻 layer4 和 fc ──
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name, param in model_ft.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)

trainable_params = [p for p in model_ft.parameters() if p.requires_grad]
print(f"可训练参数: {sum(p.numel() for p in trainable_params):,}")
print(f"总参数: {sum(p.numel() for p in model_ft.parameters()):,}")
print(f"可训练比例: {sum(p.numel() for p in trainable_params) / sum(p.numel() for p in model_ft.parameters()) * 100:.1f}%")

model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {'params': model_ft.layer4.parameters(), 'lr': 0.0001},
    {'params': model_ft.fc.parameters(), 'lr': 0.001}
], lr=0.001)

# ── 训练 ──
epochs = 5
ft_train_losses, ft_val_accs = [], []

for epoch in range(epochs):
    model_ft.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model_ft(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    model_ft.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            _, pred = model_ft(images).max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(train_loader)
    val_acc = 100. * correct / total
    ft_train_losses.append(avg_loss)
    ft_val_accs.append(val_acc)
    print(f"Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

# ── 可视化（与任务A结果对比，这里用示例值） ──
feature_acc = 92.0  # ← 替换为任务A的实际最佳验证准确率
finetune_acc = max(ft_val_accs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(range(1, epochs+1), ft_train_losses, 'ro-', label='Fine-tuning', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss (Fine-tuning)')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.bar(['Feature Extraction', 'Fine-tuning'],
        [feature_acc, finetune_acc],
        color=['#4e79a7', '#e15759'], edgecolor='black')
for i, (label, acc) in enumerate(zip(['Feature Extraction', 'Fine-tuning'], [feature_acc, finetune_acc])):
    ax2.text(i, acc + 1, f'{acc:.1f}%', ha='center', fontsize=14, fontweight='bold')
ax2.set_ylim(60, 102)
ax2.set_ylabel('Best Validation Accuracy (%)')
ax2.set_title('Best Validation Accuracy Comparison')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('task_b_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n微调最佳验证准确率: {finetune_acc:.2f}%")
print("已保存: task_b_comparison.png")
print("\n任务 B 完成！")
