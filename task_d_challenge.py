"""
任务 D：综合项目 — 从零训练 vs 迁移学习大比拼
运行方式：在容器中执行 python task_d_challenge.py
或在 Jupyter 中逐段运行
"""
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_dir = './data/PlantVillage'
num_classes = 4  # 番茄病害类别数

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
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
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
val_dataset.dataset.transform = test_transform
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"训练集: {train_size}, 验证集: {val_size}")

def train_model(model, criterion, optimizer, epochs=5, name=""):
    """通用训练函数"""
    train_losses, val_accs = [], []
    start_time = time.time()

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
        print(f"  {name} Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}%")

    elapsed = time.time() - start_time
    print(f"  {name} 完成！用时 {elapsed:.1f}s，最佳准确率: {max(val_accs):.2f}%")
    return train_losses, val_accs, elapsed

criterion = nn.CrossEntropyLoss()

# ── 方案 1：从零训练 ──
print("\n=== 方案1：从零训练 ===")
model_scratch = models.resnet18(weights=None)
model_scratch.fc = nn.Linear(model_scratch.fc.in_features, num_classes)
model_scratch = model_scratch.to(device)
opt_scratch = optim.Adam(model_scratch.parameters(), lr=0.001)
loss_scratch, acc_scratch, time_scratch = train_model(model_scratch, criterion, opt_scratch, name="Scratch")

# ── 方案 2：冻结特征提取 ──
print("\n=== 方案2：冻结特征提取 ===")
model_frozen = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in model_frozen.parameters():
    p.requires_grad = False
model_frozen.fc = nn.Linear(model_frozen.fc.in_features, num_classes)
model_frozen = model_frozen.to(device)
opt_frozen = optim.Adam(model_frozen.fc.parameters(), lr=0.001)
loss_frozen, acc_frozen, time_frozen = train_model(model_frozen, criterion, opt_frozen, name="Frozen")

# ── 方案 3：微调 ──
print("\n=== 方案3：微调 ===")
model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for name_param, param in model_ft.named_parameters():
    if 'layer4' not in name_param and 'fc' not in name_param:
        param.requires_grad = False
model_ft.fc = nn.Linear(model_ft.fc.in_features, num_classes)
model_ft = model_ft.to(device)
opt_ft = optim.Adam([
    {'params': model_ft.layer4.parameters(), 'lr': 0.0001},
    {'params': model_ft.fc.parameters(), 'lr': 0.001}
], lr=0.001)
loss_ft, acc_ft, time_ft = train_model(model_ft, criterion, opt_ft, name="Fine-tune")

# ── 可视化 ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

labels = ['从零训练', '冻结特征', '微调']
accs = [max(acc_scratch), max(acc_frozen), max(acc_ft)]
times = [f"{time_scratch:.0f}s", f"{time_frozen:.0f}s", f"{time_ft:.0f}s"]
colors = ['#4e79a7', '#f28e2b', '#e15759']
bars = ax1.bar(labels, accs, color=colors, edgecolor='black')
for bar, acc, t in zip(bars, accs, times):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%\n({t})', ha='center', fontsize=11)
ax1.set_ylim(50, 102)
ax1.set_ylabel('Best Validation Accuracy (%)')
ax1.set_title('Training Strategy Comparison')
ax1.grid(True, alpha=0.3, axis='y')

ax2.plot(range(1, 6), loss_scratch, 'bo-', label='从零训练', linewidth=2)
ax2.plot(range(1, 6), loss_frozen, 'ro-', label='冻结特征', linewidth=2)
ax2.plot(range(1, 6), loss_ft, 'go-', label='微调', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Loss')
ax2.set_title('Training Loss Comparison')
ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('task_d_final_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n从零训练: {max(acc_scratch):.2f}% ({time_scratch:.0f}s)")
print(f"冻结特征: {max(acc_frozen):.2f}% ({time_frozen:.0f}s)")
print(f"微调:     {max(acc_ft):.2f}% ({time_ft:.0f}s)")
print("已保存: task_d_final_comparison.png")
print("\n任务 D 完成！")
