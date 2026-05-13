"""
任务 C：模型对比 — ResNet18 vs MobileNetV2
运行方式：在容器中执行 python task_c_model_comparison.py
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

# ── 定义两个模型 ──
model_resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model_resnet.fc = nn.Linear(model_resnet.fc.in_features, num_classes)

model_mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
model_mobilenet.classifier[1] = nn.Linear(model_mobilenet.classifier[1].in_features, num_classes)

model_resnet = model_resnet.to(device)
model_mobilenet = model_mobilenet.to(device)

print(f"ResNet18 参数: {sum(p.numel() for p in model_resnet.parameters()):,}")
print(f"MobileNetV2 参数: {sum(p.numel() for p in model_mobilenet.parameters()):,}")

def train_frozen(model, classifier_params, epochs=5, name=""):
    """冻结特征提取训练"""
    for p in model.parameters():
        p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_params, lr=0.001)

    val_accs = []
    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, pred = model(images).max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        acc = 100. * correct / total
        val_accs.append(acc)
        print(f"{name} Epoch [{epoch+1}/{epochs}] | Val Acc: {acc:.2f}%")
    return val_accs

print("\n=== 训练 ResNet18 ===")
resnet_accs = train_frozen(model_resnet, model_resnet.fc.parameters(), name="ResNet18")

print("\n=== 训练 MobileNetV2 ===")
mobile_accs = train_frozen(model_mobilenet, model_mobilenet.classifier[1].parameters(), name="MobileNetV2")

# ── 对比图 ──
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 6), resnet_accs, 'bo-', label='ResNet18 (11.7M)', linewidth=2)
ax.plot(range(1, 6), mobile_accs, 'go-', label='MobileNetV2 (3.5M)', linewidth=2)
ax.set_xlabel('Epoch'); ax.set_ylabel('Validation Accuracy (%)')
ax.set_title('ResNet18 vs MobileNetV2: Feature Extraction')
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('task_c_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nResNet18 最佳准确率: {max(resnet_accs):.2f}%")
print(f"MobileNetV2 最佳准确率: {max(mobile_accs):.2f}%")
print("已保存: task_c_model_comparison.png")
print("\n任务 C 完成！")
