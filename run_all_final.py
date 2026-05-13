#!/usr/bin/env python3
"""
Final consolidated run: Tasks C (MobileNetV2) + Task D (3-strategy)
With unbuffered output for real-time monitoring.
"""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, sys, time, json
from pathlib import Path

# === Setup ===
BASE = Path(__file__).parent
DATA_DIR = BASE / 'data' / 'PlantVillage'
OUT_DIR = BASE / 'output'
RES_DIR = BASE / 'results'
OUT_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data ===
tf_train = transforms.Compose([
    transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
tf_test = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

ds = datasets.ImageFolder(root=str(DATA_DIR), transform=tf_train)
short = {'Tomato___healthy': 'Healthy', 'Tomato___Early_blight': 'Early Blight',
         'Tomato___Late_blight': 'Late Blight', 'Tomato___Bacterial_spot': 'Bacterial Spot'}
labels = [short.get(c, c) for c in ds.classes]
nc = len(ds.classes)

ts = int(0.8 * len(ds))
vs = len(ds) - ts
tr_ds, vl_ds = random_split(ds, [ts, vs], generator=torch.Generator().manual_seed(42))
vl_ds.dataset.transform = tf_test
tr_ld = DataLoader(tr_ds, batch_size=32, shuffle=True)
vl_ld = DataLoader(vl_ds, batch_size=32, shuffle=False)
print(f"[Data] Train={ts} Val={vs} Classes={nc} Device={device}", flush=True)

# === Results from Tasks A & B (completed earlier) ===
rA = {'losses': [0.5606, 0.2810, 0.2141, 0.1880, 0.1721],
      'accs': [93.29, 95.17, 96.00, 96.00, 95.17], 'best': 96.00, 'time': 921.2}
rB = {'losses': [0.1673, 0.0167, 0.0106, 0.0030, 0.0038],
      'accs': [98.57, 99.25, 99.47, 99.40, 99.32], 'best': 99.47, 'time': 1178.6}
# rC_rn from earlier run
rC_rn = {'accs': [92.31, 94.34, 95.63, 95.40, 96.30], 'best': 96.30, 'time': 943.1,
         'params': 11178564}
print(f"[Prev] A={rA['best']:.1f}% B={rB['best']:.1f}% C_RN={rC_rn['best']:.1f}%", flush=True)

# === Train helpers ===
def train_model(model, opt, epochs, name):
    criterion = nn.CrossEntropyLoss()
    losses, accs = [], []
    t0 = time.time()
    for ep in range(epochs):
        model.train(); rl = 0.0
        for im, lb in tr_ld:
            im, lb = im.to(device), lb.to(device)
            ls = criterion(model(im), lb)
            opt.zero_grad(); ls.backward(); opt.step()
            rl += ls.item()
        model.eval(); c = t = 0
        with torch.no_grad():
            for im, lb in vl_ld:
                im, lb = im.to(device), lb.to(device)
                _, p = model(im).max(1)
                c += p.eq(lb).sum().item(); t += lb.size(0)
        acc = 100.*c/t; loss = rl/len(tr_ld)
        losses.append(loss); accs.append(acc)
        print(f"  {name} [{ep+1}/{epochs}] Loss={loss:.4f} Acc={acc:.2f}%", flush=True)
    et = time.time() - t0
    print(f"  {name} DONE | Best={max(accs):.2f}% | Time={et:.0f}s", flush=True)
    return {'losses': losses, 'accs': accs, 'best': max(accs), 'time': et}

# ============================================================
# TASK C: MobileNetV2
# ============================================================
print("\n=== TASK C: MobileNetV2 ===", flush=True)
mb = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
for p in mb.parameters(): p.requires_grad = False
mb.classifier[1] = nn.Linear(mb.classifier[1].in_features, nc)
mb_p = list(mb.classifier[1].parameters())
for p in mb_p: p.requires_grad = True
mb = mb.to(device)
mb_params = sum(p.numel() for p in mb.parameters())
print(f"  Params: {mb_params:,}", flush=True)
rC_mb = train_model(mb, optim.Adam(mb_p, lr=0.001), 3, "MobileNetV2")
rC_mb['params'] = mb_params

rC = {'resnet': rC_rn, 'mobilenet': rC_mb}

# ============================================================
# TASK D: Three strategies
# ============================================================
print("\n=== TASK D: Strategy 1 — Scratch ===", flush=True)
mS = models.resnet18(weights=None)
mS.fc = nn.Linear(mS.fc.in_features, nc)
mS = mS.to(device)
rD_s = train_model(mS, optim.Adam(mS.parameters(), lr=0.001), 3, "Scratch")

print("\n=== TASK D: Strategy 2 — Frozen ===", flush=True)
mF = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for p in mF.parameters(): p.requires_grad = False
mF.fc = nn.Linear(mF.fc.in_features, nc)
mF = mF.to(device)
rD_f = train_model(mF, optim.Adam(mF.fc.parameters(), lr=0.001), 3, "Frozen")

print("\n=== TASK D: Strategy 3 — Fine-tune ===", flush=True)
mFT = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for nm, p in mFT.named_parameters():
    if 'layer4' not in nm and 'fc' not in nm: p.requires_grad = False
mFT.fc = nn.Linear(mFT.fc.in_features, nc)
mFT = mFT.to(device)
rD_ft = train_model(mFT, optim.Adam([
    {'params': mFT.layer4.parameters(), 'lr': 0.0001},
    {'params': mFT.fc.parameters(), 'lr': 0.001}], lr=0.001), 3, "Fine-tune")

rD = {'scratch': rD_s, 'frozen': rD_f, 'finetune': rD_ft}

# === Save all results ===
all_r = {'task_a': rA, 'task_b': rB, 'task_c': rC, 'task_d': rD,
         'labels': labels, 'nc': nc, 'device': str(device)}
with open(str(RES_DIR / 'all_results.json'), 'w', encoding='utf-8') as f:
    json.dump(all_r, f, indent=2, ensure_ascii=False,
              default=lambda x: int(x) if isinstance(x, (np.integer,)) else float(x) if isinstance(x, (np.floating,)) else x)
print("\n[OK] Results saved", flush=True)

# ============================================================
# VISUALIZATIONS — Top Journal Quality
# ============================================================
print("\n=== Generating Visualizations ===", flush=True)

# Font/style
plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 12,
    'axes.linewidth': 1.0, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'legend.fontsize': 10, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
    'lines.linewidth': 1.8, 'lines.markersize': 7,
    'axes.spines.top': False, 'axes.spines.right': False,
})

# Palettes
NPG = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4']
LANCET = ['#00468B', '#ED0000', '#42B540', '#0099B4', '#925E9F']
JAMA = ['#374E55', '#DF8F44', '#00A1D5', '#B24745', '#79AF97']
SCI = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']
D3_COLORS = ['#2166AC', '#D6604D', '#4DAF4A']

def get_pal(name, n):
    pals = {'npg': NPG, 'lancet': LANCET, 'jama': JAMA, 'science': SCI, 'd3': D3_COLORS}
    c = pals.get(name, NPG)
    return (c * ((n//len(c))+1))[:n]

# --- Fig A ---
def fig_a(r):
    epochs = range(1, len(r['losses'])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    c = get_pal('npg', 2)
    ax1.plot(epochs, r['losses'], 'o-', c=c[0], lw=2.2, ms=7, mec='white', mew=0.8)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss — Feature Extraction', fontweight='bold', color='#2C3E50')
    ax1.grid(True, alpha=0.25, linestyle='--'); ax1.set_xticks(epochs)

    ax2.plot(epochs, r['accs'], 's-', c=c[1], lw=2.2, ms=7, mec='white', mew=0.8)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Validation Accuracy (%)')
    ax2.set_title('Validation Accuracy — Feature Extraction', fontweight='bold', color='#2C3E50')
    ax2.grid(True, alpha=0.25, linestyle='--'); ax2.set_xticks(epochs)
    best = max(r['accs']); bep = r['accs'].index(best)+1
    ax2.annotate(f'Best: {best:.1f}%', xy=(bep, best), xytext=(bep, best-3.5),
                ha='center', fontsize=9, fontweight='bold', color=c[1],
                bbox=dict(boxstyle='round,pad=0.3', fc='white', ec=c[1], alpha=0.85))
    fig.tight_layout(); return fig

fig = fig_a(rA)
fig.savefig(str(OUT_DIR / 'task_a_feature_extraction.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] task_a_feature_extraction.png", flush=True)

# --- Fig B ---
def fig_b(ra, rb):
    epochs = range(1, len(ra['losses'])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    c = get_pal('lancet', 2)
    ax1.plot(epochs, ra['losses'], 'o-', c=c[0], lw=2.2, ms=7, mec='white', mew=0.8, label='Feature Extraction')
    ax1.plot(epochs, rb['losses'], 's-', c=c[1], lw=2.2, ms=7, mec='white', mew=0.8, label='Fine-tuning')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss Comparison', fontweight='bold', color='#2C3E50')
    ax1.grid(True, alpha=0.25, linestyle='--'); ax1.set_xticks(epochs)
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)

    strategies = ['Feature\nExtraction', 'Fine-tuning']
    accs = [ra['best'], rb['best']]
    bars = ax2.bar(strategies, accs, color=[c[0], c[1]], edgecolor='white', lw=0.8, width=0.5)
    for bar, acc in zip(bars, accs):
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3, f'{acc:.2f}%',
                ha='center', fontsize=13, fontweight='bold', color='#34495E')
    ax2.set_ylim(min(accs)-5, max(accs)+8)
    ax2.set_ylabel('Best Validation Accuracy (%)')
    ax2.set_title('Accuracy: Feature Extraction vs Fine-tuning', fontweight='bold', color='#2C3E50')
    ax2.grid(True, alpha=0.25, linestyle='--', axis='y')
    fig.tight_layout(); return fig

fig = fig_b(rA, rB)
fig.savefig(str(OUT_DIR / 'task_b_comparison.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] task_b_comparison.png", flush=True)

# --- Fig C ---
def fig_c(rc):
    epochs = range(1, len(rc['resnet']['accs'])+1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    c = get_pal('jama', 2)

    # Pad MobileNetV2 data if only 3 epochs
    rn_accs = rc['resnet']['accs']
    mb_accs = rc['mobilenet']['accs']

    ax1.plot(epochs, rn_accs, 'o-', c=c[0], lw=2.2, ms=7, mec='white', mew=0.8,
             label=f"ResNet18 ({rc['resnet']['params']//1000000}M params)")
    ax1.plot(range(1, len(mb_accs)+1), mb_accs, 's-', c=c[1], lw=2.2, ms=7, mec='white', mew=0.8,
             label=f"MobileNetV2 ({rc['mobilenet']['params']//1000000}M params)")
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Validation Accuracy (%)')
    ax1.set_title('ResNet18 vs MobileNetV2: Accuracy Curves', fontweight='bold', color='#2C3E50')
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)

    categories = ['Best Acc\n(%)', 'Train Time\n(s)', 'Params\n(M)']
    rn_v = [rc['resnet']['best'], rc['resnet']['time'], rc['resnet']['params']/1e6]
    mb_v = [rc['mobilenet']['best'], rc['mobilenet']['time'], rc['mobilenet']['params']/1e6]
    x = np.arange(3); w = 0.35
    ax2.bar(x-w/2, rn_v, w, color=c[0], edgecolor='white', lw=0.8, label='ResNet18')
    ax2.bar(x+w/2, mb_v, w, color=c[1], edgecolor='white', lw=0.8, label='MobileNetV2')
    ax2.set_xticks(x); ax2.set_xticklabels(categories)
    ax2.set_title('Model Metrics Summary', fontweight='bold', color='#2C3E50')
    ax2.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle='--', axis='y')
    fig.tight_layout(); return fig

fig = fig_c(rC)
fig.savefig(str(OUT_DIR / 'task_c_model_comparison.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] task_c_model_comparison.png", flush=True)

# --- Fig D ---
def fig_d(rd):
    epochs = range(1, len(rd['scratch']['losses'])+1)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    c = D3_COLORS

    # Bar
    ax1 = axes[0]
    lbls = ['Scratch\nTraining', 'Feature\nExtraction', 'Fine-tuning']
    accs = [rd['scratch']['best'], rd['frozen']['best'], rd['finetune']['best']]
    times = [rd['scratch']['time'], rd['frozen']['time'], rd['finetune']['time']]
    bars = ax1.bar(range(3), accs, color=c, edgecolor='white', lw=0.8, width=0.55)
    for i, (bar, acc, t, clr) in enumerate(zip(bars, accs, times, c)):
        ax1.text(i, acc+1.5, f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold', color=clr)
        ax1.text(i, acc-6, f'({t:.0f}s)', ha='center', fontsize=8, color='#888888')
    ax1.set_xticks(range(3)); ax1.set_xticklabels(lbls, fontsize=10)
    ax1.set_ylim(min(accs)-15, max(accs)+8)
    ax1.set_ylabel('Best Validation Accuracy (%)')
    ax1.set_title('Strategy Accuracy Comparison', fontweight='bold', color='#2C3E50', loc='left')
    ax1.grid(True, alpha=0.25, linestyle='--', axis='y')

    # Loss
    ax2 = axes[1]
    ax2.plot(epochs, rd['scratch']['losses'], 'o-', c=c[0], lw=2.2, ms=6, mec='white', mew=0.8, label='Scratch')
    ax2.plot(epochs, rd['frozen']['losses'], 's-', c=c[1], lw=2.2, ms=6, mec='white', mew=0.8, label='Feature Extraction')
    ax2.plot(epochs, rd['finetune']['losses'], 'D-', c=c[2], lw=2.2, ms=6, mec='white', mew=0.8, label='Fine-tuning')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Cross-Entropy Loss')
    ax2.set_title('Training Loss Curves', fontweight='bold', color='#2C3E50', loc='left')
    ax2.grid(True, alpha=0.25, linestyle='--'); ax2.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    # Accuracy
    ax3 = axes[2]
    ax3.plot(epochs, rd['scratch']['accs'], 'o-', c=c[0], lw=2.2, ms=6, mec='white', mew=0.8, label='Scratch')
    ax3.plot(epochs, rd['frozen']['accs'], 's-', c=c[1], lw=2.2, ms=6, mec='white', mew=0.8, label='Feature Extraction')
    ax3.plot(epochs, rd['finetune']['accs'], 'D-', c=c[2], lw=2.2, ms=6, mec='white', mew=0.8, label='Fine-tuning')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Validation Accuracy (%)')
    ax3.set_title('Accuracy Convergence', fontweight='bold', color='#2C3E50', loc='left')
    ax3.grid(True, alpha=0.25, linestyle='--'); ax3.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    fig.tight_layout(); return fig

fig = fig_d(rD)
fig.savefig(str(OUT_DIR / 'task_d_final_comparison.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] task_d_final_comparison.png", flush=True)

# --- Summary Dashboard ---
def fig_dashboard(ra, rb, rc, rd):
    c_npg = get_pal('npg', 4); c_lnc = get_pal('lancet', 4); c_jam = get_pal('jama', 4)
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.38)

    ep = range(1, 6)  # Task A/B have 5 epochs
    ep_mb = range(1, 4)  # MobileNetV2 has 3 epochs
    ep_d = range(1, 4)  # Task D has 3 epochs

    # A1
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(ep, ra['losses'], 'o-', c=c_npg[0], lw=2, ms=6, mec='white', mew=0.6)
    ax.set_title('A: Feature Extraction Loss', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.2, linestyle='--')

    # A2
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(ep, ra['accs'], 's-', c=c_npg[1], lw=2, ms=6, mec='white', mew=0.6)
    ax.set_title('A: Validation Accuracy', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Acc (%)'); ax.grid(True, alpha=0.2, linestyle='--')

    # B1
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ep, ra['losses'], 'o-', c=c_lnc[0], lw=1.5, ms=5, alpha=0.7, label='Feat. Ext.')
    ax.plot(ep, rb['losses'], 's-', c=c_lnc[1], lw=2, ms=6, mec='white', mew=0.6, label='Fine-tuning')
    ax.set_title('B: Loss Comparison', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.2, linestyle='--'); ax.legend(fontsize=7)

    # B2
    ax = fig.add_subplot(gs[0, 3])
    bars = ax.bar(['Feature\nExtraction', 'Fine-tuning'], [ra['best'], rb['best']],
                  color=[c_lnc[0], c_lnc[1]], edgecolor='white', lw=0.8)
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.3, f"{b.get_height():.1f}%",
                ha='center', fontsize=10, fontweight='bold')
    ax.set_ylim(min(ra['best'], rb['best'])-3, max(ra['best'], rb['best'])+5)
    ax.set_title('B: Best Accuracy', fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    # C1
    ax = fig.add_subplot(gs[1, 0:2])
    ax.plot(ep, rc['resnet']['accs'], 'o-', c=c_jam[0], lw=2, ms=7, mec='white', mew=0.8, label='ResNet18')
    ax.plot(ep_mb, rc['mobilenet']['accs'], 's-', c=c_jam[1], lw=2, ms=7, mec='white', mew=0.8, label='MobileNetV2')
    ax.set_title('C: ResNet18 vs MobileNetV2', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val Acc (%)'); ax.grid(True, alpha=0.2, linestyle='--'); ax.legend(fontsize=9)

    # C2
    ax = fig.add_subplot(gs[1, 2:4])
    x = np.arange(3); w = 0.35
    ax.bar(x-w/2, [rc['resnet']['best'], rc['resnet']['time'], rc['resnet']['params']/1e6], w, color=c_jam[0], edgecolor='white', label='ResNet18')
    ax.bar(x+w/2, [rc['mobilenet']['best'], rc['mobilenet']['time'], rc['mobilenet']['params']/1e6], w, color=c_jam[1], edgecolor='white', label='MobileNetV2')
    ax.set_xticks(x); ax.set_xticklabels(['Accuracy (%)', 'Time (s)', 'Params (M)'], fontsize=9)
    ax.set_title('C: Metrics Comparison', fontsize=10, fontweight='bold'); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle='--', axis='y')

    # D1
    d_colors = D3_COLORS
    ax = fig.add_subplot(gs[2, 0:2])
    d_lbls = ['Scratch Training', 'Feature Extraction', 'Fine-tuning']
    d_accs = [rd['scratch']['best'], rd['frozen']['best'], rd['finetune']['best']]
    bars = ax.bar(d_lbls, d_accs, color=d_colors, edgecolor='white', width=0.5)
    for b, a, c in zip(bars, d_accs, d_colors):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5, f'{a:.2f}%',
                ha='center', fontsize=11, fontweight='bold', color=c)
    ax.set_ylim(min(d_accs)-15, max(d_accs)+8)
    ax.set_title('D: Three Strategies Accuracy', fontsize=10, fontweight='bold')
    ax.set_ylabel('Best Val Acc (%)'); ax.grid(True, alpha=0.2, linestyle='--', axis='y')
    ax.set_xticklabels(d_lbls, rotation=10, ha='right', fontsize=9)

    # D2
    ax = fig.add_subplot(gs[2, 2:4])
    ax.plot(ep_d, rd['scratch']['losses'], 'o-', c=d_colors[0], lw=2, ms=6, mec='white', mew=0.8, label='Scratch')
    ax.plot(ep_d, rd['frozen']['losses'], 's-', c=d_colors[1], lw=2, ms=6, mec='white', mew=0.8, label='Feature Extraction')
    ax.plot(ep_d, rd['finetune']['losses'], 'D-', c=d_colors[2], lw=2, ms=6, mec='white', mew=0.8, label='Fine-tuning')
    ax.set_title('D: Loss Convergence', fontsize=10, fontweight='bold')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.grid(True, alpha=0.2, linestyle='--'); ax.legend(fontsize=9)

    fig.suptitle('Transfer Learning for Agricultural Image Recognition — Comprehensive Dashboard',
                 fontsize=15, fontweight='bold', color='#2C3E50', y=1.01)
    fig.tight_layout(); return fig

fig = fig_dashboard(rA, rB, rC, rD)
fig.savefig(str(OUT_DIR / 'summary_dashboard.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] summary_dashboard.png", flush=True)

# --- Abstract Figure ---
def fig_abstract(rd):
    c = D3_COLORS
    ep = range(1, len(rd['scratch']['losses'])+1)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    lbls = ['Scratch\nTraining', 'Feature\nExtraction', 'Fine-tuning']
    accs = [rd['scratch']['best'], rd['frozen']['best'], rd['finetune']['best']]
    times = [rd['scratch']['time'], rd['frozen']['time'], rd['finetune']['time']]
    bars = ax1.bar(range(3), accs, color=c, edgecolor='#333333', lw=0.5, width=0.55)
    for i, (bar, acc, t, clr) in enumerate(zip(bars, accs, times, c)):
        ax1.text(i, acc+1.5, f'{acc:.2f}%', ha='center', fontsize=12, fontweight='bold', color=clr)
        ax1.text(i, acc-5, f'{t:.0f}s', ha='center', fontsize=8, color='#666666')
    ax1.set_xticks(range(3)); ax1.set_xticklabels(lbls, fontsize=10)
    ax1.set_ylim(min(accs)-12, max(accs)+10)
    ax1.set_ylabel('Best Validation Accuracy (%)', fontsize=11)
    ax1.set_title('a) Strategy Comparison', fontsize=12, fontweight='bold', color='#2C3E50', loc='left')
    ax1.grid(True, alpha=0.2, linestyle='--', axis='y')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, rd['scratch']['losses'], 'o-', c=c[0], lw=2.2, ms=7, mec='white', mew=0.8, label='Scratch')
    ax2.plot(ep, rd['frozen']['losses'], 's-', c=c[1], lw=2.2, ms=7, mec='white', mew=0.8, label='Feature Extraction')
    ax2.plot(ep, rd['finetune']['losses'], 'D-', c=c[2], lw=2.2, ms=7, mec='white', mew=0.8, label='Fine-tuning')
    ax2.set_xlabel('Epoch', fontsize=11); ax2.set_ylabel('Cross-Entropy Loss', fontsize=11)
    ax2.set_title('b) Training Loss Curves', fontsize=12, fontweight='bold', color='#2C3E50', loc='left')
    ax2.grid(True, alpha=0.2, linestyle='--'); ax2.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(ep, rd['scratch']['accs'], 'o-', c=c[0], lw=2.2, ms=7, mec='white', mew=0.8, label='Scratch')
    ax3.plot(ep, rd['frozen']['accs'], 's-', c=c[1], lw=2.2, ms=7, mec='white', mew=0.8, label='Feature Extraction')
    ax3.plot(ep, rd['finetune']['accs'], 'D-', c=c[2], lw=2.2, ms=7, mec='white', mew=0.8, label='Fine-tuning')
    ax3.set_xlabel('Epoch', fontsize=11); ax3.set_ylabel('Validation Accuracy (%)', fontsize=11)
    ax3.set_title('c) Accuracy Convergence', fontsize=12, fontweight='bold', color='#2C3E50', loc='left')
    ax3.grid(True, alpha=0.2, linestyle='--'); ax3.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    ax4 = fig.add_subplot(gs[1, 1])
    for i, (s, a, t, clr) in enumerate(zip(['Scratch', 'Feature Ext.', 'Fine-tuning'],
                                            [rd['scratch']['best'], rd['frozen']['best'], rd['finetune']['best']],
                                            [rd['scratch']['time'], rd['frozen']['time'], rd['finetune']['time']], c)):
        ax4.scatter(t, a, s=250, c=clr, edgecolors='white', lw=1.5, zorder=5)
        ax4.annotate(f'  {s}\n  {a:.1f}%, {t:.0f}s', (t, a), fontsize=9, fontweight='bold', color=clr, va='center')
    ax4.set_xlabel('Training Time (seconds)', fontsize=11)
    ax4.set_ylabel('Best Accuracy (%)', fontsize=11)
    ax4.set_title('d) Efficiency Analysis', fontsize=12, fontweight='bold', color='#2C3E50', loc='left')
    ax4.grid(True, alpha=0.2, linestyle='--')

    fig.suptitle('Transfer Learning for Tomato Leaf Disease Classification\n'
                 'Benchmark of Training Strategies on PlantVillage Dataset',
                 fontsize=15, fontweight='bold', color='#2C3E50', y=1.01)
    fig.tight_layout(); return fig

fig = fig_abstract(rD)
fig.savefig(str(OUT_DIR / 'abstract_figure.png'), dpi=300, facecolor='white')
plt.close(fig)
print("  [OK] abstract_figure.png", flush=True)

# === Summary ===
print("\n" + "="*60, flush=True)
print("EXPERIMENT RESULTS SUMMARY", flush=True)
print("="*60, flush=True)
print(f"""
  Task A — Feature Extraction:  {rA['best']:.2f}% ({rA['time']:.0f}s)
  Task B — Fine-tuning:          {rB['best']:.2f}% ({rB['time']:.0f}s)
  Task C — ResNet18:             {rC['resnet']['best']:.2f}% ({rC['resnet']['params']/1e6:.1f}M, {rC['resnet']['time']:.0f}s)
  Task C — MobileNetV2:          {rC['mobilenet']['best']:.2f}% ({rC['mobilenet']['params']/1e6:.1f}M, {rC['mobilenet']['time']:.0f}s)
  Task D — Scratch:              {rD['scratch']['best']:.2f}% ({rD['scratch']['time']:.0f}s)
  Task D — Frozen:               {rD['frozen']['best']:.2f}% ({rD['frozen']['time']:.0f}s)
  Task D — Fine-tune:            {rD['finetune']['best']:.2f}% ({rD['finetune']['time']:.0f}s)
""", flush=True)
print(f"Figures saved to: {OUT_DIR}", flush=True)
print("DONE!", flush=True)
