# ============================================================
# Frequency-Restricted Adversarial Attack Analysis
# Question: Are low-frequency perturbations more dangerous than high-frequency noise?
# ============================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.fft as fft
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# ============================================================
# Dataset
# ============================================================
class CTSliceDataset(Dataset):
    def __init__(self, root_dir, img_size=(256,256)):
        self.samples, self.labels = [], []
        self.img_size = img_size
        for label, cls in enumerate(["benign", "malignant"]):
            d = os.path.join(root_dir, cls)
            if not os.path.exists(d):
                continue
            for f in os.listdir(d):
                if f.lower().endswith((".png",".jpg",".tif",".bmp")):
                    self.samples.append(os.path.join(d,f))
                    self.labels.append(label)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        img = io.imread(self.samples[idx], as_gray=True).astype(np.float32)
        img = transform.resize(img, self.img_size, mode="reflect", anti_aliasing=True)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return torch.tensor(img).unsqueeze(0), torch.tensor(self.labels[idx])

# ============================================================
# Radon Transform
# ============================================================
class Radon(nn.Module):
    def __init__(self, thetas, img_size):
        super().__init__()
        self.thetas = thetas
        self.img_size = img_size
        affines = []
        for t in thetas:
            t_f = float(t.item())
            affines.append(
                torch.tensor([[[np.cos(t_f), -np.sin(t_f), 0],
                                [np.sin(t_f),  np.cos(t_f), 0]]], dtype=torch.float32)
            )
        self.affines = torch.cat(affines, dim=0).to(device)

    def forward(self, x):
        B = x.shape[0]
        sino = []
        for i, A in enumerate(self.affines):
            grid = F.affine_grid(A.unsqueeze(0).repeat(B,1,1), x.size(), align_corners=False)
            rot = F.grid_sample(x, grid, align_corners=False)
            sino.append(rot.sum(dim=2))
        return torch.stack(sino, dim=-1)

# ============================================================
# Filtered Backprojection (FBP)
# ============================================================
def fbp_reconstruct(sino, thetas):
    B,_,D,A = sino.shape
    device = sino.device
    # Ramp filter
    ramp = torch.abs(fft.fftfreq(D, d=1.0).to(device)).view(1,1,-1,1)
    sino_f = fft.ifft(fft.fft(sino,dim=2)*ramp,dim=2).real

    # Precompute rotation matrices
    rot_mats = []
    for t in thetas:
        t_f = float(t.item())
        mat = torch.tensor([[[np.cos(t_f),-np.sin(t_f),0],
                             [np.sin(t_f), np.cos(t_f),0]]],dtype=torch.float32, device=device)
        rot_mats.append(mat)
    rot_mats = torch.stack(rot_mats, dim=0)

    # Backprojection
    recon = torch.zeros(B,1,D,D,device=device)
    for i in range(A):
        proj = sino_f[:,:,:,i].unsqueeze(-1).repeat(1,1,1,D)
        grid = F.affine_grid(rot_mats[i].repeat(B,1,1), proj.size(), align_corners=False)
        recon += F.grid_sample(proj, grid, align_corners=False)
    return recon / A

# ============================================================
# Frequency-Restricted Attacks
# ============================================================
def create_frequency_mask(D, cutoff_ratio=0.3, mask_type='low'):
    """Create frequency mask for low-pass or high-pass filtering"""
    freqs = torch.fft.fftfreq(D)
    cutoff_freq = cutoff_ratio * 0.5
    
    if mask_type == 'low':
        mask = (torch.abs(freqs) <= cutoff_freq).float()
    elif mask_type == 'high':
        mask = (torch.abs(freqs) > cutoff_freq).float()
    else:
        raise ValueError("mask_type must be 'low' or 'high'")
    
    return mask.to(device).view(1, 1, -1, 1)

def sino_pgd_attack(sino, model, lbls, eps, alpha=None, steps=10, mask_type=None, cutoff_ratio=0.3, thetas=None):
    """PGD attack with optional frequency restriction (stronger than FGSM)"""
    if alpha is None:
        alpha = eps / 5
    
    if thetas is None:
        thetas = torch.linspace(0, np.pi, 60, device=device)
    
    ce = nn.CrossEntropyLoss()
    sino_orig = sino.clone().detach()
    sino_adv = sino.clone().detach()
    
    for step in range(steps):
        sino_adv.requires_grad = True
        
        recon = fbp_reconstruct(sino_adv, thetas)
        out = model(recon)
        loss = ce(out, lbls)
        
        loss.backward()
        grad = sino_adv.grad.detach()
        
        # Apply frequency mask if specified
        if mask_type is not None:
            G = fft.fft(grad, dim=2)
            D = grad.shape[2]
            mask = create_frequency_mask(D, cutoff_ratio, mask_type)
            G_filtered = G * mask
            grad = fft.ifft(G_filtered, dim=2).real
        
        with torch.no_grad():
            sino_adv = sino_adv + alpha * grad.sign()
            delta = torch.clamp(sino_adv - sino_orig, -eps, eps)
            sino_adv = (sino_orig + delta).detach()
    
    return sino_adv

# ============================================================
# CNN Classifier
# ============================================================
class SimpleCTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64,2)
        )
    def forward(self,x): return self.net(x)

# ============================================================
# Training Function
# ============================================================
def train(model, train_loader, val_loader, radon, thetas, epochs=20):
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), 1e-3)

    for ep in range(1, epochs+1):
        model.train()
        total = correct = 0
        loop = tqdm(train_loader, desc=f"Epoch {ep}/{epochs}")
        
        for imgs, lbls in loop:
            imgs, lbls = imgs.to(device), lbls.to(device)
            with torch.no_grad():
                sino = radon(imgs)
                recon = fbp_reconstruct(sino, thetas)
            out = model(recon)
            loss = ce(out, lbls)
            opt.zero_grad()
            loss.backward()
            opt.step()

            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
            loop.set_postfix(Acc=f"{correct/total:.3f}")

        # Validation
        model.eval()
        val_total = val_correct = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                recon = fbp_reconstruct(radon(imgs), thetas)
                out = model(recon)
                val_correct += (out.argmax(1) == lbls).sum().item()
                val_total += lbls.size(0)
        
        print(f"Epoch {ep}: Train Acc={correct/total:.3f} | Val Acc={val_correct/val_total:.3f}")

# ============================================================
# Evaluation Function
# ============================================================
def evaluate_attacks(model, loader, radon, thetas, eps=0.2, cutoff_ratio=0.3):
    """
    Evaluate three attack types with strong PGD:
    1. Standard (all frequencies)
    2. Low-frequency only
    3. High-frequency only
    """
    ce = nn.CrossEntropyLoss()
    model.eval()

    results = {
        'clean': {'correct': 0, 'total': 0},
        'standard': {'correct': 0, 'total': 0, 'flips': 0},
        'low_freq': {'correct': 0, 'total': 0, 'flips': 0},
        'high_freq': {'correct': 0, 'total': 0, 'flips': 0}
    }

    sample_data = None

    for batch_idx, (imgs, lbls) in enumerate(tqdm(loader, desc="Evaluating attacks")):
        imgs, lbls = imgs.to(device), lbls.to(device)

        # Clean evaluation
        with torch.no_grad():
            sino_clean = radon(imgs)
            recon_clean = fbp_reconstruct(sino_clean, thetas)
            out_clean = model(recon_clean)
            pred_clean = out_clean.argmax(1)
            results['clean']['correct'] += (pred_clean == lbls).sum().item()

        # Standard PGD attack
        sino = radon(imgs).detach()
        adv_sino_standard = sino_pgd_attack(sino, model, lbls, eps, steps=15, thetas=thetas)

        # Low-frequency PGD attack
        sino = radon(imgs).detach()
        adv_sino_low = sino_pgd_attack(sino, model, lbls, eps, steps=15, mask_type='low', cutoff_ratio=cutoff_ratio, thetas=thetas)

        # High-frequency PGD attack
        sino = radon(imgs).detach()
        adv_sino_high = sino_pgd_attack(sino, model, lbls, eps, steps=15, mask_type='high', cutoff_ratio=cutoff_ratio, thetas=thetas)

        # Evaluate all attacks
        with torch.no_grad():
            # Standard
            recon_std = fbp_reconstruct(adv_sino_standard, thetas)
            out_std = model(recon_std)
            pred_std = out_std.argmax(1)
            results['standard']['correct'] += (pred_std == lbls).sum().item()
            results['standard']['flips'] += (pred_clean != pred_std).sum().item()

            # Low-frequency
            recon_low = fbp_reconstruct(adv_sino_low, thetas)
            out_low = model(recon_low)
            pred_low = out_low.argmax(1)
            results['low_freq']['correct'] += (pred_low == lbls).sum().item()
            results['low_freq']['flips'] += (pred_clean != pred_low).sum().item()

            # High-frequency
            recon_high = fbp_reconstruct(adv_sino_high, thetas)
            out_high = model(recon_high)
            pred_high = out_high.argmax(1)
            results['high_freq']['correct'] += (pred_high == lbls).sum().item()
            results['high_freq']['flips'] += (pred_clean != pred_high).sum().item()

        # Store first batch for visualization
        if batch_idx == 0 and sample_data is None:
            sample_data = {
                'recon_clean': recon_clean[0].detach().cpu(),
                'recon_std': recon_std[0].detach().cpu(),
                'recon_low': recon_low[0].detach().cpu(),
                'recon_high': recon_high[0].detach().cpu(),
            }

        # Update totals
        for key in results:
            results[key]['total'] += lbls.size(0)

    # Calculate metrics
    metrics = {}
    metrics['clean_acc'] = results['clean']['correct'] / results['clean']['total']

    for attack in ['standard', 'low_freq', 'high_freq']:
        total = results[attack]['total']
        metrics[f'{attack}_acc'] = results[attack]['correct'] / total
        metrics[f'{attack}_flip'] = results[attack]['flips'] / total
        metrics[f'{attack}_drop'] = metrics['clean_acc'] - metrics[f'{attack}_acc']

    return metrics, sample_data

# ============================================================
# Compact Research-Grade Visualizations
# ============================================================
def visualize_comprehensive_comparison(metrics, sample_data):
    """Compact, research-grade visualization"""
    
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
    
    colors = {
        'clean': '#27AE60',
        'standard': '#2E86AB',
        'low_freq': '#A23B72',
        'high_freq': '#F18F01'
    }
    bg_color = '#FAFBFC'
    
    # ============================================================
    # Panel A: Accuracy Comparison
    # ============================================================
    ax1 = fig.add_subplot(gs[0, 0])
    attack_types = ['Clean', 'Standard', 'Low-Freq', 'High-Freq']
    accs = [metrics['clean_acc'], metrics['standard_acc'], metrics['low_freq_acc'], metrics['high_freq_acc']]
    color_list = [colors['clean'], colors['standard'], colors['low_freq'], colors['high_freq']]
    
    bars = ax1.bar(range(len(attack_types)), accs, color=color_list, alpha=0.85, edgecolor='#333333', linewidth=0.8)
    ax1.set_ylabel('Accuracy', fontsize=10, fontweight='600')
    ax1.set_title('A. Model Accuracy', fontsize=11, fontweight='700')
    ax1.set_xticks(range(len(attack_types)))
    ax1.set_xticklabels(attack_types, fontsize=9, rotation=45, ha='right')
    ax1.set_ylim([0, 1.05])
    ax1.set_facecolor(bg_color)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_axisbelow(True)
    
    for bar, acc in zip(bars, accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{acc:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='600')
    
    # ============================================================
    # Panel B: Accuracy Drop Comparison
    # ============================================================
    ax2 = fig.add_subplot(gs[0, 1])
    drops = [metrics['standard_drop'], metrics['low_freq_drop'], metrics['high_freq_drop']]
    attack_labels = ['Standard', 'Low-Freq', 'High-Freq']
    color_list2 = [colors['standard'], colors['low_freq'], colors['high_freq']]
    
    bars = ax2.bar(range(len(attack_labels)), drops, color=color_list2, alpha=0.85, edgecolor='#333333', linewidth=0.8)
    ax2.set_ylabel('Accuracy Drop', fontsize=10, fontweight='600')
    ax2.set_title('B. Attack Effectiveness', fontsize=11, fontweight='700')
    ax2.set_xticks(range(len(attack_labels)))
    ax2.set_xticklabels(attack_labels, fontsize=9, rotation=45, ha='right')
    ax2.set_facecolor(bg_color)
    ax2.grid(True, alpha=0.2, axis='y')
    ax2.set_axisbelow(True)
    
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01, f'{drop:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='600')
    
    # ============================================================
    # Panel C: Flip Rate Comparison
    # ============================================================
    ax3 = fig.add_subplot(gs[0, 2])
    flips = [metrics['standard_flip'], metrics['low_freq_flip'], metrics['high_freq_flip']]
    
    bars = ax3.bar(range(len(attack_labels)), flips, color=color_list2, alpha=0.85, edgecolor='#333333', linewidth=0.8)
    ax3.set_ylabel('Flip Rate', fontsize=10, fontweight='600')
    ax3.set_title('C. Prediction Flip Rate', fontsize=11, fontweight='700')
    ax3.set_xticks(range(len(attack_labels)))
    ax3.set_xticklabels(attack_labels, fontsize=9, rotation=45, ha='right')
    ax3.set_ylim([0, 1.05])
    ax3.set_facecolor(bg_color)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_axisbelow(True)
    
    for bar, flip in zip(bars, flips):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02, f'{flip:.3f}', 
                ha='center', va='bottom', fontsize=8, fontweight='600')
    
    # ============================================================
    # Panel D-G: Reconstructed Images
    # ============================================================
    recon_data = [
        ('Clean', sample_data['recon_clean']),
        ('Standard', sample_data['recon_std']),
        ('Low-Freq', sample_data['recon_low']),
        ('High-Freq', sample_data['recon_high'])
    ]
    
    for idx, (title, img) in enumerate(recon_data):
        ax = fig.add_subplot(gs[1, idx % 3])
        im = ax.imshow(img[0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(title, fontsize=10, fontweight='700')
        ax.axis('off')
    
    fig.patch.set_facecolor('white')
    fig.suptitle('Frequency-Restricted Adversarial Attacks Analysis', fontsize=13, fontweight='700', y=0.98)
    
    return fig

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("FREQUENCY-RESTRICTED ADVERSARIAL ATTACK ANALYSIS")
    print("="*80)
    print("\nQuestion: Are low-frequency perturbations more dangerous than high-frequency?")
    print("Hypothesis: Low-frequency attacks survive FBP, high-frequency attacks are filtered\n")

    # Configuration
    ROOT_DIR = "path1"  # Update this path
    EPOCHS = 20
    EPSILON = 0.25
    CUTOFF_RATIO = 0.3

    # Load data
    print("Loading datasets...")
    train_ds = CTSliceDataset(os.path.join(ROOT_DIR, "train"))
    val_ds = CTSliceDataset(os.path.join(ROOT_DIR, "valid"))

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"\n✗ No data found in {ROOT_DIR}")
        print("Please update ROOT_DIR to point to your dataset")
        print("Expected structure: ROOT_DIR/train/benign, ROOT_DIR/train/malignant, etc.")
        exit(1)

    print(f"✓ Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # Setup
    thetas = torch.linspace(0, np.pi, 60, device=device)
    radon = Radon(thetas, img_size=train_ds.img_size[0]).to(device)
    model = SimpleCTCNN().to(device)

    # Train
    print("\nTraining model...")
    train(model, train_loader, val_loader, radon, thetas, epochs=EPOCHS)

    # Get clean accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            sino = radon(imgs)
            recon = fbp_reconstruct(sino, thetas)
            out = model(recon)
            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)
    clean_acc = correct / total
    print(f"Clean Accuracy: {clean_acc:.4f}")

    # Evaluate attacks
    print(f"\nEvaluating frequency-restricted attacks (ε={EPSILON}, cutoff={CUTOFF_RATIO})...")
    metrics, sample_data = evaluate_attacks(model, val_loader, radon, thetas, eps=EPSILON, cutoff_ratio=CUTOFF_RATIO)

    # Display results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"\nClean Accuracy:          {metrics['clean_acc']:.4f}")
    print(f"\nStandard Attack (PGD):")
    print(f"  Accuracy:              {metrics['standard_acc']:.4f}")
    print(f"  Accuracy Drop:         {metrics['standard_drop']:.4f}")
    print(f"  Flip Rate:             {metrics['standard_flip']:.4f}")
    print(f"\nLow-Frequency Attack:")
    print(f"  Accuracy:              {metrics['low_freq_acc']:.4f}")
    print(f"  Accuracy Drop:         {metrics['low_freq_drop']:.4f}")
    print(f"  Flip Rate:             {metrics['low_freq_flip']:.4f}")
    print(f"\nHigh-Frequency Attack:")
    print(f"  Accuracy:              {metrics['high_freq_acc']:.4f}")
    print(f"  Accuracy Drop:         {metrics['high_freq_drop']:.4f}")
    print(f"  Flip Rate:             {metrics['high_freq_flip']:.4f}")

    # Analysis
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if metrics['low_freq_drop'] > metrics['high_freq_drop']:
        ratio = metrics['low_freq_drop'] / max(metrics['high_freq_drop'], 1e-6)
        print(f"\n✓ Low-frequency perturbations are {ratio:.2f}× MORE DANGEROUS")
        print("  → Low-freq perturbations survive FBP reconstruction")
        print("  → High-freq perturbations are filtered by ramp filter")
    else:
        print(f"\n✗ Unexpected result: High-frequency more effective")
        ratio = metrics['high_freq_drop'] / max(metrics['low_freq_drop'], 1e-6)
        print(f"  High-freq is {ratio:.2f}× more dangerous")

    print("\n" + "="*80)

    # Visualizations
    print("\nGenerating research-grade visualization...")

    fig = visualize_comprehensive_comparison(metrics, sample_data)
    plt.show()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)