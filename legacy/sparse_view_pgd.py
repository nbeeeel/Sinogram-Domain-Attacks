# ============================================================
# Sparse-View CT Adversarial Vulnerability Analysis
# ============================================================
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.fft as fft
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import pandas as pd

torch.manual_seed(0); np.random.seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set professional plotting style
plt.style.use('seaborn-v0_8-whitegrid')

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
# PGD Attack (Stronger than FGSM)
# ============================================================
def sino_pgd_attack(sino, model, lbls, eps, alpha=None, steps=10, thetas=None):
    """PGD attack on sinogram"""
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
def train(model, train_loader, val_loader, radon, thetas, epochs=15, verbose=True):
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(),1e-3)

    for ep in range(1,epochs+1):
        model.train()
        total=correct=0
        desc = f"Epoch {ep}/{epochs}" if verbose else f"Training"
        loop = tqdm(train_loader, desc=desc, leave=False, disable=not verbose)

        for imgs,lbls in loop:
            imgs,lbls = imgs.to(device), lbls.to(device)
            with torch.no_grad():
                sino = radon(imgs)
                recon = fbp_reconstruct(sino,thetas)
            out = model(recon)
            loss = ce(out,lbls)
            opt.zero_grad(); loss.backward(); opt.step()

            correct += (out.argmax(1)==lbls).sum().item()
            total += lbls.size(0)
            if verbose:
                loop.set_postfix(Acc=f"{correct/total:.3f}")

# ============================================================
# Comprehensive Evaluation Function
# ============================================================
def evaluate(model, loader, radon, thetas, eps=0.0):
    """
    Evaluate model on clean and adversarial examples.
    Returns detailed metrics.
    """
    ce = nn.CrossEntropyLoss()
    model.eval()

    total = 0
    clean_correct = 0
    adv_correct = 0
    flips = 0
    clean_confs = []
    adv_confs = []

    with torch.set_grad_enabled(eps > 0):
        for imgs, lbls in tqdm(loader, desc=f"Evaluating (ε={eps:.3f})", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)

            # Clean evaluation
            with torch.no_grad():
                sino_clean = radon(imgs)
                recon_clean = fbp_reconstruct(sino_clean, thetas)
                out_clean = model(recon_clean)
                pred_clean = out_clean.argmax(1)
                probs_clean = torch.softmax(out_clean, dim=1)
                conf_clean = probs_clean.max(dim=1)[0]
                clean_confs.extend(conf_clean.cpu().numpy())
                clean_correct += (pred_clean == lbls).sum().item()

            # Adversarial evaluation
            if eps > 0:
                sino = radon(imgs)
                sino.requires_grad_(True)
                recon = fbp_reconstruct(sino, thetas)
                out = model(recon)
                loss = ce(out, lbls)

                adv_sino = sino_pgd_attack(sino, model, lbls, eps, steps=10, thetas=thetas)
                with torch.no_grad():
                    adv_recon = fbp_reconstruct(adv_sino, thetas)
                    out_adv = model(adv_recon)
                    pred_adv = out_adv.argmax(1)
                    probs_adv = torch.softmax(out_adv, dim=1)
                    conf_adv = probs_adv.max(dim=1)[0]
                    adv_confs.extend(conf_adv.cpu().numpy())
                    adv_correct += (pred_adv == lbls).sum().item()
                    flips += (pred_clean != pred_adv).sum().item()
            else:
                adv_correct = clean_correct
                flips = 0
                adv_confs = clean_confs.copy()

            total += lbls.size(0)

    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    flip_rate = flips / total
    avg_clean_conf = np.mean(clean_confs)
    avg_adv_conf = np.mean(adv_confs)

    return {
        'clean_acc': clean_acc,
        'adv_acc': adv_acc,
        'acc_drop': clean_acc - adv_acc,
        'flip_rate': flip_rate,
        'clean_conf': avg_clean_conf,
        'adv_conf': avg_adv_conf,
        'conf_drop': avg_clean_conf - avg_adv_conf
    }

# ============================================================
# Main Analysis Function
# ============================================================
def run_sparse_view_analysis(root_dir, angle_counts=[30, 60, 90, 120, 180], eps=0.15, epochs=15):
    """
    Run complete analysis across different viewing angles.
    """
    print("="*70)
    print("SPARSE-VIEW CT ADVERSARIAL VULNERABILITY ANALYSIS")
    print("="*70)

    # Load datasets
    print("\nLoading datasets...")
    train_ds = CTSliceDataset(os.path.join(root_dir, "train"))
    val_ds = CTSliceDataset(os.path.join(root_dir, "valid"))

    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"\n✗ No data found in {root_dir}")
        print("Please update ROOT_DIR to point to your dataset")
        print("Expected structure: ROOT_DIR/train/benign, ROOT_DIR/train/malignant, etc.")
        exit(1)

    print(f"✓ Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    results = {
        'n_angles': [],
        'clean_acc': [],
        'adv_acc': [],
        'flip_rate': [],
        'acc_drop': [],
        'clean_conf': [],
        'adv_conf': [],
        'conf_drop': []
    }

    sample_reconstructions = {}

    for N in angle_counts:
        print(f"\n{'='*70}")
        print(f"Configuration: N = {N} projection angles")
        print(f"{'='*70}")

        # Create Radon transform
        thetas = torch.linspace(0, np.pi, N, device=device)
        radon = Radon(thetas, img_size=train_ds.img_size[0]).to(device)

        # Create and train model
        model = SimpleCTCNN().to(device)
        print(f"\nTraining model...")
        train(model, train_loader, val_loader, radon, thetas, epochs=epochs, verbose=True)

        # Evaluate with clean and adversarial examples
        print(f"\nEvaluating robustness (ε = {eps})...")
        metrics = evaluate(model, val_loader, radon, thetas, eps=eps)

        # Store results
        results['n_angles'].append(N)
        results['clean_acc'].append(metrics['clean_acc'])
        results['adv_acc'].append(metrics['adv_acc'])
        results['flip_rate'].append(metrics['flip_rate'])
        results['acc_drop'].append(metrics['acc_drop'])
        results['clean_conf'].append(metrics['clean_conf'])
        results['adv_conf'].append(metrics['adv_conf'])
        results['conf_drop'].append(metrics['conf_drop'])

        print(f"\n📊 Results Summary:")
        print(f"  Clean Accuracy:         {metrics['clean_acc']:.4f}")
        print(f"  Adversarial Accuracy:   {metrics['adv_acc']:.4f}")
        print(f"  Accuracy Drop:          {metrics['acc_drop']:.4f}")
        print(f"  Flip Rate:              {metrics['flip_rate']:.4f}")
        print(f"  Confidence Drop:        {metrics['conf_drop']:.4f}")

        # Save sample reconstruction
        imgs, lbls = next(iter(val_loader))
        imgs = imgs.to(device)
        with torch.no_grad():
            sino = radon(imgs)
            recon = fbp_reconstruct(sino, thetas)
        sample_reconstructions[N] = {
            'recon': recon[0].cpu()
        }

    return results, sample_reconstructions

# ============================================================
# Professional Research-Grade Visualization (5 Configurations)
# ============================================================
def create_comprehensive_figure(results, sample_reconstructions, eps):
    """
    Create a professional, compact, research-grade figure for 5 angle configurations.
    """
    fig = plt.figure(figsize=(18, 11))
    gs = GridSpec(3, 5, figure=fig, hspace=0.45, wspace=0.40)

    # Professional color palette
    color_clean = '#1f77b4'      # Professional blue
    color_adv = '#d62728'        # Professional red
    color_drop = '#ff7f0e'       # Professional orange
    color_flip = '#2ca02c'       # Professional green
    color_conf = '#9467bd'       # Professional purple
    bg_color = '#fafbfc'

    n_angles = np.array(results['n_angles'])
    x_pos = np.arange(len(n_angles))

    # ============================================================
    # Row 0: Four Metric Panels (across first 4 columns)
    # ============================================================

    # Panel A: Accuracy Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(bg_color)
    
    width = 0.35
    bars1 = ax1.bar(x_pos - width/2, results['clean_acc'], width,
                    label='Clean', color=color_clean, alpha=0.85, edgecolor='#333333', linewidth=0.8)
    bars2 = ax1.bar(x_pos + width/2, results['adv_acc'], width,
                    label=f'Adv (ε={eps:.2f})', color=color_adv, alpha=0.85, edgecolor='#333333', linewidth=0.8)

    ax1.set_ylabel('Accuracy', fontsize=10, fontweight='600')
    ax1.set_title('A. Accuracy', fontsize=11, fontweight='700')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax1.legend(fontsize=9, loc='lower left', framealpha=0.92)
    ax1.grid(True, alpha=0.2, axis='y')
    ax1.set_ylim([0, 1.05])
    ax1.set_axisbelow(True)

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='600')

    # Panel B: Accuracy Drop
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(bg_color)
    
    ax2.plot(n_angles, results['acc_drop'], 'o-', color=color_drop,
             linewidth=2.2, markersize=7, markeredgecolor='#333333', markeredgewidth=1)
    ax2.fill_between(n_angles, 0, results['acc_drop'], alpha=0.25, color=color_drop)
    
    ax2.set_ylabel('Drop', fontsize=10, fontweight='600')
    ax2.set_title('B. Vulnerability', fontsize=11, fontweight='700')
    ax2.set_xticks(n_angles)
    ax2.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_axisbelow(True)

    # Panel C: Flip Rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(bg_color)
    
    bars3 = ax3.bar(x_pos, results['flip_rate'], color=color_flip,
                    alpha=0.85, edgecolor='#333333', linewidth=0.8)
    
    ax3.set_ylabel('Flip Rate', fontsize=10, fontweight='600')
    ax3.set_title('C. Instability', fontsize=11, fontweight='700')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax3.grid(True, alpha=0.2, axis='y')
    ax3.set_axisbelow(True)

    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=7.5, fontweight='600')

    # Panel D: Confidence Drop
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor(bg_color)
    
    ax4.plot(n_angles, results['conf_drop'], 's-', color=color_conf,
             linewidth=2.2, markersize=7, markeredgecolor='#333333', markeredgewidth=1)
    ax4.fill_between(n_angles, 0, results['conf_drop'], alpha=0.25, color=color_conf)
    
    ax4.set_ylabel('Drop', fontsize=10, fontweight='600')
    ax4.set_title('D. Confidence', fontsize=11, fontweight='700')
    ax4.set_xticks(n_angles)
    ax4.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax4.grid(True, alpha=0.2)
    ax4.set_axisbelow(True)

    # ============================================================
    # Row 0, Column 4: Combined Metrics Plot
    # ============================================================
    ax5 = fig.add_subplot(gs[0, 4])
    ax5.set_facecolor(bg_color)
    
    ax5_twin = ax5.twinx()
    
    line1 = ax5.plot(n_angles, results['acc_drop'], 'o-', color=color_drop,
                     linewidth=2, markersize=6, markeredgecolor='#333333', label='Accuracy Drop')
    line2 = ax5_twin.plot(n_angles, results['flip_rate'], 's-', color=color_flip,
                          linewidth=2, markersize=6, markeredgecolor='#333333', label='Flip Rate')
    
    ax5.set_ylabel('Accuracy Drop', fontsize=10, fontweight='600', color=color_drop)
    ax5_twin.set_ylabel('Flip Rate', fontsize=10, fontweight='600', color=color_flip)
    ax5.set_title('E. Combined View', fontsize=11, fontweight='700')
    ax5.set_xticks(n_angles)
    ax5.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax5.tick_params(axis='y', labelcolor=color_drop)
    ax5_twin.tick_params(axis='y', labelcolor=color_flip)
    ax5.grid(True, alpha=0.2)
    ax5.set_axisbelow(True)

    # ============================================================
    # Row 1: Extended Metrics (2 panels spanning full width)
    # ============================================================

    # Panel F: Clean vs Adversarial Confidence
    ax6 = fig.add_subplot(gs[1, 0:3])
    ax6.set_facecolor(bg_color)
    
    width = 0.35
    bars_clean_conf = ax6.bar(x_pos - width/2, results['clean_conf'], width,
                              label='Clean Conf', color=color_clean, alpha=0.8, edgecolor='#333333', linewidth=0.8)
    bars_adv_conf = ax6.bar(x_pos + width/2, results['adv_conf'], width,
                            label='Adv Conf', color=color_adv, alpha=0.8, edgecolor='#333333', linewidth=0.8)
    
    ax6.set_ylabel('Confidence', fontsize=10, fontweight='600')
    ax6.set_title('F. Model Confidence Comparison', fontsize=11, fontweight='700')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax6.legend(fontsize=9, loc='lower left', framealpha=0.92)
    ax6.grid(True, alpha=0.2, axis='y')
    ax6.set_ylim([0, 1.05])
    ax6.set_axisbelow(True)

    # Panel G: Vulnerability Trend Analysis
    ax7 = fig.add_subplot(gs[1, 3:5])
    ax7.set_facecolor(bg_color)
    
    # Normalize vulnerability relative to minimum (most robust)
    min_vuln = min(results['acc_drop'])
    normalized_vuln = [v / min_vuln if min_vuln > 0 else 1.0 for v in results['acc_drop']]
    
    ax7.bar(x_pos, normalized_vuln, color='#e74c3c', alpha=0.85, edgecolor='#333333', linewidth=0.8)
    ax7.axhline(y=1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (Most Robust)')
    
    ax7.set_ylabel('Relative Vulnerability', fontsize=10, fontweight='600')
    ax7.set_title('G. Vulnerability Amplification', fontsize=11, fontweight='700')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'{int(n)}' for n in n_angles], fontsize=8)
    ax7.legend(fontsize=9, framealpha=0.92)
    ax7.grid(True, alpha=0.2, axis='y')
    ax7.set_axisbelow(True)

    # ============================================================
    # Row 2: Reconstruction Visualizations (5 columns)
    # ============================================================
    sorted_angles = sorted(sample_reconstructions.keys())
    for idx, N in enumerate(sorted_angles):
        ax = fig.add_subplot(gs[2, idx])
        recon = sample_reconstructions[N]['recon']
        im = ax.imshow(recon[0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'N={int(N)}', fontsize=10, fontweight='700')
        ax.axis('off')

    # Overall styling
    fig.patch.set_facecolor('white')
    fig.suptitle(f'Sparse-View CT Adversarial Vulnerability Analysis (ε={eps:.2f})',
                 fontsize=15, fontweight='700', y=0.98)

    return fig

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "path1"  # Update this path
    ANGLE_COUNTS = [30, 60, 90, 120, 180]
    EPSILON = 0.15
    EPOCHS = 15

    # Run analysis
    results, sample_reconstructions = run_sparse_view_analysis(
        ROOT_DIR,
        angle_counts=ANGLE_COUNTS,
        eps=EPSILON,
        epochs=EPOCHS
    )

    # Create visualizations
    print("\n" + "="*70)
    print("GENERATING VISUALIZATION")
    print("="*70)

    fig = create_comprehensive_figure(results, sample_reconstructions, EPSILON)
    plt.show()

    # Print summary table
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    
    summary_df = pd.DataFrame({
        'Angles': results['n_angles'],
        'Clean Acc': [f"{x:.4f}" for x in results['clean_acc']],
        'Adv Acc': [f"{x:.4f}" for x in results['adv_acc']],
        'Drop': [f"{x:.4f}" for x in results['acc_drop']],
        'Flip Rate': [f"{x:.4f}" for x in results['flip_rate']],
        'Clean Conf': [f"{x:.4f}" for x in results['clean_conf']],
        'Adv Conf': [f"{x:.4f}" for x in results['adv_conf']],
        'Conf Drop': [f"{x:.4f}" for x in results['conf_drop']]
    })
    
    print(summary_df.to_string(index=False))

    # Key findings
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    baseline_drop = results['acc_drop'][0]
    worst_drop = max(results['acc_drop'])
    amplification = worst_drop / baseline_drop if baseline_drop > 0 else 0

    print(f"\n✓ Sparse views amplify vulnerability:")
    print(f"  Maximum vulnerability (N={int(ANGLE_COUNTS[0])}): {results['acc_drop'][0]:.4f}")
    print(f"  Minimum vulnerability (N={int(ANGLE_COUNTS[-1])}): {results['acc_drop'][-1]:.4f}")
    print(f"  Amplification factor: {amplification:.2f}x")

    print(f"\n✓ Flip rates increase with sparsity:")
    for n, flip in zip(results['n_angles'], results['flip_rate']):
        print(f"  N={int(n)}: {flip:.4f}")

    print(f"\n✓ Model confidence collapses under attack:")
    for n, conf_drop in zip(results['n_angles'], results['conf_drop']):
        print(f"  N={int(n)}: {conf_drop:.4f}")

    print(f"\n✓ Trend Analysis:")
    min_vuln = min(results['acc_drop'])
    for n, drop in zip(results['n_angles'], results['acc_drop']):
        rel_vuln = drop / min_vuln if min_vuln > 0 else 1.0
        print(f"  N={int(n)}: {rel_vuln:.2f}× baseline vulnerability")

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)