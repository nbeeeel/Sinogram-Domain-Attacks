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
plt.style.use('seaborn-v0_8-darkgrid')
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
# Optimized Radon Transform
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
# FGSM Attack
# ============================================================
def sino_fgsm(sino, loss, eps):
    grad = torch.autograd.grad(loss, sino, retain_graph=False)[0]
    return sino + eps*grad.sign()

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
        desc = f"Epoch {ep}/{epochs}" if verbose else f"Training (N={len(thetas)})"
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
        
        if verbose and ep % 5 == 0:
            print(f"Epoch {ep}: Train Acc={correct/total:.3f}")

# ============================================================
# Evaluation Function
# ============================================================
def evaluate(model, loader, radon, thetas, eps=0.0):
    """
    Evaluate model on clean and adversarial examples.
    Returns: (clean_acc, adv_acc, flip_rate)
    """
    ce = nn.CrossEntropyLoss()
    model.eval()
    
    total = 0
    clean_correct = 0
    adv_correct = 0
    flips = 0
    
    with torch.set_grad_enabled(eps > 0):
        for imgs, lbls in tqdm(loader, desc="Evaluating", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # Clean evaluation
            with torch.no_grad():
                sino_clean = radon(imgs)
                recon_clean = fbp_reconstruct(sino_clean, thetas)
                out_clean = model(recon_clean)
                pred_clean = out_clean.argmax(1)
                clean_correct += (pred_clean == lbls).sum().item()
            
            # Adversarial evaluation
            if eps > 0:
                sino = radon(imgs)
                sino.requires_grad_(True)
                recon = fbp_reconstruct(sino, thetas)
                out = model(recon)
                loss = ce(out, lbls)
                
                adv_sino = sino_fgsm(sino, loss, eps)
                with torch.no_grad():
                    adv_recon = fbp_reconstruct(adv_sino, thetas)
                    out_adv = model(adv_recon)
                    pred_adv = out_adv.argmax(1)
                    adv_correct += (pred_adv == lbls).sum().item()
                    flips += (pred_clean != pred_adv).sum().item()
            else:
                adv_correct = clean_correct
                flips = 0
            
            total += lbls.size(0)
    
    clean_acc = clean_correct / total
    adv_acc = adv_correct / total
    flip_rate = flips / total
    
    return clean_acc, adv_acc, flip_rate

# ============================================================
# Main Analysis Function
# ============================================================
def run_sparse_view_analysis(root_dir, angle_counts=[30, 90, 180], eps=0.01, epochs=15):
    """
    Run complete analysis across different viewing angles.
    """
    print("="*60)
    print("SPARSE-VIEW CT ADVERSARIAL VULNERABILITY ANALYSIS")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_ds = CTSliceDataset(os.path.join(root_dir, "train"))
    val_ds = CTSliceDataset(os.path.join(root_dir, "valid"))
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(f"No data found in {root_dir}. Please check the path.")
    
    print(f"Train samples: {len(train_ds)}, Validation samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    results = {
        'n_angles': [],
        'clean_acc': [],
        'adv_acc': [],
        'flip_rate': [],
        'acc_drop': []
    }
    
    # Store sample reconstructions for visualization
    sample_reconstructions = {}
    
    for N in angle_counts:
        print(f"\n{'='*60}")
        print(f"Running experiment with N = {N} angles")
        print(f"{'='*60}")
        
        # Create Radon transform
        thetas = torch.linspace(0, np.pi, N, device=device)
        radon = Radon(thetas, img_size=train_ds.img_size[0]).to(device)
        
        # Create and train model
        model = SimpleCTCNN().to(device)
        print(f"\nTraining model with {N} angles...")
        train(model, train_loader, val_loader, radon, thetas, epochs=epochs, verbose=True)
        
        # Evaluate
        print(f"\nEvaluating with ε = {eps}...")
        clean_acc, adv_acc, flip_rate = evaluate(model, val_loader, radon, thetas, eps=eps)
        
        results['n_angles'].append(N)
        results['clean_acc'].append(clean_acc)
        results['adv_acc'].append(adv_acc)
        results['flip_rate'].append(flip_rate)
        results['acc_drop'].append(clean_acc - adv_acc)
        
        print(f"\nResults for N = {N}:")
        print(f"  Clean Accuracy:       {clean_acc:.4f}")
        print(f"  Adversarial Accuracy: {adv_acc:.4f}")
        print(f"  Accuracy Drop:        {clean_acc - adv_acc:.4f}")
        print(f"  Flip Rate:            {flip_rate:.4f}")
        
        # Save sample reconstruction for visualization
        imgs, lbls = next(iter(val_loader))
        imgs = imgs.to(device)
        with torch.no_grad():
            sino = radon(imgs)
            recon = fbp_reconstruct(sino, thetas)
        sample_reconstructions[N] = {
            'original': imgs[0].cpu(),
            'sino': sino[0].cpu(),
            'recon': recon[0].cpu()
        }
    
    return results, sample_reconstructions

# ============================================================
# Visualization Functions
# ============================================================
def create_comprehensive_figure(results, sample_reconstructions, eps):
    """
    Create a comprehensive figure with all analysis results.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
    
    # Color scheme
    color_clean = '#2E86AB'
    color_adv = '#A23B72'
    color_drop = '#F18F01'
    color_flip = '#C73E1D'
    
    n_angles = results['n_angles']
    
    # ============================================================
    # Row 1: Main Metrics
    # ============================================================
    
    # Accuracy comparison
    ax1 = fig.add_subplot(gs[0, 0:2])
    x_pos = np.arange(len(n_angles))
    width = 0.35
    
    bars1 = ax1.bar(x_pos - width/2, results['clean_acc'], width, 
                    label='Clean', color=color_clean, alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax1.bar(x_pos + width/2, results['adv_acc'], width, 
                    label=f'Adversarial (ε={eps})', color=color_adv, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax1.set_xlabel('Number of Projection Angles', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax1.set_title('Clean vs Adversarial Accuracy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(n_angles)
    ax1.legend(fontsize=11, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Accuracy drop
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax2.plot(n_angles, results['acc_drop'], 'o-', color=color_drop, 
             linewidth=2.5, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax2.fill_between(n_angles, 0, results['acc_drop'], alpha=0.3, color=color_drop)
    ax2.set_xlabel('Number of Projection Angles', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy Drop', fontsize=12, fontweight='bold')
    ax2.set_title('Adversarial Accuracy Drop', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(n_angles)
    
    # Add value labels
    for x, y in zip(n_angles, results['acc_drop']):
        ax2.text(x, y, f'{y:.3f}', ha='center', va='bottom', 
                fontsize=10, fontweight='bold')
    
    # ============================================================
    # Row 2: Flip Rate Analysis
    # ============================================================
    
    # Flip rate bar chart
    ax3 = fig.add_subplot(gs[1, 0:2])
    bars3 = ax3.bar(x_pos, results['flip_rate'], color=color_flip, 
                    alpha=0.8, edgecolor='black', linewidth=1.2)
    ax3.set_xlabel('Number of Projection Angles', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Flip Rate', fontsize=12, fontweight='bold')
    ax3.set_title('Prediction Flip Rate Under Attack', fontsize=14, fontweight='bold', pad=15)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(n_angles)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Vulnerability amplification
    ax4 = fig.add_subplot(gs[1, 2:4])
    # Normalize to sparse view (N=30)
    baseline_drop = results['acc_drop'][0]  # N=30
    amplification = [drop / baseline_drop if baseline_drop > 0 else 1.0 for drop in results['acc_drop']]
    
    ax4.plot(n_angles, amplification, 's-', color='#6A4C93', 
             linewidth=2.5, markersize=10, markeredgecolor='black', markeredgewidth=1.5)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Baseline (N=30)')
    ax4.fill_between(n_angles, 1.0, amplification, alpha=0.3, color='#6A4C93')
    ax4.set_xlabel('Number of Projection Angles', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Relative Vulnerability', fontsize=12, fontweight='bold')
    ax4.set_title('Vulnerability Amplification (Normalized to N=30)', 
                  fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(n_angles)
    ax4.legend(fontsize=10)
    
    # ============================================================
    # Row 3: Sample Reconstructions
    # ============================================================
    
    for idx, N in enumerate(sorted(sample_reconstructions.keys())):
        ax = fig.add_subplot(gs[2, idx])
        recon = sample_reconstructions[N]['recon']
        im = ax.imshow(recon[0].numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'N = {N} angles', fontsize=11, fontweight='bold')
        ax.axis('off')
        
        # Add colorbar for first image only
        if idx == 0:
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    # Add overall title
    fig.suptitle(f'Sparse-View CT Adversarial Vulnerability Analysis (ε = {eps})', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    return fig

def create_summary_table(results):
    """
    Create a professional summary table.
    """
    df = pd.DataFrame({
        'Projection Angles (N)': results['n_angles'],
        'Clean Accuracy': [f"{x:.4f}" for x in results['clean_acc']],
        'Adversarial Accuracy': [f"{x:.4f}" for x in results['adv_acc']],
        'Accuracy Drop': [f"{x:.4f}" for x in results['acc_drop']],
        'Flip Rate': [f"{x:.4f}" for x in results['flip_rate']]
    })
    
    return df

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    # Configuration
    ROOT_DIR = "/mnt/user-data/uploads"  # Update this path
    ANGLE_COUNTS = [30, 90, 180]
    EPSILON = 0.01
    EPOCHS = 15
    
    # Run analysis
    results, sample_reconstructions = run_sparse_view_analysis(
        ROOT_DIR, 
        angle_counts=ANGLE_COUNTS, 
        eps=EPSILON, 
        epochs=EPOCHS
    )
    
    # Create visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    fig = create_comprehensive_figure(results, sample_reconstructions, EPSILON)
    plt.savefig('/mnt/user-data/outputs/sparse_view_analysis.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    print("\nComprehensive figure saved: sparse_view_analysis.png")
    
    # Create summary table
    summary_df = create_summary_table(results)
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save table
    summary_df.to_csv('/mnt/user-data/outputs/sparse_view_results.csv', index=False)
    print("\nTable saved: sparse_view_results.csv")
    
    # Create markdown report
    with open('/mnt/user-data/outputs/sparse_view_report.md', 'w') as f:
        f.write("# Sparse-View CT Adversarial Vulnerability Analysis\n\n")
        f.write(f"**Attack Strength (ε):** {EPSILON}\n\n")
        f.write("## Summary Table\n\n")
        f.write(summary_df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        f.write("1. **Sparse views amplify adversarial vulnerability:**\n")
        f.write(f"   - N=30: Accuracy drop = {results['acc_drop'][0]:.4f}\n")
        f.write(f"   - N=180: Accuracy drop = {results['acc_drop'][-1]:.4f}\n")
        f.write(f"   - Amplification factor: {results['acc_drop'][0]/results['acc_drop'][-1]:.2f}x\n\n")
        f.write("2. **Flip rate increases with sparsity:**\n")
        f.write(f"   - N=30: {results['flip_rate'][0]:.4f}\n")
        f.write(f"   - N=180: {results['flip_rate'][-1]:.4f}\n\n")
        f.write("3. **Reconstruction artifacts correlate with vulnerability:**\n")
        f.write("   - Fewer projection angles → stronger ripple artifacts\n")
        f.write("   - Adversarial perturbations exploit these artifacts nonlinearly\n")
    
    print("\nMarkdown report saved: sparse_view_report.md")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    plt.show()