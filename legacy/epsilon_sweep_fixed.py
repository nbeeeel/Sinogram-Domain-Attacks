# ============================================================
# Frequency-Restricted Adversarial Attack Analysis (FIXED)
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
# Frequency-Restricted FGSM Attacks
# ============================================================
def create_frequency_mask(D, cutoff_ratio=0.3, mask_type='low'):
    """
    Create frequency mask for low-pass or high-pass filtering
    
    Args:
        D: Size of frequency domain
        cutoff_ratio: Fraction of frequencies to keep (0-1)
        mask_type: 'low' for low-pass, 'high' for high-pass
    """
    freqs = torch.fft.fftfreq(D)
    cutoff_freq = cutoff_ratio * 0.5  # Nyquist is 0.5
    
    if mask_type == 'low':
        # Keep low frequencies (smooth perturbations)
        mask = (torch.abs(freqs) <= cutoff_freq).float()
    elif mask_type == 'high':
        # Keep high frequencies (sharp perturbations)
        mask = (torch.abs(freqs) > cutoff_freq).float()
    else:
        raise ValueError("mask_type must be 'low' or 'high'")
    
    return mask.to(device).view(1, 1, -1, 1)

def sino_fgsm_frequency_restricted(sino, loss, eps, mask_type='low', cutoff_ratio=0.3):
    """
    Frequency-restricted FGSM attack
    
    Args:
        sino: Sinogram tensor
        loss: Loss to maximize
        eps: Attack strength
        mask_type: 'low' or 'high' frequency
        cutoff_ratio: Frequency cutoff (0-1)
    """
    grad = torch.autograd.grad(loss, sino, retain_graph=False)[0]
    
    # Transform gradient to frequency domain
    G = fft.fft(grad, dim=2)
    
    # Apply frequency mask
    D = grad.shape[2]
    mask = create_frequency_mask(D, cutoff_ratio, mask_type)
    G_filtered = G * mask
    
    # Transform back and create adversarial example
    grad_filtered = fft.ifft(G_filtered, dim=2).real
    
    return sino + eps * grad_filtered.sign()

def sino_fgsm_standard(sino, loss, eps):
    """Standard FGSM (no frequency restriction)"""
    grad = torch.autograd.grad(loss, sino, retain_graph=False)[0]
    return sino + eps * grad.sign()

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
# Evaluation Function for Epsilon Sweep
# ============================================================
def evaluate_epsilon_sweep(model, loader, radon, thetas, eps_list):
    """
    Sweep over epsilon values and record accuracy and flip rates
    
    Args:
        model: Trained model
        loader: Data loader
        radon: Radon transform
        thetas: Angles
        eps_list: List of epsilon values to test
    
    Returns:
        results: Dictionary with metrics for each epsilon
    """
    ce = nn.CrossEntropyLoss()
    model.eval()
    
    results = {
        'epsilon': [],
        'accuracy': [],
        'flip_rate': []
    }
    
    for eps in eps_list:
        correct_clean = 0
        correct_adv = 0
        flips = 0
        total = 0
        
        for imgs, lbls in tqdm(loader, desc=f"ε={eps:.4f}", leave=False):
            imgs, lbls = imgs.to(device), lbls.to(device)
            total += lbls.size(0)
            
            # Clean predictions
            with torch.no_grad():
                sino_clean = radon(imgs)
                recon_clean = fbp_reconstruct(sino_clean, thetas)
                out_clean = model(recon_clean)
                pred_clean = out_clean.argmax(1)
                correct_clean += (pred_clean == lbls).sum().item()
            
            # Adversarial attack
            sino = radon(imgs)
            sino = sino.detach().requires_grad_(True)
            recon = fbp_reconstruct(sino, thetas)
            out = model(recon)
            loss = ce(out, lbls)
            
            # FGSM attack
            adv_sino = sino_fgsm_standard(sino, loss, eps)
            
            # Evaluate adversarial
            with torch.no_grad():
                recon_adv = fbp_reconstruct(adv_sino, thetas)
                out_adv = model(recon_adv)
                pred_adv = out_adv.argmax(1)
                correct_adv += (pred_adv == lbls).sum().item()
                flips += (pred_clean != pred_adv).sum().item()
        
        # Store results
        results['epsilon'].append(eps)
        results['accuracy'].append(correct_adv / total)
        results['flip_rate'].append(flips / total)
    
    return results

# ============================================================
# Visualization: Epsilon-Amplification Curves
# ============================================================
def visualize_epsilon_amplification(results, clean_acc):
    """
    Visualize epsilon-amplification curves with sophisticated research-grade aesthetic
    
    Args:
        results: Dictionary with epsilon, accuracy, flip_rate lists
        clean_acc: Clean baseline accuracy
    """
    # Light, sophisticated color scheme
    color_acc = '#4A90E2'      # Soft blue
    color_flip = '#E86B5A'     # Soft coral
    bg_color = '#FAFBFC'       # Very light gray
    grid_color = '#E8EAED'     # Light grid
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Set background
    for ax in [ax1, ax2]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
    
    eps = np.array(results['epsilon'])
    acc = np.array(results['accuracy'])
    flip = np.array(results['flip_rate'])
    
    # ============================================================
    # Panel A: Accuracy vs Epsilon
    # ============================================================
    ax1.plot(eps, acc, marker='o', markersize=8, linewidth=2.5, 
             color=color_acc, label='Adversarial Accuracy', zorder=3)
    
    # Add baseline
    ax1.axhline(y=clean_acc, color='#999999', linestyle='--', 
                linewidth=1.5, alpha=0.7, label='Clean Baseline')
    
    # Styling
    ax1.set_xlabel('Perturbation Budget (ε)', fontsize=12, fontweight='600')
    ax1.set_ylabel('Accuracy', fontsize=12, fontweight='600')
    ax1.set_title('A. Accuracy Degradation', fontsize=13, fontweight='700')
    ax1.legend(loc='best', framealpha=0.95, fontsize=10, frameon=True, fancybox=False)
    ax1.set_ylim([0, 1.05])
    
    # Tick styling
    ax1.tick_params(labelsize=10, colors='#333333')
    for spine in ax1.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # ============================================================
    # Panel B: Flip Rate vs Epsilon
    # ============================================================
    ax2.plot(eps, flip, marker='s', markersize=8, linewidth=2.5, 
             color=color_flip, label='Prediction Flip Rate', zorder=3)
    
    # Styling
    ax2.set_xlabel('Perturbation Budget (ε)', fontsize=12, fontweight='600')
    ax2.set_ylabel('Flip Rate', fontsize=12, fontweight='600')
    ax2.set_title('B. Prediction Flip Sensitivity', fontsize=13, fontweight='700')
    ax2.legend(loc='best', framealpha=0.95, fontsize=10, frameon=True, fancybox=False)
    ax2.set_ylim([0, 1.05])
    
    # Tick styling
    ax2.tick_params(labelsize=10, colors='#333333')
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Overall styling
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    return fig

def visualize_amplification_analysis(results, clean_acc):
    """
    Create detailed analysis with amplification factor
    """
    color_acc = '#4A90E2'
    color_flip = '#E86B5A'
    bg_color = '#FAFBFC'
    grid_color = '#E8EAED'
    
    eps = np.array(results['epsilon'])
    acc = np.array(results['accuracy'])
    flip = np.array(results['flip_rate'])
    
    # Calculate amplification: flip_rate / accuracy_drop
    acc_drop = clean_acc - acc
    # Avoid division by zero
    amplification = np.divide(flip, acc_drop, where=acc_drop>1e-6, 
                              out=np.zeros_like(flip))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax in [ax1, ax2]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
    
    # ============================================================
    # Panel A: Accuracy Drop vs Epsilon
    # ============================================================
    ax1.fill_between(eps, 0, acc_drop, alpha=0.3, color=color_acc, label='Accuracy Drop Region')
    ax1.plot(eps, acc_drop, marker='o', markersize=8, linewidth=2.5, 
             color=color_acc, label='Accuracy Drop', zorder=3)
    
    ax1.set_xlabel('Perturbation Budget (ε)', fontsize=12, fontweight='600')
    ax1.set_ylabel('Accuracy Drop (1 - Accuracy)', fontsize=12, fontweight='600')
    ax1.set_title('A. Accuracy Drop Curve', fontsize=13, fontweight='700')
    ax1.legend(loc='best', framealpha=0.95, fontsize=10)
    ax1.tick_params(labelsize=10, colors='#333333')
    
    for spine in ax1.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # ============================================================
    # Panel B: Amplification Factor (Flip Rate / Accuracy Drop)
    # ============================================================
    # Only plot where accuracy drop > 0
    valid_idx = acc_drop > 1e-6
    if np.any(valid_idx):
        ax2.plot(eps[valid_idx], amplification[valid_idx], marker='D', markersize=8, 
                 linewidth=2.5, color=color_flip, label='Amplification Factor', zorder=3)
    
    ax2.set_xlabel('Perturbation Budget (ε)', fontsize=12, fontweight='600')
    ax2.set_ylabel('Amplification (Flip Rate / Accuracy Drop)', fontsize=12, fontweight='600')
    ax2.set_title('B. Nonlinear Amplification', fontsize=13, fontweight='700')
    ax2.legend(loc='best', framealpha=0.95, fontsize=10)
    ax2.tick_params(labelsize=10, colors='#333333')
    
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    return fig

def visualize_transition_analysis(results, clean_acc):
    """
    Highlight sharp transition region
    """
    color_acc = '#4A90E2'
    color_flip = '#E86B5A'
    color_transition = '#F5A623'
    bg_color = '#FAFBFC'
    grid_color = '#E8EAED'
    
    eps = np.array(results['epsilon'])
    acc = np.array(results['accuracy'])
    flip = np.array(results['flip_rate'])
    
    fig = plt.figure(figsize=(14, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor(bg_color)
    ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
    ax.set_axisbelow(True)
    
    # Dual axis
    ax2 = ax.twinx()
    ax2.set_facecolor(bg_color)
    
    # Plot accuracy on left axis
    line1 = ax.plot(eps, acc, marker='o', markersize=10, linewidth=3, 
                    color=color_acc, label='Adversarial Accuracy', zorder=3)
    ax.axhline(y=clean_acc, color='#999999', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Plot flip rate on right axis
    line2 = ax2.plot(eps, flip, marker='s', markersize=10, linewidth=3, 
                     color=color_flip, label='Prediction Flip Rate', zorder=3)
    
    # Find sharp transition region (steepest slope)
    if len(eps) > 2:
        slopes = np.diff(acc) / np.diff(eps)
        steepest_idx = np.argmin(slopes)  # Most negative slope
        
        if steepest_idx > 0 and steepest_idx < len(eps) - 1:
            trans_eps = eps[steepest_idx:steepest_idx+2]
            ax.axvspan(trans_eps[0], trans_eps[-1], alpha=0.15, color=color_transition, 
                      label='Transition Region')
    
    # Styling
    ax.set_xlabel('Perturbation Budget (ε)', fontsize=12, fontweight='600')
    ax.set_ylabel('Adversarial Accuracy', fontsize=12, fontweight='600', color=color_acc)
    ax2.set_ylabel('Prediction Flip Rate', fontsize=12, fontweight='600', color=color_flip)
    ax.set_title('ε-Amplification: Accuracy vs Flip Rate (Dual Axis)', 
                fontsize=13, fontweight='700')
    
    ax.tick_params(axis='y', labelcolor=color_acc, labelsize=10)
    ax2.tick_params(axis='y', labelcolor=color_flip, labelsize=10)
    ax.tick_params(axis='x', labelsize=10, colors='#333333')
    
    # Legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right', framealpha=0.95, fontsize=10)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    ax.set_ylim([0, 1.05])
    ax2.set_ylim([0, 1.05])
    
    fig.patch.set_facecolor('white')
    plt.tight_layout()
    
    return fig

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("="*80)
    print("FREQUENCY-RESTRICTED ADVERSARIAL ATTACK ANALYSIS")
    print("="*80)
    print("\nQuestion: Are low-frequency perturbations more dangerous than high-frequency?")
    print("\nHypothesis: Low-frequency attacks survive FBP, high-frequency attacks cancel\n")
    
    # Configuration
    ROOT_DIR = "path1"  # Update this path
    EPOCHS = 20
    EPS_LIST = [0, 0.05, 0.1, 0.15, 0.2, 0.3]  # Epsilon sweep (increased 10-20x for effectiveness)
    
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
    
    # Epsilon sweep
    print(f"\nEvaluating epsilon-amplification sweep...")
    print(f"Epsilon values: {EPS_LIST}")
    results = evaluate_epsilon_sweep(model, val_loader, radon, thetas, EPS_LIST)
    
    # Display results
    print("\n" + "="*80)
    print("ε-AMPLIFICATION ANALYSIS RESULTS")
    print("="*80)
    print(f"\nClean Baseline Accuracy: {clean_acc:.4f}\n")
    print(f"{'ε':<10} {'Accuracy':<15} {'Flip Rate':<15}")
    print("-" * 40)
    for eps, acc, flip in zip(results['epsilon'], results['accuracy'], results['flip_rate']):
        print(f"{eps:<10.4f} {acc:<15.4f} {flip:<15.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("VULNERABILITY ANALYSIS")
    print("="*80)
    
    accs = np.array(results['accuracy'])
    flips = np.array(results['flip_rate'])
    eps = np.array(results['epsilon'])
    
    # Find transition region
    acc_drops = clean_acc - accs
    valid_idx = acc_drops > 1e-6
    
    if np.any(valid_idx):
        # Find steepest drop
        slopes = np.diff(accs) / np.diff(eps)
        steepest_idx = np.argmin(slopes)
        
        print(f"\n✓ Sharp transition detected between ε={eps[steepest_idx]:.4f} and ε={eps[steepest_idx+1]:.4f}")
        print(f"  Accuracy drops from {accs[steepest_idx]:.4f} to {accs[steepest_idx+1]:.4f}")
        print(f"  Drop rate: {abs(slopes[steepest_idx]):.4f} per unit ε")
        
        # Amplification analysis
        amplification = np.divide(flips[valid_idx], acc_drops[valid_idx], 
                                 out=np.zeros_like(flips[valid_idx]))
        print(f"\n✓ Amplification Factor (Flip Rate / Accuracy Drop):")
        for i, (e, amp) in enumerate(zip(eps[valid_idx], amplification)):
            if not np.isnan(amp) and not np.isinf(amp):
                print(f"  ε={e:.4f}: {amp:.2f}× amplification")
        
        # Nonlinearity check
        if np.any(flips[valid_idx] > 2 * acc_drops[valid_idx]):
            print(f"\n✓ NONLINEAR EFFECT DETECTED: Flip rate >> accuracy drop")
            print(f"  This indicates vulnerability to low perturbations")
    else:
        print("\n⚠️  Insufficient attack effectiveness detected")
        print("   Try increasing epsilon further: [0, 0.1, 0.2, 0.3, 0.4, 0.5]")
        print("   Or adjust the model/attack parameters")
    
    print("\n" + "="*80)
    
    # Visualizations
    print("\nGenerating research-grade visualizations...")
    
    fig1 = visualize_epsilon_amplification(results, clean_acc)
    plt.show()
    
    fig2 = visualize_amplification_analysis(results, clean_acc)
    plt.show()
    
    fig3 = visualize_transition_analysis(results, clean_acc)
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)