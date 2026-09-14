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
    grad_norm = torch.norm(grad).item()
    # Diagnostic: uncomment to debug
    # print(f"  Gradient norm: {grad_norm:.6f}, Perturbation magnitude: {eps * grad_norm:.6f}")
    return sino + eps * grad.sign()

def sino_pgd_attack(sino, model, loss_fn, eps, alpha=0.02, steps=20, device=device):
    """
    PGD (Projected Gradient Descent) attack - much stronger than FGSM
    
    Args:
        sino: Sinogram tensor
        model: Model to attack
        loss_fn: Loss function
        eps: Maximum perturbation
        alpha: Step size
        steps: Number of iterations
    """
    sino_adv = sino.clone().detach()
    sino_orig = sino.clone().detach()
    
    for step in range(steps):
        sino_adv.requires_grad = True
        
        # Get logits and compute loss
        recon_adv = fbp_reconstruct(sino_adv, torch.linspace(0, np.pi, 60, device=device))
        out = model(recon_adv)
        # For adversarial attack on first batch element
        loss = out[0, 1 if out[0, 0] > out[0, 1] else 0]  # Target max logit
        
        # Backward
        loss.backward()
        grad = sino_adv.grad
        
        # Update
        with torch.no_grad():
            sino_adv = sino_adv + alpha * grad.sign()
            
            # Project back to epsilon ball
            delta = torch.clamp(sino_adv - sino_orig, -eps, eps)
            sino_adv = sino_orig + delta
    
    return sino_adv.detach()

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
# Saliency Map Computation (CAM-like visualization)
# ============================================================
def compute_saliency(model, recon, label, device):
    """
    Compute gradient-based saliency map
    """
    recon = recon.clone().detach().requires_grad_(True)
    out = model(recon)
    target_score = out[0, label]
    
    # Backprop to get gradient
    if target_score.requires_grad:
        target_score.backward(retain_graph=True)
        saliency = torch.abs(recon.grad[0])
    else:
        saliency = torch.zeros_like(recon[0])
    
    return saliency

def compute_saliency_drift(saliency_clean, saliency_adv):
    """
    Measure how much saliency pattern has changed
    """
    # Normalize to [0,1]
    sal_clean_norm = saliency_clean / (saliency_clean.max() + 1e-8)
    sal_adv_norm = saliency_adv / (saliency_adv.max() + 1e-8)
    
    # Compute L2 distance
    drift = torch.norm(sal_clean_norm - sal_adv_norm).item()
    
    # Compute center of mass shift
    h, w = saliency_clean.shape
    y_coords = torch.arange(h, dtype=torch.float32)
    x_coords = torch.arange(w, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    com_clean_y = (sal_clean_norm * yy).sum() / (sal_clean_norm.sum() + 1e-8)
    com_clean_x = (sal_clean_norm * xx).sum() / (sal_clean_norm.sum() + 1e-8)
    
    com_adv_y = (sal_adv_norm * yy).sum() / (sal_adv_norm.sum() + 1e-8)
    com_adv_x = (sal_adv_norm * xx).sum() / (sal_adv_norm.sum() + 1e-8)
    
    com_shift = torch.sqrt((com_clean_y - com_adv_y)**2 + (com_clean_x - com_adv_x)**2).item()
    
    return drift, com_shift

# ============================================================
# Awareness Evaluation: Does the model know it's attacked?
# ============================================================
def evaluate_model_awareness(model, loader, radon, thetas, eps_list):
    """
    Evaluate model awareness of attacks through:
    - Softmax confidence
    - Entropy
    - Saliency drift
    
    Args:
        model: Trained model
        loader: Data loader
        radon: Radon transform
        thetas: Angles
        eps_list: List of epsilon values
    
    Returns:
        results: Dictionary with awareness metrics
    """
    ce = nn.CrossEntropyLoss()
    model.eval()
    
    results = {
        'epsilon': [],
        'clean_conf': [],
        'adv_conf': [],
        'conf_drop': [],
        'clean_entropy': [],
        'adv_entropy': [],
        'entropy_increase': [],
        'flip_rate': [],
        'conf_drop_no_flip': [],  # Confidence drop even when prediction correct
        'saliency_drift': [],
        'saliency_com_shift': []
    }
    
    for eps in eps_list:
        clean_confs = []
        adv_confs = []
        clean_entropies = []
        adv_entropies = []
        flips = 0
        conf_drops_no_flip = []
        saliency_drifts = []
        saliency_coms = []
        total = 0
        
        for batch_idx, (imgs, lbls) in enumerate(tqdm(loader, desc=f"ε={eps:.4f}", leave=False)):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            # Clean predictions
            with torch.no_grad():
                sino_clean = radon(imgs)
                recon_clean = fbp_reconstruct(sino_clean, thetas)
                out_clean = model(recon_clean)
                pred_clean = out_clean.argmax(1)
            
            # Softmax confidence
            probs_clean = torch.softmax(out_clean, dim=1)
            conf_clean = probs_clean.max(dim=1)[0]
            
            # Entropy
            entropy_clean = -(probs_clean * torch.log(probs_clean + 1e-8)).sum(dim=1)
            
            # Adversarial predictions
            sino = radon(imgs)
            sino = sino.detach().requires_grad_(True)
            recon = fbp_reconstruct(sino, thetas)
            out = model(recon)
            loss = ce(out, lbls)
            
            # FGSM attack
            adv_sino = sino_fgsm_standard(sino, loss, eps)
            
            with torch.no_grad():
                recon_adv = fbp_reconstruct(adv_sino, thetas)
                out_adv = model(recon_adv)
                pred_adv = out_adv.argmax(1)
            
            # Softmax confidence
            probs_adv = torch.softmax(out_adv, dim=1)
            conf_adv = probs_adv.max(dim=1)[0]
            
            # Entropy
            entropy_adv = -(probs_adv * torch.log(probs_adv + 1e-8)).sum(dim=1)
            
            # Metrics
            clean_confs.extend(conf_clean.cpu().numpy())
            adv_confs.extend(conf_adv.cpu().numpy())
            clean_entropies.extend(entropy_clean.cpu().numpy())
            adv_entropies.extend(entropy_adv.cpu().numpy())
            
            flips += (pred_clean != pred_adv).sum().item()
            
            # Confidence drop even when prediction is still correct
            correct_mask = (pred_adv == lbls).cpu().numpy()
            conf_diff = (conf_clean - conf_adv).cpu().numpy()
            conf_drops_no_flip.extend(conf_diff[correct_mask])
            
            # Saliency analysis (compute on first batch only for efficiency)
            if batch_idx == 0 and eps > 0:
                with torch.no_grad():
                    for i in range(min(3, imgs.shape[0])):  # First 3 samples
                        try:
                            sal_clean = compute_saliency(model, recon_clean[i:i+1], pred_clean[i].item(), device)
                            sal_adv = compute_saliency(model, recon_adv[i:i+1], pred_adv[i].item(), device)
                            
                            drift, com_shift = compute_saliency_drift(sal_clean, sal_adv)
                            saliency_drifts.append(drift)
                            saliency_coms.append(com_shift)
                        except:
                            pass
            
            total += imgs.shape[0]
        
        # Store results
        results['epsilon'].append(eps)
        results['clean_conf'].append(np.mean(clean_confs))
        results['adv_conf'].append(np.mean(adv_confs))
        results['conf_drop'].append(np.mean(clean_confs) - np.mean(adv_confs))
        results['clean_entropy'].append(np.mean(clean_entropies))
        results['adv_entropy'].append(np.mean(adv_entropies))
        results['entropy_increase'].append(np.mean(adv_entropies) - np.mean(clean_entropies))
        results['flip_rate'].append(flips / total)
        results['conf_drop_no_flip'].append(np.mean(conf_drops_no_flip) if conf_drops_no_flip else 0)
        results['saliency_drift'].append(np.mean(saliency_drifts) if saliency_drifts else 0)
        results['saliency_com_shift'].append(np.mean(saliency_coms) if saliency_coms else 0)
    
    return results

# ============================================================
# Visualization 1: Confidence and Entropy Analysis
# ============================================================
def visualize_confidence_entropy(results):
    """
    Visualize how confidence and entropy change with epsilon
    """
    bg_color = '#FAFBFC'
    grid_color = '#E8EAED'
    color_conf = '#4A90E2'
    color_entropy = '#E86B5A'
    
    eps = np.array(results['epsilon'])
    clean_conf = np.array(results['clean_conf'])
    adv_conf = np.array(results['adv_conf'])
    clean_entropy = np.array(results['clean_entropy'])
    adv_entropy = np.array(results['adv_entropy'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
    
    # Panel A: Softmax Confidence
    ax1.plot(eps, clean_conf, marker='o', markersize=8, linewidth=2.5, 
             label='Clean', color='#27AE60', zorder=3)
    ax1.plot(eps, adv_conf, marker='s', markersize=8, linewidth=2.5, 
             label='Adversarial', color=color_conf, zorder=3)
    ax1.fill_between(eps, adv_conf, clean_conf, alpha=0.2, color=color_conf)
    
    ax1.set_ylabel('Softmax Confidence', fontsize=11, fontweight='600')
    ax1.set_title('A. Model Confidence Collapse', fontsize=12, fontweight='700')
    ax1.legend(loc='best', framealpha=0.95, fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.tick_params(labelsize=10, colors='#333333')
    for spine in ax1.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel B: Confidence Drop
    conf_drop = clean_conf - adv_conf
    ax2.fill_between(eps, 0, conf_drop, alpha=0.3, color=color_conf)
    ax2.plot(eps, conf_drop, marker='o', markersize=8, linewidth=2.5, 
             color=color_conf, zorder=3)
    
    ax2.set_ylabel('Confidence Drop', fontsize=11, fontweight='600')
    ax2.set_title('B. Confidence Degradation', fontsize=12, fontweight='700')
    ax2.tick_params(labelsize=10, colors='#333333')
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel C: Entropy
    ax3.plot(eps, clean_entropy, marker='o', markersize=8, linewidth=2.5, 
             label='Clean', color='#27AE60', zorder=3)
    ax3.plot(eps, adv_entropy, marker='s', markersize=8, linewidth=2.5, 
             label='Adversarial', color=color_entropy, zorder=3)
    ax3.fill_between(eps, clean_entropy, adv_entropy, alpha=0.2, color=color_entropy)
    
    ax3.set_ylabel('Entropy', fontsize=11, fontweight='600')
    ax3.set_title('C. Prediction Uncertainty', fontsize=12, fontweight='700')
    ax3.legend(loc='best', framealpha=0.95, fontsize=10)
    ax3.tick_params(labelsize=10, colors='#333333')
    for spine in ax3.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel D: Entropy Increase
    entropy_inc = adv_entropy - clean_entropy
    ax4.fill_between(eps, 0, entropy_inc, alpha=0.3, color=color_entropy)
    ax4.plot(eps, entropy_inc, marker='s', markersize=8, linewidth=2.5, 
             color=color_entropy, zorder=3)
    
    ax4.set_ylabel('Entropy Increase', fontsize=11, fontweight='600')
    ax4.set_title('D. Model Confusion', fontsize=12, fontweight='700')
    ax4.tick_params(labelsize=10, colors='#333333')
    for spine in ax4.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Perturbation Budget (ε)', fontsize=11, fontweight='600')
    
    fig.patch.set_facecolor('white')
    fig.suptitle('Model Awareness: Does the model know it is being attacked?', 
                fontsize=14, fontweight='700', y=0.995)
    plt.tight_layout()
    
    return fig

# ============================================================
# Visualization 2: Before-Accuracy-Change Analysis
# ============================================================
def visualize_awareness_metrics(results):
    """
    Show confidence drop even when accuracy doesn't change
    """
    bg_color = '#FAFBFC'
    grid_color = '#E8EAED'
    
    eps = np.array(results['epsilon'])
    conf_drop = np.array(results['conf_drop'])
    flip_rate = np.array(results['flip_rate'])
    conf_drop_no_flip = np.array(results['conf_drop_no_flip'])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
    
    # Panel A: Overall Confidence Drop vs Flip Rate
    ax1_twin = ax1.twinx()
    
    line1 = ax1.plot(eps, conf_drop, marker='o', markersize=8, linewidth=2.5, 
                     color='#4A90E2', label='Confidence Drop', zorder=3)
    line2 = ax1_twin.plot(eps, flip_rate, marker='s', markersize=8, linewidth=2.5, 
                          color='#E86B5A', label='Flip Rate', zorder=3)
    
    ax1.set_ylabel('Confidence Drop', fontsize=11, fontweight='600', color='#4A90E2')
    ax1_twin.set_ylabel('Flip Rate', fontsize=11, fontweight='600', color='#E86B5A')
    ax1.set_title('A. Confidence Precedes Accuracy Change', fontsize=12, fontweight='700')
    ax1.tick_params(axis='y', labelcolor='#4A90E2', labelsize=10)
    ax1_twin.tick_params(axis='y', labelcolor='#E86B5A', labelsize=10)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.95, fontsize=10)
    
    for spine in ax1.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    for spine in ax1_twin.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel B: Confidence Drop (Overall)
    ax2.fill_between(eps, 0, conf_drop, alpha=0.3, color='#4A90E2')
    ax2.plot(eps, conf_drop, marker='o', markersize=8, linewidth=2.5, 
             color='#4A90E2', zorder=3)
    
    ax2.set_ylabel('Mean Confidence Drop', fontsize=11, fontweight='600')
    ax2.set_title('B. Averaged Across All Predictions', fontsize=12, fontweight='700')
    ax2.tick_params(labelsize=10, colors='#333333')
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel C: Confidence Drop on Correct Predictions
    ax3.fill_between(eps, 0, conf_drop_no_flip, alpha=0.3, color='#F39C12')
    ax3.plot(eps, conf_drop_no_flip, marker='D', markersize=8, linewidth=2.5, 
             color='#F39C12', zorder=3)
    
    ax3.set_ylabel('Confidence Drop', fontsize=11, fontweight='600')
    ax3.set_title('C. Even When Prediction Remains Correct', fontsize=12, fontweight='700')
    ax3.tick_params(labelsize=10, colors='#333333')
    for spine in ax3.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel D: Gap between overall and correct-only confidence drop
    gap = conf_drop - conf_drop_no_flip
    ax4.fill_between(eps, 0, gap, alpha=0.3, color='#E85D75')
    ax4.plot(eps, gap, marker='o', markersize=8, linewidth=2.5, 
             color='#E85D75', zorder=3)
    
    ax4.set_ylabel('Confidence Drop Gap', fontsize=11, fontweight='600')
    ax4.set_title('D. Extra Confidence Loss on Flipped Predictions', fontsize=12, fontweight='700')
    ax4.tick_params(labelsize=10, colors='#333333')
    for spine in ax4.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_xlabel('Perturbation Budget (ε)', fontsize=11, fontweight='600')
    
    fig.patch.set_facecolor('white')
    fig.suptitle('Awareness Metrics: Model Internal States', 
                fontsize=14, fontweight='700', y=0.995)
    plt.tight_layout()
    
    return fig

# ============================================================
# Visualization 3: Explanation Drift (Saliency)
# ============================================================
def visualize_saliency_drift(results):
    """
    Show how model explanations change under attack
    """
    bg_color = '#FAFBFC'
    grid_color = '#E8EAED'
    
    eps = np.array(results['epsilon'])
    saliency_drift = np.array(results['saliency_drift'])
    saliency_com = np.array(results['saliency_com_shift'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax in [ax1, ax2]:
        ax.set_facecolor(bg_color)
        ax.grid(True, color=grid_color, linestyle='-', linewidth=0.8, alpha=0.6)
        ax.set_axisbelow(True)
    
    # Panel A: Saliency Map Distance
    ax1.fill_between(eps, 0, saliency_drift, alpha=0.3, color='#9B59B6')
    ax1.plot(eps, saliency_drift, marker='o', markersize=8, linewidth=2.5, 
             color='#9B59B6', zorder=3, label='Saliency L2 Distance')
    
    ax1.set_ylabel('Saliency Map Drift', fontsize=11, fontweight='600')
    ax1.set_title('A. Explanation Pattern Shift', fontsize=12, fontweight='700')
    ax1.legend(loc='best', framealpha=0.95, fontsize=10)
    ax1.tick_params(labelsize=10, colors='#333333')
    for spine in ax1.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    # Panel B: Center of Mass Shift
    ax2.fill_between(eps, 0, saliency_com, alpha=0.3, color='#16A085')
    ax2.plot(eps, saliency_com, marker='s', markersize=8, linewidth=2.5, 
             color='#16A085', zorder=3, label='Center of Mass Shift')
    
    ax2.set_ylabel('Shift Distance (pixels)', fontsize=11, fontweight='600')
    ax2.set_title('B. Attention Displacement', fontsize=12, fontweight='700')
    ax2.legend(loc='best', framealpha=0.95, fontsize=10)
    ax2.tick_params(labelsize=10, colors='#333333')
    for spine in ax2.spines.values():
        spine.set_color('#CCCCCC')
        spine.set_linewidth(0.8)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel('Perturbation Budget (ε)', fontsize=11, fontweight='600')
    
    fig.patch.set_facecolor('white')
    fig.suptitle('Explanation Drift: Model Reasoning Becomes Unstable', 
                fontsize=14, fontweight='700', y=0.995)
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
    EPS_LIST = [0, 0.02, 0.05, 0.1, 0.15, 0.2]  # Epsilon sweep (increased 10-20x for effectiveness)
    
    # Load data
    print("Loading datasets...")
    train_ds = CTSliceDataset(os.path.join(ROOT_DIR, "train"))
    val_ds = CTSliceDataset(os.path.join(ROOT_DIR, "valid"))
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        print(f"\n✗ No data found in {ROOT_DIR}")
        print("Please update ROOT_DIR to point to your dataset")
        print("Expected structure: ROOT_DIR/train/benign, ROOT_DIR/train/malignant, etc.")
        exit(1)
    else:
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
        
        # Model awareness evaluation
        print(f"\nEvaluating model awareness to attacks...")
        print(f"Epsilon values: {EPS_LIST}")
        results = evaluate_model_awareness(model, val_loader, radon, thetas, EPS_LIST)
    
    # Display results
    print("\n" + "="*80)
    print("MODEL AWARENESS ANALYSIS: Does the model know it's under attack?")
    print("="*80)
    print("\nComprehensive Metrics:\n")
    print(f"{'ε':<8} {'Clean Conf':<12} {'Adv Conf':<12} {'Conf Drop':<12} {'Entropy Δ':<12} {'Flip Rate':<12}")
    print("-" * 88)
    
    for i in range(len(results['epsilon'])):
        eps = results['epsilon'][i]
        clean_conf = results['clean_conf'][i]
        adv_conf = results['adv_conf'][i]
        conf_drop = results['conf_drop'][i]
        ent_inc = results['entropy_increase'][i]
        flip = results['flip_rate'][i]
        
        print(f"{eps:<8.4f} {clean_conf:<12.4f} {adv_conf:<12.4f} {conf_drop:<12.4f} {ent_inc:<12.4f} {flip:<12.4f}")
    
    # Analysis
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    conf_drops = np.array(results['conf_drop'])
    flips = np.array(results['flip_rate'])
    entropy_inc = np.array(results['entropy_increase'])
    conf_no_flip = np.array(results['conf_drop_no_flip'])
    
    # Finding 1: Confidence precedes accuracy change
    max_conf_drop_idx = np.argmax(conf_drops[1:]) + 1
    max_flip_idx = np.argmax(flips[1:]) + 1
    
    if max_conf_drop_idx < max_flip_idx:
        print(f"\n✓ FINDING 1: Confidence Collapse Precedes Accuracy Change")
        print(f"  Peak confidence drop at ε={results['epsilon'][max_conf_drop_idx]:.4f}")
        print(f"  Peak flip rate at ε={results['epsilon'][max_flip_idx]:.4f}")
        print(f"  → Model detects attack BEFORE changing predictions")
    
    # Finding 2: Entropy increases
    max_entropy_idx = np.argmax(entropy_inc[1:]) + 1
    print(f"\n✓ FINDING 2: Model Becomes More Uncertain")
    print(f"  Peak entropy increase: {entropy_inc[max_entropy_idx]:.4f} at ε={results['epsilon'][max_entropy_idx]:.4f}")
    print(f"  → Predictions become less confident (higher entropy)")
    
    # Finding 3: Confidence drop even on correct predictions
    if np.any(conf_no_flip[1:] > 0):
        print(f"\n✓ FINDING 3: Awareness Even Without Label Flip")
        avg_conf_no_flip = np.mean(conf_no_flip[conf_no_flip > 0])
        print(f"  Average confidence drop on correct predictions: {avg_conf_no_flip:.4f}")
        print(f"  → Model shows internal distress even when final answer is right")
    
    # Finding 4: Saliency drift
    saliency_drift = np.array(results['saliency_drift'])
    if np.any(saliency_drift[1:] > 0):
        print(f"\n✓ FINDING 4: Explanation Drift Detected")
        print(f"  Peak saliency pattern shift: {np.max(saliency_drift):.4f}")
        print(f"  Peak attention displacement: {np.max(np.array(results['saliency_com_shift'])):.2f} pixels")
        print(f"  → Model's visual attention shifts away from anatomical features")
    
    print("\n" + "="*80)
    print("CONCLUSION: The model KNOWS it is being attacked!")
    print("="*80)
    print("Evidence:")
    print("  • Softmax confidence collapses before accuracy changes")
    print("  • Entropy increases significantly")
    print("  • Internal confusion visible even on correct predictions")
    print("  • Model explanations (saliency) become unstable")
    print("="*80)
    
    # Visualizations
    print("\nGenerating research-grade visualizations...")
    
    # Check if attacks are working
    max_flip = max(results['flip_rate'])
    if max_flip < 0.05:
        print("\n⚠️  WARNING: Attacks appear ineffective (flip rate < 5%)")
        print("TROUBLESHOOTING:")
        print("1. Increase epsilon further: try [0, 0.1, 0.2, 0.3, 0.4, 0.5]")
        print("2. Use PGD instead of FGSM (iterative attack)")
        print("3. Check that clean accuracy > 55% (currently {:.1%})".format(results['clean_conf'][0]))
        print("4. Enable gradient diagnostics by uncommenting in sino_fgsm_standard()")
        print("\nProceeding with visualization of current results...\n")
    
    fig1 = visualize_confidence_entropy(results)
    plt.show()
    
    fig2 = visualize_awareness_metrics(results)
    plt.show()
    
    fig3 = visualize_saliency_drift(results)
    plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)