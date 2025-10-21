#!/usr/bin/env python3
"""
ddpm_pac_bayes_demo.py

Multi-step DDPM-style PAC-Bayes demo (toy 2D data).

Requirements:
    pip install torch numpy matplotlib

Run:
    python ddpm_pac_bayes_demo.py
"""

import math
import argparse
import numpy as np
import os
import torch
from torch import nn
import matplotlib.pyplot as plt

def parse_args():
    p = argparse.ArgumentParser(description="Multi-step DDPM-style PAC-Bayes demo (toy 2D)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-data", type=int, default=800)
    p.add_argument("--T", type=int, default=30, help="diffusion steps")
    p.add_argument("--hidden", type=int, default=64, help="hidden units in MLP")
    p.add_argument("--time-emb-dim", type=int, default=8)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--mc-samples", type=int, default=10, help="posterior samples per step")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--sigma-p", type=float, default=1.0, help="prior std")
    p.add_argument("--mode", choices=["pac-bayes","surrogate"], default="pac-bayes",
                   help="optimization mode: 'pac-bayes' minimizes bound; 'surrogate' minimizes emp + lambda*KL")
    p.add_argument("--lambda-kl", type=float, default=1e-3, help="weight for KL in surrogate mode")
    p.add_argument("--adapter-dim", type=int, default=0, help="if >0, posterior only covers adapter params (low-dim)")
    p.add_argument("--save", type=str, default="posterior.pt")
    return p.parse_args()

def make_2d_mog(n_samples=1000):
    centers = np.array([[2., 0.], [-2., 0.], [0., 2.]])
    k = centers.shape[0]
    labels = np.random.randint(0, k, size=n_samples)
    xs = centers[labels] + 0.25 * np.random.randn(n_samples, 2)
    return xs.astype(np.float32)

def build_schedule(T, beta_start=1e-4, beta_end=0.02):
    betas = np.linspace(beta_start, beta_end, T, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas)
    return betas, alphas, alpha_bars

def q_sample_batch(x0, t_indices, alpha_bars_t, noise=None):
    # x0: (B,2), t_indices: (B,) long, alpha_bars_t: np array or tensor (T,)
    B = x0.shape[0]
    if noise is None:
        noise = torch.randn_like(x0)
    # gather sqrt_ab and sqrt_1_ab per-sample
    if isinstance(alpha_bars_t, np.ndarray):
        alpha_bars_t = torch.from_numpy(alpha_bars_t).to(x0.device)
    sqrt_ab = torch.sqrt(alpha_bars_t[t_indices]).unsqueeze(-1)  # (B,1)
    sqrt_1_ab = torch.sqrt(1.0 - alpha_bars_t[t_indices]).unsqueeze(-1)
    return sqrt_ab * x0 + sqrt_1_ab * noise, noise

def param_shapes(in_dim, hidden, time_emb_dim, out_dim=2):
    shapes = []
    shapes.append(("Wt", (time_emb_dim, 1)))
    shapes.append(("bt", (time_emb_dim,)))
    shapes.append(("W1", (hidden, in_dim + time_emb_dim)))
    shapes.append(("b1", (hidden,)))
    shapes.append(("W2", (hidden, hidden)))
    shapes.append(("b2", (hidden,)))
    shapes.append(("W3", (out_dim, hidden)))
    shapes.append(("b3", (out_dim,)))
    return shapes

def flatten_shapes(shapes):
    return sum(int(np.prod(s[1])) for s in shapes)

def unpack_theta(theta_flat, shapes):
    out = {}
    idx = 0
    for name, shape in shapes:
        size = int(np.prod(shape))
        seg = theta_flat[idx: idx + size]
        out[name] = seg.view(shape)
        idx += size
    return out

def forward_theta(theta_flat, x_t, t_indices, shapes, T):
    # x_t: (B,2), t_indices: (B,) long
    params = unpack_theta(theta_flat, shapes)
    # time embedding via linear projection (Wt, bt)
    Wt = params["Wt"]
    bt = params["bt"]
    te = (t_indices.float().unsqueeze(-1) / float(T))  # (B,1)
    te_proj = te @ Wt.t() + bt  # (B, time_emb_dim)
    inp = torch.cat([x_t, te_proj], dim=1)  # (B, 2 + time_emb_dim)
    h = torch.relu(inp @ params["W1"].t() + params["b1"])
    h = torch.relu(h @ params["W2"].t() + params["b2"])
    eps_hat = h @ params["W3"].t() + params["b3"]
    return eps_hat

def kl_diag_gaussians(mu, sigma, sigma_p2):
    var_q = sigma ** 2
    term1 = (var_q + mu ** 2) / sigma_p2
    kl = 0.5 * (term1 - 1 - 2.0 * torch.log(sigma / math.sqrt(sigma_p2))).sum()
    return kl

def mc_empirical_ddpm_loss(mu, sigma, x_batch, shapes, alpha_bars, T, S=3):
    B = x_batch.shape[0]
    device = x_batch.device
    losses = []
    for s in range(S):
        eps_theta = torch.randn_like(mu)
        theta = mu + sigma * eps_theta
        t_indices = torch.randint(low=0, high=T, size=(B,), device=device)
        x_t, noise = q_sample_batch(x_batch, t_indices, alpha_bars)
        eps_hat = forward_theta(theta, x_t, t_indices, shapes, T)
        mse = ((eps_hat - noise) ** 2).mean()
        losses.append(mse)
    return torch.stack(losses).mean()

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    # Data
    X = make_2d_mog(args.n_data)
    X_torch = torch.from_numpy(X).to(device)
    N = X_torch.shape[0]

    # Schedule
    betas, alphas, alpha_bars = build_schedule(args.T)
    alpha_bars_t = alpha_bars  # numpy array

    # Model shapes (smaller by default to keep KL manageable)
    shapes = param_shapes(in_dim=2, hidden=args.hidden, time_emb_dim=args.time_emb_dim, out_dim=2)
    D = flatten_shapes(shapes)
    print(f"Flattened parameter dim D = {D}")

    # Adapter option: if adapter_dim > 0, only that many parameters are free (posterior size)
    adapter_dim = int(args.adapter_dim)
    use_adapter = adapter_dim > 0 and adapter_dim < D
    if use_adapter:
        print(f"Using adapter posterior of dim {adapter_dim} (rest fixed at prior mean 0)")
        # Posterior mu/rho live in adapter space; we'll embed into full D by placing adapter at front
        post_dim = adapter_dim
        def embed_to_full(theta_post):
            # theta_post: (post_dim,)
            full = torch.zeros(D, device=theta_post.device)
            full[:post_dim] = theta_post
            return full
    else:
        post_dim = D

    # Variational parameters (mu, rho) of length post_dim
    mu = torch.zeros(post_dim, device=device, requires_grad=True)
    rho = torch.full((post_dim,), -4.0, device=device, requires_grad=True)  # log std
    optimizer = torch.optim.Adam([mu, rho], lr=args.lr)

    sigma_p = args.sigma_p
    sigma_p2 = sigma_p ** 2

    # Clip or scale empirical loss into [0,1] for PAC-Bayes bound usage (heuristic)
    B_scale = 1.0 #8.0

    history = {"emp": [], "kl": [], "bound": []}

    logterm = math.log(2.0 * math.sqrt(N) / 0.05)  # delta=0.05 fixed for demo

    for step in range(args.steps):
        idx = np.random.choice(N, size=min(args.batch, N), replace=False)
        x_batch = X_torch[idx]

        sigma = torch.exp(rho)
        # compute empirical loss
        if use_adapter:
            # sample in adapter space, embed to full parameter vector
            def sample_theta_full():
                eps = torch.randn(post_dim, device=device)
                theta_post = mu + sigma * eps
                return embed_to_full(theta_post)
            # Monte-Carlo manual
            losses = []
            for s in range(args.mc_samples):
                theta_full = sample_theta_full()
                t_indices = torch.randint(low=0, high=args.T, size=(x_batch.shape[0],), device=device)
                x_t, noise = q_sample_batch(x_batch, t_indices, alpha_bars_t)
                eps_hat = forward_theta(theta_full, x_t, t_indices, shapes, args.T)
                losses.append(((eps_hat - noise) ** 2).mean())
            emp_loss = torch.stack(losses).mean()
            # KL computed in adapter space vs prior projected variance: assume prior std for adapter dims = sigma_p
            kl = kl_diag_gaussians(mu, sigma, sigma_p2)
        else:
            emp_loss = mc_empirical_ddpm_loss(mu, sigma, x_batch, shapes, alpha_bars_t, args.T, S=args.mc_samples)
            kl = kl_diag_gaussians(mu, sigma, sigma_p2)

        # scale
        hatR = emp_loss / B_scale
        if args.mode == "pac-bayes":
            pac_bound = hatR + torch.sqrt((kl + logterm) / (2.0 * N))
            obj = pac_bound
        else:  # surrogate
            obj = emp_loss + args.lambda_kl * kl
            pac_bound = hatR + torch.sqrt((kl + logterm) / (2.0 * N))  # still track bound for logging

        optimizer.zero_grad()
        obj.backward()
        # optional gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([mu, rho], max_norm=5.0)
        optimizer.step()

        if step % 40 == 0 or step == args.steps - 1:
            print(f"step {step:4d} emp_loss {emp_loss.item():.4f} kl {kl.item():.2f} pac_bound {pac_bound.item():.4f}")

        history["emp"].append(emp_loss.item())
        history["kl"].append(kl.item())
        history["bound"].append(pac_bound.item())

    # Save posterior params
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    torch.save({"mu": mu.detach().cpu(), "rho": rho.detach().cpu(), "shapes": shapes,
                "use_adapter": use_adapter, "adapter_dim": adapter_dim,
                "args": vars(args)}, args.save)
    print(f"Saved posterior to {args.save}")

    # Plots
    plt.figure(figsize=(10,4))
    plt.subplot(3,1,1)
    plt.plot(history["emp"])
    plt.title("Empirical MSE (predicted noise)")
    plt.xlabel("step")
    plt.ylabel("MSE")

    plt.subplot(3,1,2)
    plt.plot(history["kl"], label="KL")
    plt.legend()
    plt.title("KL divergence")
    plt.xlabel("step")
    
    plt.subplot(3,1,3)
    plt.plot(history["bound"], label="PAC-Bayes bound (scaled)")
    plt.legend()
    plt.title("PAC-Bayes bound")
    plt.xlabel("step")

    plt.tight_layout()
    plt.show()

    # Visualize denoising field for one posterior sample (embed if adapter)
    with torch.no_grad():
        sigma = torch.exp(rho)
        eps_theta = torch.randn_like(mu)
        theta_post = mu + sigma * eps_theta
        if use_adapter:
            theta_full = embed_to_full(theta_post.to(device))
        else:
            theta_full = theta_post.to(device)
        # visualize at middle time
        t_vis = args.T // 2
        grid_x = np.linspace(-3, 3, 20)
        grid_y = np.linspace(-3, 3, 20)
        pts = np.array([[xx, yy] for xx in grid_x for yy in grid_y], dtype=np.float32)
        pts_t = torch.from_numpy(pts).to(device)
        t_vis_vec = torch.full((pts_t.shape[0],), t_vis, dtype=torch.long, device=device)
        noise = torch.randn_like(pts_t)
        x_t_vis, _ = q_sample_batch(pts_t, t_vis_vec, alpha_bars_t, noise=noise)
        eps_hat = forward_theta(theta_full, x_t_vis, t_vis_vec, shapes, args.T).cpu().numpy()
        ab = float(alpha_bars_t[t_vis])
        sqrt_ab = math.sqrt(ab)
        sqrt_1_ab = math.sqrt(1.0 - ab)
        x0_hat = (x_t_vis.cpu().numpy() - (sqrt_1_ab) * eps_hat) / sqrt_ab

    plt.figure(figsize=(5,5))
    plt.quiver(x_t_vis.cpu().numpy()[:,0], x_t_vis.cpu().numpy()[:,1],
               x0_hat[:,0]-x_t_vis.cpu().numpy()[:,0],
               x0_hat[:,1]-x_t_vis.cpu().numpy()[:,1],
               angles='xy', scale_units='xy', scale=1)
    plt.scatter(X[:,0], X[:,1], s=6, alpha=0.6)
    plt.title("Denoising vectors (one posterior sample)")
    plt.xlim(-4,4); plt.ylim(-4,4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()