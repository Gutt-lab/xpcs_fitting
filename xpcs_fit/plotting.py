"""Diagnostic plots for the BO/VarPro-vs-CF comparison."""

from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from .physics import compute_channel_residuals, g2_kww_model


def plot_joint_vs_independent_curves(tau_list, g2_list, sig_list, qs,
                                      D_true, beta_true, B_true, alpha_true,
                                      D_joint, alpha_joint, beta_joint, B_joint,
                                      D_cf, alpha_cf, beta_cf, B_cf,
                                      filename_stem):
    """g2(tau) per q-channel: data points, ground truth, joint-BO fit
    (solid), and independent-CF fit (dotted)."""
    n_q = len(qs)
    colors = cm.viridis(np.linspace(0, 0.92, n_q))
    fig, ax = plt.subplots(figsize=(12, 8))

    for k in range(n_q):
        tau, g2, sig, q = tau_list[k], g2_list[k], sig_list[k], qs[k]
        if len(tau) == 0:
            continue
        c = colors[k]
        tau_fine = np.logspace(np.log10(np.min(tau)), np.log10(np.max(tau)), 400)

        y_joint = g2_kww_model(tau_fine, q, D_joint[k], alpha_joint[k], beta_joint[k], B_joint[k])
        y_true = g2_kww_model(tau, q, D_true[k], alpha_true[k], beta_true[k], B_true[k])

        ax.errorbar(tau, g2, yerr=sig, fmt=".", color=c, ms=4, alpha=0.25, ecolor=c,
                     elinewidth=0.6, capsize=0)
        ax.plot(tau, y_true, "s-", color=c, lw=2.2,
                label=f"True g2 q={q:.4f} D={D_true[k]:.2e}")
        ax.plot(tau_fine, y_joint, "-", color=c, lw=2.2,
                label=f"joint BO q={q:.4f} D={D_joint[k]:.2e}")

        if np.isfinite(D_cf[k]) and np.isfinite(beta_cf[k]) and np.isfinite(B_cf[k]):
            y_cf = g2_kww_model(tau_fine, q, D_cf[k], alpha_cf[k], beta_cf[k], B_cf[k])
            ax.plot(tau_fine, y_cf, ":", color=c, lw=1.8, alpha=0.9,
                    label=f"CF q={q:.4f} D={D_cf[k]:.2e}")

    ax.set_xscale("log")
    ax.set(xlabel="τ (us)", ylabel="g₂(τ)",
           title=f"Joint BO (solid) vs CF (dotted) — {Path(filename_stem).name}")
    ax.legend(fontsize=6, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()


def plot_D_vs_q_three_way(qs, D_true, D_joint, err_joint_D, D_cf, err_cf_D, filename_stem):
    """D(q): ground truth vs joint-smooth BO vs independent CF."""
    q_nm = np.asarray(qs)

    def conv(D, err):
        D = np.asarray(D, float)
        err = None if err is None else np.nan_to_num(np.asarray(err, float), nan=0.0, posinf=0.0)
        return D, err

    D_j, e_j = conv(D_joint, err_joint_D)
    D_c, e_c = conv(D_cf, err_cf_D)
    true = D_true

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.errorbar(q_nm, D_j, yerr=e_j, fmt="o-", ms=5, capsize=3, elinewidth=1, label="joint-smooth BO")
    ax.errorbar(q_nm, D_c, yerr=e_c, fmt="s-.", ms=5, capsize=3, elinewidth=1, color="k", label="independent CF")
    ax.plot(q_nm, true, "-o", color="r", ms=5, label="True Data")

    # Focus the y-axis on the data points themselves, ignoring how far
    # the CF error bars stretch (CF error isn't the point of this plot).
    all_D = np.concatenate([D_j, D_c])
    all_D = all_D[np.isfinite(all_D)]
    y_lo, y_hi = np.min(all_D), np.max(all_D)
    pad = 0.15 * (y_hi - y_lo) if y_hi > y_lo else 1.0
    ax.set_ylim(y_lo - pad, y_hi + pad)

    ax.set_xlabel("q (nm⁻¹)"); ax.set_ylabel("D (nm²/µs)")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    ax.set_title(f"Diffusion Coefficient D(q) — Joint-Smooth BO {Path(filename_stem).name}")
    plt.tight_layout(); plt.show()


def plot_joint_vs_independent_parameters(qs, D_true, beta_true, B_true, alpha_true,
                                          D_joint, alpha_joint, beta_joint, B_joint,
                                          D_cf, alpha_cf, beta_cf, B_cf,
                                          filename_stem, err_joint=None, err_cf=None):
    """2x2 panel of D, alpha, beta, and tau_c vs q for truth/BO/CF."""
    q_nm = np.asarray(qs)

    tau_joint = 1.0 / (D_joint * np.asarray(qs) ** 2)
    tau_cf = 1.0 / (D_cf * np.asarray(qs) ** 2)
    tau_c = 1.0 / (D_true * np.asarray(qs) ** 2)

    def col(err, i):
        if err is None:
            return None
        return np.nan_to_num(err[:, i], nan=0.0, posinf=0.0)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    panels = [
        (axes[0, 0], D_true, D_joint, D_cf, "D (nm²/us)", col(err_joint, 0), col(err_cf, 0)),
        (axes[0, 1], alpha_true, alpha_joint, alpha_cf, "alpha", col(err_joint, 1), col(err_cf, 1)),
        (axes[1, 0], beta_true, beta_joint, beta_cf, "beta", col(err_joint, 2), col(err_cf, 2)),
        (axes[1, 1], tau_c, tau_joint, tau_cf, "τ_c (us)  [=1/(Dq²)]", None, None),
    ]
    for ax, true, joint, cf, label, e_jnt, e_cf in panels:
        ax.errorbar(q_nm, joint, yerr=e_jnt, fmt="o-", ms=5,
                     capsize=3, elinewidth=1, label="joint-smooth BO")
        ax.errorbar(q_nm, cf, yerr=e_cf, fmt="s-.", ms=5, color="k", alpha=0.9,
                     capsize=3, elinewidth=1, label="independent CF")
        ax.plot(q_nm, true, "-o", color="r", ms=5, label="True Data")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[1, 1].set_yscale("log")
    axes[1, 0].set_xlabel("q (nm⁻¹)"); axes[1, 1].set_xlabel("q (nm⁻¹)")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(f"Joint-smooth vs CF parameters — {Path(filename_stem).name}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_lambda_scan(scan_df, filename_stem):
    """Held-out validation RMSE and the data-fit/penalty tradeoff vs lambda."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    lam = scan_df["lambda"].to_numpy()
    lam_plot = np.where(lam <= 0, lam.max() * 1e-3 if lam.max() > 0 else 1e-8, lam)  # log-safe for lambda=0

    ax1.plot(lam_plot, scan_df["val_RMSE"], "o-", color="tab:red")
    ax1.set_xscale("log"); ax1.set_xlabel("lambda (lambda=0 shown at left edge)")
    ax1.set_ylabel("held-out validation RMSE"); ax1.grid(alpha=0.25)
    ax1.set_title("Lambda selection curve")

    ax2.plot(lam_plot, scan_df["training_data_sse"], "o-", label="training data SSE")
    ax2.plot(lam_plot, scan_df["penalty"], "s-", label="curvature penalty")
    ax2.set_xscale("log"); ax2.set_yscale("log"); ax2.set_xlabel("lambda")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    ax2.set_title("Data fit vs. penalty tradeoff")

    fig.suptitle(f"{Path(filename_stem).name}", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_residuals(tau_list, g2_list, qs, D_joint,
                    alpha_joint, beta_joint, B_joint, D_cf,
                    alpha_cf, beta_cf, B_cf, filename_stem):
    """Per-q-channel data-minus-model residuals for joint-BO vs CF."""
    n_q = len(qs)
    fig, axes = plt.subplots(n_q, 1, figsize=(9, 2.0 * n_q), sharex=True)
    if n_q == 1:
        axes = [axes]
    for k in range(n_q):
        resid_joint = compute_channel_residuals(tau_list[k], g2_list[k], qs[k],
                                                  D_joint[k], alpha_joint[k], beta_joint[k], B_joint[k])
        resid_cf = compute_channel_residuals(tau_list[k], g2_list[k], qs[k],
                                              D_cf[k], alpha_cf[k], beta_cf[k], B_cf[k])
        ax = axes[k]
        ax.axhline(0, color="gray", lw=0.8)
        ax.plot(tau_list[k], resid_joint, "o", color="r", ms=4, alpha=0.7, label="joint BO")
        ax.plot(tau_list[k], resid_cf, ".", color="k", ms=4, alpha=0.5, label="CF")
        ax.set_xscale("log")
        ax.set_ylabel(f"q={qs[k]:.4f}\nresid")
        if k == 0:
            ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("τ (us)")
    fig.suptitle(f"Per-channel fit residuals — {Path(filename_stem).name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.show()
