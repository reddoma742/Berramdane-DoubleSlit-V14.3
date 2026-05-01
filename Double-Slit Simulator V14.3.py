# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   Berramdane Double-Slit Simulator V14.3 — Colab-Ready Laboratory Edition   ║
║   Maxwell-Boltzmann · Slit Thickness · Vector Potential A · auto_calibrate  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from scipy.signal import find_peaks
from scipy.optimize import fsolve
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from ipywidgets import (interact, FloatSlider, Checkbox, IntSlider,
                            Dropdown, Button, Output, IntText, HBox, Label)
    from IPython.display import display
    WIDGETS_OK = True
except ImportError:
    WIDGETS_OK = False
    print("⚠️ ipywidgets not installed. Run: !pip install ipywidgets -q")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────
h       = 6.626e-34
hbar    = h / (2 * np.pi)
m_e     = 9.109e-31
c       = 2.998e8
k_B     = 1.381e-23
e_c     = 1.602e-19

REF_JONSSON = {'v': 70000., 'a': 0.3e-6, 'd': 1.0e-6,
               'L': 0.35,   'fringe_mm': 0.18}

# ──────────────────────────────────────────────────────────────────────────────
# MAXWELL-BOLTZMANN
# ──────────────────────────────────────────────────────────────────────────────
def maxwell_boltzmann_samples(v_mean, T_source, n_samples, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    T_fitted = np.pi * m_e * v_mean**2 / (8 * k_B)
    T_eff = T_fitted * (1 + T_source / 1e4)
    sigma_mb = np.sqrt(k_B * T_eff / m_e)
    samples = sigma_mb * np.sqrt(rng.chisquare(3, size=n_samples))
    samples = samples[(samples > 0.2*v_mean) & (samples < 2.5*v_mean)]
    if len(samples) < 2:
        samples = np.array([v_mean])
    def mb_pdf(v, T):
        A = 4 * np.pi * (m_e / (2 * np.pi * k_B * T))**1.5
        return A * v**2 * np.exp(-m_e * v**2 / (2 * k_B * T))
    weights = mb_pdf(samples, T_eff)
    weights /= np.sum(weights)
    return samples, T_fitted, weights

def de_broglie_relativistic(v):
    beta = np.clip(v / c, 0, 0.9999)
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return h / (gamma * m_e * v)

# ──────────────────────────────────────────────────────────────────────────────
# SLIT THICKNESS MODEL
# ──────────────────────────────────────────────────────────────────────────────
def slit_transmission(y, d_slit, a_width, t_slit, v, N_slits,
                      n_wall=1.5, alpha_abs=1e6, back_r0=0.05):
    lam = de_broglie_relativistic(v)
    k = 2 * np.pi / lam
    in_slit = np.zeros(len(y), dtype=bool)
    for s in range(N_slits):
        yc = (s - (N_slits - 1) / 2.0) * d_slit
        mask = np.abs(y - yc) <= a_width / 2
        in_slit |= mask
    dist_to_edge = np.full(len(y), a_width / 2)
    for s in range(N_slits):
        yc = (s - (N_slits - 1) / 2.0) * d_slit
        left = yc - a_width / 2
        right = yc + a_width / 2
        mask = (y >= left) & (y <= right)
        if np.any(mask):
            d_left = np.abs(y[mask] - left)
            d_right = np.abs(y[mask] - right)
            dist_to_edge[mask] = np.minimum(d_left, d_right)
    sigma_wall = a_width * 0.15
    wall_prox = np.exp(-dist_to_edge**2 / (2 * sigma_wall**2))
    wall_prox[~in_slit] = 0.0
    if t_slit > 0:
        T_amp = np.where(in_slit, np.exp(-alpha_abs * t_slit * wall_prox), 0.0)
    else:
        T_amp = np.where(in_slit, 1.0, 0.0)
    phi_wall = np.where(in_slit, k * (n_wall - 1.0) * t_slit * wall_prox, 0.0)
    v_threshold = 15_000.0
    P_back = back_r0 * np.exp(-v / v_threshold)
    P_back = float(np.clip(P_back, 0, 0.5))
    return T_amp, phi_wall, P_back

# ──────────────────────────────────────────────────────────────────────────────
# VECTORISED PATH INTEGRAL
# ──────────────────────────────────────────────────────────────────────────────
def feynman_vectorized(x, lam, L, a_width, d_slit, N_slits,
                       n_paths, rng_seed,
                       edge_strength=0.0, edge_angle=0.2,
                       A_y=0.0,
                       phi_wall_fn=None,
                       T_amp_fn=None):
    rng = np.random.default_rng(rng_seed)
    k = 2 * np.pi / lam
    half_a = a_width / 2.0
    all_y, all_T, all_pw = [], [], []
    for s in range(N_slits):
        yc = (s - (N_slits - 1) / 2.0) * d_slit
        y_s = rng.uniform(yc - half_a, yc + half_a, n_paths)
        all_y.append(y_s)
        dist_edge = np.minimum(np.abs(y_s - (yc - half_a)),
                               np.abs(y_s - (yc + half_a)))
        if edge_strength > 0:
            near = dist_edge < a_width * 0.2
            T_edge = np.where(near,
                              np.exp(-edge_strength * (a_width*0.2 - dist_edge)),
                              1.0)
        else:
            T_edge = np.ones(n_paths)
        all_T.append(T_edge)
        if phi_wall_fn is not None:
            pw = phi_wall_fn(y_s, yc)
        else:
            pw = np.zeros(n_paths)
        all_pw.append(pw)
    y_all = np.concatenate(all_y)
    T_all = np.concatenate(all_T)
    pw_all = np.concatenate(all_pw)
    if edge_strength > 0:
        near_mask = np.zeros(len(y_all), dtype=bool)
        for s in range(N_slits):
            yc = (s - (N_slits - 1) / 2.0) * d_slit
            left = yc - half_a
            right = yc + half_a
            dist_l = np.abs(y_all - left)
            dist_r = np.abs(y_all - right)
            near_mask |= (np.minimum(dist_l, dist_r) < a_width * 0.2)
        phi_edge = np.where(near_mask,
                            rng.normal(0, edge_angle * edge_strength, len(y_all)),
                            0.0)
    else:
        phi_edge = np.zeros(len(y_all))
    y_col = y_all[:, np.newaxis]
    x_row = x[np.newaxis, :]
    r = np.sqrt(L**2 + (x_row - y_col)**2)
    if A_y != 0.0:
        phi_AB = (e_c / hbar) * A_y * (x_row - y_col)
    else:
        phi_AB = 0.0
    phase_total = k * r + phi_AB + pw_all[:, np.newaxis] + phi_edge[:, np.newaxis]
    amplitude = T_all[:, np.newaxis] / r
    psi = np.sum(amplitude * np.exp(1j * phase_total), axis=0)
    I = np.abs(psi)**2
    return I / np.max(I) if np.max(I) > 0 else I

# ──────────────────────────────────────────────────────────────────────────────
# QUANTUM SYSTEM
# ──────────────────────────────────────────────────────────────────────────────
class QuantumSystem:
    def __init__(self, N):
        self.N = N
    def pure_state(self):
        psi = np.ones(self.N, dtype=complex) / np.sqrt(self.N)
        return np.outer(psi, psi.conj())
    def lindblad(self, rho, gamma, n_steps=60):
        N = self.N
        dt = 1.0 / n_steps
        decay = np.exp(-gamma * dt)
        D = np.full((N, N), decay, dtype=complex)
        np.fill_diagonal(D, 1.0)
        rho_t = rho.copy().astype(complex)
        for _ in range(n_steps):
            rho_t = rho_t * D
        return rho_t
    def coherence(self, rho):
        N = self.N
        if N < 2: return 0.0
        mask = ~np.eye(N, dtype=bool)
        return float(np.sum(np.abs(rho[mask])) / (N * (N - 1)))
    def purity(self, rho):
        return float(np.real(np.trace(rho @ rho)))
    def entropy(self, rho):
        ev = np.linalg.eigvalsh(rho)
        ev = ev[ev > 1e-15]
        return float(-np.sum(ev * np.log(ev)))
    def wigner_n2(self, rho):
        if self.N != 2: return None, None, None
        a, b = np.real(rho[0,0]), np.real(rho[1,1])
        rc = np.real(rho[0,1])
        ic = np.imag(rho[0,1])
        q = np.linspace(-3, 3, 100)
        p = np.linspace(-3, 3, 100)
        Q, P = np.meshgrid(q, p)
        W = (1/(2*np.pi)) * (1 + 2*rc*np.cos(Q)*np.cos(P)
                             - 2*ic*np.sin(Q)*np.cos(P)
                             + (a - b)*np.sin(P))
        return q, p, W

# ──────────────────────────────────────────────────────────────────────────────
# DETECTOR
# ──────────────────────────────────────────────────────────────────────────────
class Detector:
    def __init__(self, eta, dark_rate, T, readout_noise_mm, pixel_um, seed):
        self.eta = eta
        self.dcr = dark_rate
        self.sigma = np.sqrt((pixel_um * 1e-6)**2 + (readout_noise_mm * 1e-3)**2)
        self.V_jn = np.sqrt(4 * k_B * T * 1e6 * 1e6)
        self.rng = np.random.default_rng(seed)
    def detect(self, x, I_prob, n):
        I_norm = np.maximum(I_prob, 0)
        if np.sum(I_norm) > 0:
            I_norm /= np.sum(I_norm)
        cdf = np.cumsum(I_norm)
        u = self.rng.uniform(0, 1, n)
        idx = np.clip(np.searchsorted(cdf, u), 0, len(x)-1)
        mask = self.rng.random(n) < self.eta
        x_det = x[idx[mask]]
        total_sigma = np.sqrt(self.sigma**2 +
                              (self.V_jn / 1e6 * (x[-1]-x[0]))**2)
        x_det += self.rng.normal(0, total_sigma, len(x_det))
        x_det = np.clip(x_det, x[0], x[-1])
        n_dark = self.rng.poisson(self.dcr * n)
        x_dark = self.rng.uniform(x[0], x[-1], n_dark)
        x_all = np.concatenate([x_det, x_dark])
        counts, _ = np.histogram(x_all, bins=len(x), range=(x[0], x[-1]))
        return counts.astype(float), len(x_det), n_dark

# ──────────────────────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_visibility(x, I):
    pk, _ = find_peaks(I, distance=max(5, len(x)//40))
    if len(pk) < 2: return 0.0
    Imax = np.max(I[pk])
    ctr = np.where(np.abs(x) < np.max(np.abs(x))*0.15)[0]
    Imin = np.min(I[ctr]) if len(ctr) > 0 else np.min(I)
    denom = Imax + Imin
    return float((Imax - Imin)/denom) if denom > 0 else 0.0

def normalize(arr):
    mx = np.max(arr)
    return arr / mx if mx > 0 else arr

# ──────────────────────────────────────────────────────────────────────────────
# MAIN SIMULATION V14.3 (FIXED)
# ──────────────────────────────────────────────────────────────────────────────
def run_v14_3(
    v_mean          = 70_000.,
    T_source        = 1000.,
    n_vel_samples   = 50,
    L_mm            = 350.,
    a_um            = 0.3,
    d_um            = 1.0,
    N_slits         = 2,
    t_slit_nm       = 0.,
    n_wall          = 1.5,
    back_r0         = 0.05,
    gamma_meas      = 0.0,
    K_input         = 0.0,
    edge_strength   = 0.0,
    edge_angle      = 0.2,
    A_y             = 0.0,
    stray_rms       = 0.0,
    n_particles     = 5000,
    eta             = 0.85,
    dark_rate       = 0.02,
    det_T           = 300.,
    readout_nm      = 500.,
    pixel_um        = 5.0,
    pattern_mode    = 'Mixed',
    x_limit_mm      = None,
    n_paths         = 100,
    auto_calibrate  = False,
    rng_seed        = 42,
):
    L = L_mm / 1e3
    a_width = a_um * 1e-6
    d_slit = d_um * 1e-6
    t_slit = t_slit_nm * 1e-9

    if auto_calibrate:
        target_fringe_m = 0.18e-3
        lam_target = target_fringe_m * d_slit / L
        def lam_of_v(v):
            beta = v / c
            gamma = 1.0 / np.sqrt(1.0 - beta**2)
            return h / (gamma * m_e * v)
        v_adj = fsolve(lambda v: lam_of_v(v) - lam_target, v_mean)[0]
        v_mean = float(v_adj)
        print(f"🌟 Auto‑calibrated: v_mean = {v_mean/1e3:.1f} km/s (Jönsson fringe = 0.18 mm)")

    lam_avg = de_broglie_relativistic(v_mean)
    v_samples, T_fitted, mb_weights = maxwell_boltzmann_samples(
        v_mean, T_source, n_vel_samples, rng_seed)
    lam_samples = np.array([de_broglie_relativistic(v) for v in v_samples])

    spacing = lam_avg * L / d_slit
    if x_limit_mm is None:
        x_lim = max(0.004, 4.5 * spacing)
    else:
        x_lim = x_limit_mm / 1e3
    x = np.linspace(-x_lim, x_lim, 1500)

    qs = QuantumSystem(N_slits)
    rho_0 = qs.pure_state()
    coh_target = 1 - np.clip(K_input, 0, 1)
    g_eff = (-np.log(max(coh_target, 1e-6)) + gamma_meas)
    rho = qs.lindblad(rho_0, g_eff)
    if stray_rms > 0:
        rng_s = np.random.default_rng(rng_seed + 99)
        phi_s = rng_s.normal(0, stray_rms)
        for i in range(N_slits):
            for j in range(N_slits):
                if i != j:
                    rho[i, j] *= np.exp(1j * phi_s)
    coh = qs.coherence(rho)
    K_act = float(1 - coh)
    pur = qs.purity(rho)
    ent = qs.entropy(rho)

    y_dense = np.linspace(-3*d_slit, 3*d_slit, 300)
    if t_slit > 0:
        T_amp_arr, phi_wall_arr, P_back_avg = slit_transmission(
            y_dense, d_slit, a_width, t_slit, v_mean, N_slits,
            n_wall=n_wall, back_r0=back_r0)
        def phi_wall_fn(y_s, yc):
            return np.interp(y_s, y_dense, phi_wall_arr, left=0, right=0)
        def T_amp_fn(y_s, yc):
            return np.interp(y_s, y_dense, T_amp_arr, left=0, right=0)
    else:
        P_back_avg = 0.0
        phi_wall_fn = None
        T_amp_fn = None

    I_wave = np.zeros_like(x)
    for lam_v, w in zip(lam_samples, mb_weights):
        I_v = feynman_vectorized(
            x, lam_v, L, a_width, d_slit, N_slits,
            n_paths=n_paths, rng_seed=rng_seed,
            edge_strength=edge_strength, edge_angle=edge_angle,
            A_y=A_y, phi_wall_fn=phi_wall_fn, T_amp_fn=T_amp_fn)
        I_wave += w * I_v
    I_wave = normalize(I_wave)

    I_part = np.zeros_like(x)
    sigma_p = a_width * L / lam_avg
    for s in range(N_slits):
        yc = (s - (N_slits - 1) / 2.0) * d_slit
        I_part += np.exp(-(x - yc)**2 / (2 * sigma_p**2))
    I_part = normalize(I_part)

    I_mix = normalize(coh * I_wave + (1 - coh) * I_part)
    if pattern_mode == 'Wave only':
        I_main = I_wave
    elif pattern_mode == 'Particle only':
        I_main = I_part
    else:
        I_main = I_mix
    if P_back_avg > 0:
        I_main = normalize(I_main * (1 - P_back_avg))

    det = Detector(eta, dark_rate, det_T, readout_nm*1e-6, pixel_um, rng_seed)
    counts, n_det, n_dark = det.detect(x, I_main, n_particles)
    I_det = normalize(counts)

    V_theory = compute_visibility(x, I_main)
    V_meas = compute_visibility(x, I_det)
    VK2 = V_theory**2 + K_act**2
    sim_fringe = spacing * 1e3
    ref_fringe = REF_JONSSON['fringe_mm']
    fringe_err = abs(sim_fringe - ref_fringe) / ref_fringe * 100
    lam_nr = h / (m_e * v_mean)
    rel_corr = abs(lam_nr - lam_avg) / lam_nr * 100

    q_w, p_w, W = qs.wigner_n2(rho)

    # ──────────────────────────────────────────────────────────────────────────
    # PLOTTING
    # ──────────────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(26, 17))
    fig.patch.set_facecolor('#0d1117')
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.32,
                           top=0.92, bottom=0.05)

    def style_ax(ax, title='', xlabel='', ylabel='', tc='#58a6ff'):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_title(title, color=tc, fontsize=9, pad=4)
        ax.set_xlabel(xlabel, color='#8b949e', fontsize=8)
        ax.set_ylabel(ylabel, color='#8b949e', fontsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor('#30363d')
        ax.grid(alpha=0.18, color='#30363d')
        return ax

    x_mm = x * 1e3

    # Row 0
    ax = style_ax(fig.add_subplot(gs[0,0]), 'Intensity I(x)', 'Position (mm)', 'Intensity')
    ax.plot(x_mm, I_wave, '--', color='#3fb950', lw=1, alpha=0.5, label='Wave')
    ax.plot(x_mm, I_part, '--', color='#f85149', lw=1, alpha=0.5, label='Particle')
    ax.plot(x_mm, I_main, color='#58a6ff', lw=2, label=pattern_mode)
    ax.fill_between(x_mm, I_main, alpha=0.12, color='#58a6ff')
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')
    ax.text(0.02, 0.93, f'V={V_theory:.3f}', transform=ax.transAxes, color='#e3b341', fontsize=10, weight='bold')
    if A_y != 0:
        ax.text(0.02, 0.84, f'A_y={A_y:.3f} T·m\n(AB shift)', transform=ax.transAxes, color='#a371f7', fontsize=7.5)

    ax = style_ax(fig.add_subplot(gs[0,1]), f'Detector  η={eta:.0%}  T={det_T:.0f}K', 'Position (mm)', 'Counts')
    ax.bar(x_mm, I_det, width=x_mm[1]-x_mm[0], color='#388bfd', alpha=0.65)
    ax.plot(x_mm, I_main, color='#f0883e', lw=1.5, label='Theory')
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')
    ax.text(0.02, 0.88, f'V_meas={V_meas:.3f}\nN_det={n_det}\nDark={n_dark}', transform=ax.transAxes, color='#8b949e', fontsize=8)

    ax = style_ax(fig.add_subplot(gs[0,2]), 'Density Matrix ρ', 'Slit j', 'Slit i')
    cmap_r = LinearSegmentedColormap.from_list('rho', ['#f85149','#0d1117','#3fb950'])
    im = ax.imshow(np.real(rho), cmap=cmap_r, vmin=-0.5, vmax=0.5)
    labels = [f'|{i+1}⟩' for i in range(N_slits)]
    ax.set_xticks(range(N_slits), labels, color='#c9d1d9', fontsize=10)
    ax.set_yticks(range(N_slits), labels, color='#c9d1d9', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(colors='#8b949e')
    for i in range(N_slits):
        for j in range(N_slits):
            vv = np.real(rho[i,j])
            ax.text(j, i, f'{vv:.3f}', ha='center', va='center',
                    color='white' if abs(vv)<0.3 else 'black', fontsize=10, weight='bold')
    ax.text(0.02, 0.04, f'C={coh:.3f}  Tr(ρ²)={pur:.3f}', transform=ax.transAxes, color='#a371f7', fontsize=8)

    ax = style_ax(fig.add_subplot(gs[0,3]), 'Wigner Function', 'q', 'p')
    if W is not None:
        w_max = max(abs(np.min(W)), abs(np.max(W)), 1e-9)
        norm_w = TwoSlopeNorm(vmin=-w_max, vcenter=0, vmax=w_max)
        cmap_w = LinearSegmentedColormap.from_list('wig', ['#f85149','#0d1117','#58a6ff'])
        ax.pcolormesh(q_w, p_w, W, cmap=cmap_w, norm=norm_w, shading='auto')
        ax.contour(q_w, p_w, W, levels=[0], colors=['#e3b341'], linewidths=1)
        ax.text(0.02, 0.92, 'Yellow=W=0\nBlue<0=quantum', transform=ax.transAxes, color='#e3b341', fontsize=7, va='top')
    else:
        ax.text(0.5,0.5,'N>2: Wigner\nnot shown', transform=ax.transAxes, ha='center', color='#8b949e')
        ax.axis('off')

    # Row 1
    ax = style_ax(fig.add_subplot(gs[1,0]), 'Maxwell-Boltzmann', 'v (km/s)', 'f(v) [norm.]', tc='#3fb950')
    v_plot = np.linspace(0.02*v_mean, 2.5*v_mean, 400)
    T_eff = T_fitted * (1 + T_source/1e4)
    f_mb = (4*np.pi*(m_e/(2*np.pi*k_B*T_eff))**1.5 * v_plot**2 * np.exp(-m_e*v_plot**2/(2*k_B*T_eff)))
    f_mb /= np.max(f_mb)
    ax.fill_between(v_plot/1e3, f_mb, alpha=0.25, color='#3fb950')
    ax.plot(v_plot/1e3, f_mb, color='#3fb950', lw=2, label='MB')
    ax.hist(v_samples/1e3, bins=25, density=True, weights=mb_weights*len(v_samples), color='#58a6ff', alpha=0.5, label='Sampled')
    ax.axvline(v_mean/1e3, color='#e3b341', ls='--', label=f'v_mean={v_mean/1e3:.0f} km/s')
    ax.set_xlim(0, 2.5*v_mean/1e3)
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')
    ax.text(0.55,0.85, f'T_source={T_source:.0f}K\nT_fit={T_fitted:.0f}K\nΔv/v={np.std(v_samples)/v_mean:.3f}',
            transform=ax.transAxes, color='#c9d1d9', fontsize=7.5,
            bbox=dict(facecolor='#0d2b0d', edgecolor='#3fb950', boxstyle='round,pad=0.3', alpha=0.85))

    ax = style_ax(fig.add_subplot(gs[1,1]), 'Slit Thickness', 'y (µm)', 'Amplitude / Phase', tc='#e3b341')
    if t_slit > 0:
        ax.plot(y_dense*1e6, T_amp_arr, color='#3fb950', lw=2, label='T(y)')
        ax2 = ax.twinx()
        ax2.plot(y_dense*1e6, phi_wall_arr, color='#e3b341', lw=1.5, ls='--', label='φ_wall')
        ax2.set_ylabel('φ_wall (rad)', color='#e3b341', fontsize=8)
        ax2.tick_params(colors='#e3b341')
        ax.set_xlim(y_dense[0]*1e6, y_dense[-1]*1e6)
        ax.legend(fontsize=7, loc='upper left')
        ax2.legend(fontsize=7, loc='upper right')
        ax.text(0.02,0.08, f't_slit={t_slit_nm:.0f}nm\nn_wall={n_wall:.2f}\nP_back={P_back_avg:.3f}',
                transform=ax.transAxes, color='#e3b341', fontsize=8,
                bbox=dict(facecolor='#2b2200', edgecolor='#e3b341', boxstyle='round,pad=0.3', alpha=0.85))
        for s in range(N_slits):
            yc = (s-(N_slits-1)/2)*d_slit
            for edge in [yc-a_width/2, yc+a_width/2]:
                ax.axvline(edge*1e6, color='#f85149', ls=':', lw=0.8, alpha=0.7)
    else:
        ax.text(0.5,0.5,'Set t_slit_nm > 0', transform=ax.transAxes, ha='center', color='#8b949e')
        ax.axis('off')

    ax = style_ax(fig.add_subplot(gs[1,2]), 'Aharonov-Bohm (A_y)', 'Position (mm)', 'Intensity', tc='#a371f7')
    A_range = [0.0, A_y/2 if A_y!=0 else 0.0001, A_y, A_y*2 if A_y!=0 else 0.0002]
    cols_ab = ['#3fb950','#58a6ff','#e3b341','#f85149']
    for A_val, col_ab in zip(A_range, cols_ab):
        I_ab = feynman_vectorized(x, lam_avg, L, a_width, d_slit, N_slits, n_paths=60,
                                  rng_seed=rng_seed, edge_strength=0, A_y=A_val)
        ax.plot(x_mm, normalize(I_ab), color=col_ab, lw=1.2, alpha=0.8, label=f'A_y={A_val:.4f}')
    ax.set_xlim(x_mm[0], x_mm[-1])
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9', title='T·m')
    ax.text(0.02,0.88,'Pattern SHIFTS\n(not just phase)', transform=ax.transAxes, color='#a371f7', fontsize=8,
            bbox=dict(facecolor='#1a0d2b', edgecolor='#a371f7', boxstyle='round,pad=0.3', alpha=0.85))

    ax = style_ax(fig.add_subplot(gs[1,3]), 'Complementarity', 'K', 'V')
    theta_c = np.linspace(0, np.pi/2, 300)
    ax.plot(np.cos(theta_c), np.sin(theta_c), '--', color='#f0883e', lw=2, label='Bohr limit')
    ax.fill_between(np.cos(theta_c), np.sin(theta_c), alpha=0.07, color='#f0883e')
    rho_p = qs.pure_state()
    g_arr = np.linspace(0,3,60)
    V_tr, K_tr = [], []
    for g in g_arr:
        r_g = qs.lindblad(rho_p, g, n_steps=20)
        c_g = qs.coherence(r_g)
        I_g = normalize(c_g * I_wave + (1-c_g) * I_part)
        V_tr.append(compute_visibility(x, I_g))
        K_tr.append(1-c_g)
    ax.plot(K_tr, V_tr, color='#a371f7', lw=2, label='Lindblad')
    ax.scatter([K_act], [V_theory], s=150, color='#3fb950', zorder=5, label='Current')
    ax.set_xlim(0,1.05); ax.set_ylim(0,1.05); ax.set_aspect('equal')
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')
    col_vk = '#3fb950' if VK2<=1 else '#f85149'
    ax.text(0.05,0.07, f'V²+K² = {VK2:.4f}', transform=ax.transAxes, color=col_vk, fontsize=10, weight='bold')

    # Row 2
    ax = style_ax(fig.add_subplot(gs[2,0]), 'Coherence C(ρ)', 'γ_eff', 'C')
    g_arr2 = np.linspace(0,5,200)
    C_arr = [qs.coherence(qs.lindblad(rho_p, g, 30)) for g in g_arr2]
    ax.plot(g_arr2, C_arr, color='#58a6ff', lw=2.5)
    ax.axvline(g_eff, color='#f85149', ls=':', label=f'γ_eff={g_eff:.2f}')
    ax.axhline(coh, color='#3fb950', ls='--', label=f'C={coh:.3f}')
    ax.set_xlim(0,5); ax.set_ylim(0,1.05)
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')

    ax = style_ax(fig.add_subplot(gs[2,1]), 'Tonomura buildup', 'Position (mm)', 'Counts')
    det2 = Detector(eta, dark_rate, det_T, readout_nm*1e-6, pixel_um, rng_seed+1)
    stages = [10,100,500,n_particles]
    cols_s = ['#f85149','#e3b341','#58a6ff','#3fb950']
    for ns, col_s in zip(stages, cols_s):
        cnt, _, _ = det2.detect(x, I_main, ns)
        cnt = cnt / max(np.max(cnt),1)
        ax.plot(x_mm, cnt + stages.index(ns)*1.15, color=col_s, lw=0.9, label=f'N={ns}')
    ax.set_xlim(x_mm[0], x_mm[-1]); ax.set_yticks([])
    ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')

    ax = style_ax(fig.add_subplot(gs[2,2]), 'Multi-slit + AB', 'Position (mm)', 'Intensity')
    for Ns, col_ns in zip([2,3,5], ['#58a6ff','#3fb950','#f0883e']):
        I_ns = feynman_vectorized(x, lam_avg, L, a_width, d_slit, Ns, n_paths=60,
                                  rng_seed=rng_seed, edge_strength=edge_strength, A_y=A_y)
        ax.plot(x_mm, normalize(I_ns), color=col_ns, lw=1.2, alpha=0.85, label=f'{Ns} slits')
    ax.set_xlim(x_mm[0], x_mm[-1]); ax.legend(fontsize=7, facecolor='#161b22', labelcolor='#c9d1d9')

    # Summary panel (FIXED: no weight='italic')
    ax = fig.add_subplot(gs[2,3])
    ax.set_facecolor('#161b22'); ax.axis('off')
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_edgecolor('#388bfd'); sp.set_linewidth(1.5)

    lines = [
        ("Berramdane V14.3 (final)", '#58a6ff', 11, 'bold'),
        ("MB · Thickness · AB · auto_calibrate", '#8b949e', 7, 'italic'),
        ("", '', 7, 'normal'),
        ("── WAVELENGTH ──", '#e3b341', 8, 'bold'),
        (f"  λ_rel  = {lam_avg*1e9:.4f} nm", '#c9d1d9', 8, 'normal'),
        (f"  Δλ/λ   = {rel_corr:.4f}%", '#3fb950', 8, 'normal'),
        ("── MAXWELL-BOLTZMANN ──", '#3fb950', 8, 'bold'),
        (f"  T_source = {T_source:.0f} K", '#c9d1d9', 8, 'normal'),
        (f"  T_fitted = {T_fitted:.0f} K", '#c9d1d9', 8, 'normal'),
        (f"  Δv/v    = {np.std(v_samples)/v_mean:.4f}", '#c9d1d9', 8, 'normal'),
        ("── SLIT THICKNESS ──", '#e3b341', 8, 'bold'),
        (f"  t_slit  = {t_slit_nm:.0f} nm", '#c9d1d9', 8, 'normal'),
        (f"  n_wall  = {n_wall:.2f}", '#c9d1d9', 8, 'normal'),
        (f"  P_back  = {P_back_avg:.4f}", '#c9d1d9', 8, 'normal'),
        ("── AHARONOV-BOHM ──", '#a371f7', 8, 'bold'),
        (f"  A_y     = {A_y:.4f} T·m", '#a371f7', 8, 'normal'),
        (f"  φ_AB(d) = {(e_c/hbar)*A_y*d_slit:.4f} rad", '#a371f7', 8, 'normal'),
        ("── QUANTUM ──", '#e3b341', 8, 'bold'),
        (f"  V       = {V_theory:.4f}", '#3fb950', 8, 'normal'),
        (f"  V_meas  = {V_meas:.4f}", '#c9d1d9', 8, 'normal'),
        (f"  K       = {K_act:.4f}", '#c9d1d9', 8, 'normal'),
        (f"  V²+K²   = {VK2:.4f} {'✓' if VK2<=1 else '✗'}", '#3fb950' if VK2<=1 else '#f85149', 8, 'bold'),
        (f"  Purity  = {pur:.4f}", '#c9d1d9', 8, 'normal'),
        (f"  S(ρ)    = {ent:.4f}", '#c9d1d9', 8, 'normal'),
        ("── DETECTOR ──", '#e3b341', 8, 'bold'),
        (f"  N_det   = {n_det}/{n_particles}", '#c9d1d9', 8, 'normal'),
        (f"  Dark    = {n_dark}", '#c9d1d9', 8, 'normal'),
        (f"  Fringe  = {sim_fringe:.3f} mm ({fringe_err:.1f}%)", '#3fb950' if fringe_err<5 else '#e3b341', 8, 'normal'),
    ]

    y_s = 0.98
    for txt, col, sz, wt in lines:
        if txt:
            # Fix: treat 'italic' as fontstyle, not weight
            if wt == 'italic':
                fw = 'normal'
                fs = 'italic'
            elif wt == 'bold':
                fw = 'bold'
                fs = 'normal'
            else:
                fw = 'normal'
                fs = 'normal'
            ax.text(0.04, y_s, txt, transform=ax.transAxes, va='top',
                    color=col, fontsize=sz, fontweight=fw, fontstyle=fs,
                    family='monospace')
        y_s -= 0.034

    fig.text(0.5, 0.975, 'Berramdane V14.3 — MB · Slit Thickness · Vector Potential A · auto_calibrate',
             ha='center', va='top', color='#c9d1d9', fontsize=13, weight='bold')
    fig.text(0.5, 0.958, f'v={v_mean/1e3:.0f}km/s | L={L_mm:.0f}mm | a={a_um:.2f}µm | d={d_um:.2f}µm | '
                         f't={t_slit_nm:.0f}nm | A_y={A_y:.4f}T·m | T_src={T_source:.0f}K',
             ha='center', va='top', color='#8b949e', fontsize=8)
    plt.show()

    print("═"*65)
    print("  Berramdane V14.3 — Full Report")
    print("═"*65)
    print(f"  ★ MB: T_source={T_source:.0f}K, Δv/v={np.std(v_samples)/v_mean:.4f}")
    print(f"  ★ Slit thickness: t={t_slit_nm:.0f}nm, P_back={P_back_avg:.4f}")
    print(f"  ★ AB: A_y={A_y:.4f} T·m, φ_AB(d)={(e_c/hbar)*A_y*d_slit:.4f} rad")
    print(f"  V={V_theory:.4f} | V_meas={V_meas:.4f} | K={K_act:.4f} | V²+K²={VK2:.4f}")
    print(f"  Fringe={sim_fringe:.3f}mm (err={fringe_err:.1f}%) | Purity={pur:.4f} | S={ent:.4f}")
    print(f"  Detected: {n_det}/{n_particles} | Dark: {n_dark}")
    print("═"*65)

    return {'I': I_main, 'x': x, 'V': V_theory, 'purity': pur, 'entropy': ent,
            'v_samples': v_samples, 'mb_weights': mb_weights, 'lam': lam_avg, 'rho': rho}

# ──────────────────────────────────────────────────────────────────────────────
# CSV EXPORT
# ──────────────────────────────────────────────────────────────────────────────
def export_csv(result, filename='berramdane_v14_3.csv'):
    x = result['x']; I = result['I']; dx = x[1]-x[0]
    Px = I / (np.sum(I)*dx)
    df = pd.DataFrame({'position_mm': x*1e3, 'intensity': I, 'P_x_mm': Px*1e-3,
                       'CDF_x': np.cumsum(Px)*dx, 'lambda_nm': result['lam']*1e9})
    df.to_csv(filename, index=False)
    print(f"✅ Exported {len(df)} rows to {filename}")
    return df

# ──────────────────────────────────────────────────────────────────────────────
# INTERACTIVE WIDGETS (FIXED)
# ──────────────────────────────────────────────────────────────────────────────
if WIDGETS_OK:
    _gdata = {}
    exp_btn = Button(description="📥 Export CSV", button_style='success')
    exp_out = Output()
    def on_export(b):
        with exp_out:
            exp_out.clear_output()
            if 'x' in _gdata:
                export_csv(_gdata)
            else:
                print("❌ Run simulation first.")
    exp_btn.on_click(on_export)

    @interact(
        v_mean=FloatSlider(value=70000, min=20000, max=150000, step=1000, description='v_mean (m/s)', continuous_update=False),
        T_source=FloatSlider(value=1000, min=10, max=20000, step=100, description='T_source (K)', continuous_update=False),
        auto_calibrate=Checkbox(value=False, description='Auto-calibrate to Jönsson fringe'),
        L_mm=FloatSlider(value=350, min=100, max=1000, step=10, description='L (mm)', continuous_update=False),
        a_um=FloatSlider(value=0.3, min=0.1, max=2.0, step=0.05, description='a (µm)', continuous_update=False),
        d_um=FloatSlider(value=1.0, min=0.2, max=3.0, step=0.05, description='d (µm)', continuous_update=False),
        N_slits=IntSlider(value=2, min=2, max=5, step=1, description='N slits'),
        t_slit_nm=FloatSlider(value=0, min=0, max=500, step=10, description='t_slit (nm)', continuous_update=False),
        n_wall=FloatSlider(value=1.5, min=1.0, max=3.0, step=0.1, description='n_wall', continuous_update=False),
        back_r0=FloatSlider(value=0.05, min=0.0, max=0.3, step=0.01, description='back_r0', continuous_update=False),
        gamma_meas=FloatSlider(value=0.0, min=0.0, max=5.0, step=0.1, description='γ_meas', continuous_update=False),
        K_input=FloatSlider(value=0.0, min=0.0, max=1.0, step=0.05, description='K (which-path)'),
        edge_strength=FloatSlider(value=0.0, min=0.0, max=2.0, step=0.1, description='Edge strength', continuous_update=False),
        A_y=FloatSlider(value=0.0, min=-0.01, max=0.01, step=0.0005, description='A_y (T·m)', continuous_update=False),
        stray_rms=FloatSlider(value=0.0, min=0.0, max=2.0, step=0.1, description='Stray RMS (rad)', continuous_update=False),
        n_particles=IntSlider(value=5000, min=500, max=20000, step=500, description='N particles', continuous_update=False),
        eta=FloatSlider(value=0.85, min=0.1, max=1.0, step=0.05, description='η'),
        dark_rate=FloatSlider(value=0.02, min=0.0, max=0.2, step=0.01, description='Dark rate'),
        det_T=FloatSlider(value=300, min=4, max=800, step=10, description='T_det (K)', continuous_update=False),
        readout_nm=FloatSlider(value=500, min=0, max=2000, step=50, description='Readout (nm)', continuous_update=False),
        pixel_um=FloatSlider(value=5.0, min=1.0, max=20.0, step=1.0, description='Pixel (µm)'),
        n_paths=IntSlider(value=100, min=30, max=300, step=10, description='N paths', continuous_update=False),
        pattern_mode=Dropdown(options=['Mixed','Wave only','Particle only'], value='Mixed', description='Pattern'),
        rng_seed=IntText(value=42, description='Seed'),
    )
    def irun(v_mean, T_source, auto_calibrate, L_mm, a_um, d_um, N_slits,
             t_slit_nm, n_wall, back_r0, gamma_meas, K_input,
             edge_strength, A_y, stray_rms, n_particles,
             eta, dark_rate, det_T, readout_nm, pixel_um,
             n_paths, pattern_mode, rng_seed):
        res = run_v14_3(
            v_mean=v_mean, T_source=T_source, auto_calibrate=auto_calibrate,
            L_mm=L_mm, a_um=a_um, d_um=d_um, N_slits=N_slits,
            t_slit_nm=t_slit_nm, n_wall=n_wall, back_r0=back_r0,
            gamma_meas=gamma_meas, K_input=K_input,
            edge_strength=edge_strength, A_y=A_y, stray_rms=stray_rms,
            n_particles=n_particles, eta=eta, dark_rate=dark_rate,
            det_T=det_T, readout_nm=readout_nm, pixel_um=pixel_um,
            n_paths=n_paths, pattern_mode=pattern_mode, rng_seed=rng_seed
        )
        _gdata.update(res)

    display(HBox([exp_btn, Label("← exports data to CSV")]), exp_out)
else:
    print("Static run (no widgets) — to enable widgets, install ipywidgets and restart kernel.")
    run_v14_3(auto_calibrate=True, t_slit_nm=50, A_y=0.002)
