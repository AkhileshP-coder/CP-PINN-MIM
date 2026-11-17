#!/usr/bin/env python
"""
Parameterized coupled PINN (CP-PINN) for mobile-immobile transport.

Training (3 seeds), loss-weight sensitivity sweep, test-set export, inference timing
and figures. Set the environment variable QUICK_TEST=1 for a short smoke test.
Input data: BTC_spatial_Cm_Cim.csv (columns: case, alpha, thetam, thetaim, f, rho, kd, Dxx, qx,
x_over_L, x, time, Cm, Cim). The breakthrough curve (BTC) of each case is the x_over_L = 1 slice
(outlet) of this single file; concentrations are normalised by the inlet concentration C0.
"""

import os
import re
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 11

warnings.filterwarnings('ignore')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------------------------------------------- user options
QUICK_TEST = os.environ.get('QUICK_TEST', '0') == '1'
EPOCHS_FULL = 20 if QUICK_TEST else 2000      # seeds + final training
EPOCHS_SWEEP = 10 if QUICK_TEST else 500      # sensitivity sweeps
# True -> natural sort: case1..case30 are the training cases and case31..case50 the test cases
NATURAL_SORT = True
N_WARMUP = 10          # warm-up forward passes before timing inference
N_REPEAT = 200         # repeated forward passes used for the timing statistics
# ------------------------------------------------------------------------------

PARAM_KEYS = ['alpha', 'theta_m', 'theta_im', 'f', 'rho', 'kd', 'D_xx', 'q_x']

print("\n" + "=" * 90)
print(" " * 15 + "COMPREHENSIVE PARAMETERIZED CP-PINN (v2)")
print(" " * 10 + "Complete Analysis Suite with Timing Tables & Full Test Export")
print("=" * 90)
print(f"Device: {DEVICE.upper()}   QUICK_TEST={QUICK_TEST}\n")

timings = {}


def sync():
    if DEVICE == 'cuda':
        torch.cuda.synchronize()


# ============================================================================
# PARAMETERIZED NETWORK ARCHITECTURE
# ============================================================================

class ParameterizedCPPINN(nn.Module):
    """
    Dual-stream PINN with transport parameters as inputs.
    Input: (x, t, alpha, theta_m, theta_im, f, rho, K_d, D_xx, q_x) - 10 inputs
    Output: (C_m, C_im)
    """
    def __init__(self, hidden=64):
        super(ParameterizedCPPINN, self).__init__()

        self.encoder_m = nn.Sequential(
            nn.Linear(10, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )

        self.encoder_im = nn.Sequential(
            nn.Linear(10, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh()
        )

        self.decoder_m = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

        self.decoder_im = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1), nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.8)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, t, alpha, theta_m, theta_im, f, rho, kd, D_xx, q_x):
        inputs = torch.cat([x, t, alpha, theta_m, theta_im, f, rho, kd, D_xx, q_x], dim=1)

        feat_m = self.encoder_m(inputs)
        feat_im = self.encoder_im(inputs)

        C_m = self.decoder_m(feat_m)
        C_im = self.decoder_im(feat_im)

        return C_m, C_im


# ============================================================================
# DATA LOADING & NORMALIZATION
# ============================================================================

print("Loading data...")
t0_data = time.time()

possible_paths = [
    'BTC_spatial_Cm_Cim.csv',
    './BTC_spatial_Cm_Cim.csv',
    '../data/BTC_spatial_Cm_Cim.csv',
    '../../data/BTC_spatial_Cm_Cim.csv',
    Path.home() / 'Downloads' / 'BTC_spatial_Cm_Cim.csv',
]

csv_file = None
for path in possible_paths:
    if Path(path).exists():
        csv_file = str(path)
        print(f"Found: {csv_file}\n")
        break

if csv_file is None:
    print("Enter path of BTC_spatial_Cm_Cim.csv: ")
    csv_file = input().strip().strip('"\'')

df_all = pd.read_csv(csv_file)

# Inlet concentration C0 [M L^-3] of the dataset: maximum (pulse) concentration of the mobile phase at the inlet x = 0
C0 = float(np.round(df_all[np.isclose(df_all['x_over_L'], 0.0)]['Cm'].max(), 6))
print(f"Inlet concentration C0 = {C0:.6g}  (all concentrations are divided by C0)")

# Breakthrough curve = outlet slice (x/L = 1) of the same file, mobile-phase concentration Cm
df = df_all[np.isclose(df_all['x_over_L'], 1.0)].copy()
df = df.rename(columns={'Cm': 'concentration'})


def _natural_key(s):
    return [int(p) if p.isdigit() else p for p in re.split(r'(\d+)', s)]


all_cases = sorted(df['case'].unique(), key=_natural_key if NATURAL_SORT else None)

train_cases = all_cases[:30]
test_cases = all_cases[30:50]

print(f"Training cases: {len(train_cases)} -> {train_cases}")
print(f"Test cases:     {len(test_cases)} -> {test_cases}\n")

print("Computing parameter statistics from TRAINING CASES ONLY...")
print("(No test-case data is used in normalization)\n")

alpha_vals, theta_m_vals, theta_im_vals, f_vals = [], [], [], []
rho_vals, kd_vals, D_xx_vals, q_x_vals = [], [], [], []

for case in train_cases:
    case_data = df[df['case'] == case]
    alpha_vals.append(float(case_data['alpha'].iloc[0]))
    theta_m_vals.append(float(case_data['thetam'].iloc[0]))
    theta_im_vals.append(float(case_data['thetaim'].iloc[0]))
    f_vals.append(float(case_data['f'].iloc[0]))
    rho_vals.append(float(case_data['rho'].iloc[0]))
    kd_vals.append(float(case_data['kd'].iloc[0]))
    D_xx_vals.append(float(case_data['Dxx'].iloc[0]))
    q_x_vals.append(float(case_data['qx'].iloc[0]))

param_stats = {
    'alpha': {'mean': np.mean(alpha_vals), 'std': np.std(alpha_vals)},
    'theta_m': {'mean': np.mean(theta_m_vals), 'std': np.std(theta_m_vals)},
    'theta_im': {'mean': np.mean(theta_im_vals), 'std': np.std(theta_im_vals)},
    'f': {'mean': np.mean(f_vals), 'std': np.std(f_vals)},
    'rho': {'mean': np.mean(rho_vals), 'std': np.std(rho_vals)},
    'kd': {'mean': np.mean(kd_vals), 'std': np.std(kd_vals)},
    'D_xx': {'mean': np.mean(D_xx_vals), 'std': np.std(D_xx_vals)},
    'q_x': {'mean': np.mean(q_x_vals), 'std': np.std(q_x_vals)},
}

print("Training cases parameter ranges (normalization basis):")
print(f"  alpha ∈ [{np.min(alpha_vals):.4f}, {np.max(alpha_vals):.4f}]")
print(f"  theta_m ∈ [{np.min(theta_m_vals):.4f}, {np.max(theta_m_vals):.4f}]")
print(f"  K_d ∈ [{np.min(kd_vals):.6f}, {np.max(kd_vals):.6f}]")
print(f"  D_xx ∈ [{np.min(D_xx_vals):.6f}, {np.max(D_xx_vals):.6f}]")
print(f"  q_x ∈ [{np.min(q_x_vals):.4f}, {np.max(q_x_vals):.4f}]\n")


def normalize_param(value, param_name):
    mean = param_stats[param_name]['mean']
    std = param_stats[param_name]['std']
    if std < 1e-10:
        return 0.0
    return (value - mean) / (std + 1e-10)


def prepare_case(case_name, data_df):
    case_data = data_df[data_df['case'] == case_name].copy()

    alpha = float(case_data['alpha'].iloc[0])
    theta_m = float(case_data['thetam'].iloc[0])
    theta_im = float(case_data['thetaim'].iloc[0])
    f = float(case_data['f'].iloc[0])
    rho = float(case_data['rho'].iloc[0])
    kd = float(case_data['kd'].iloc[0])
    D_xx = float(case_data['Dxx'].iloc[0])
    q_x = float(case_data['qx'].iloc[0])

    params_true = {
        'alpha': alpha, 'theta_m': theta_m, 'theta_im': theta_im, 'f': f,
        'rho': rho, 'kd': kd, 'D_xx': D_xx, 'q_x': q_x,
        'L': float(case_data['x'].max()),
        'T': float(case_data['time'].max()),
        'tp': 7.708,
    }

    params_normalized = {k: normalize_param(params_true[k], k) for k in PARAM_KEYS}

    x_vals = case_data['x'].values
    t_vals = case_data['time'].values
    C_obs = case_data['concentration'].values
    C_norm = C_obs / C0   # dimensionless C/C0

    return {
        'x': torch.tensor(x_vals, dtype=torch.float32, device=DEVICE).reshape(-1, 1),
        't': torch.tensor(t_vals, dtype=torch.float32, device=DEVICE).reshape(-1, 1),
        'C_obs': torch.tensor(C_norm, dtype=torch.float32, device=DEVICE),
        'C0': C0,
        'params_true': params_true,
        'params_normalized': params_normalized,
    }


train_data = [prepare_case(c, df) for c in train_cases]
test_data = [prepare_case(c, df) for c in test_cases]


def build_inputs(case):
    """Model inputs (x, t, 8 normalised params) for one case, in forward() order."""
    x, t = case['x'], case['t']
    pn = case['params_normalized']
    return (x, t, *[torch.full_like(x, pn[k]) for k in PARAM_KEYS])


timings['data_loading'] = time.time() - t0_data
print(f"Data loading time: {timings['data_loading']:.2f}s\n")


# ============================================================================
# COLLOCATION POINT GENERATION
# ============================================================================

def generate_collocation_points(params, strategy='uniform', n_interior=100,
                                n_bc_inlet=50, n_bc_outlet=50, n_ic=50):
    """Generate collocation points with different strategies"""
    L = params['L']
    T = params['T']
    tp = params['tp']

    if strategy == 'uniform':
        x_interior = np.random.uniform(0, L, n_interior)
        t_interior = np.random.uniform(0, T, n_interior)
    elif strategy == 'additional_less_cluster':
        t_interior = np.random.beta(2, 5, n_interior) * T      # more points at early time
        x_interior = np.random.uniform(0, L, n_interior)
    elif strategy == 'invariable_less_cluster':
        t_interior = np.random.uniform(0, T, n_interior)
        x_interior = np.random.uniform(0, L, n_interior)
    else:
        x_interior = np.random.uniform(0, L, n_interior)
        t_interior = np.random.uniform(0, T, n_interior)

    # Inlet BC
    n_pulse = n_bc_inlet // 2
    n_post_pulse = n_bc_inlet - n_pulse

    t_pulse = np.random.uniform(0.0001, tp, n_pulse) if tp > 0 else np.array([])
    if tp < T:
        t_post_pulse = np.random.uniform(tp + 0.0001, T, n_post_pulse)
        t_inlet = np.concatenate([t_pulse, t_post_pulse])
    else:
        t_inlet = t_pulse

    x_inlet = np.zeros(len(t_inlet))

    # Outlet BC
    x_outlet = np.full(n_bc_outlet, L)
    t_outlet = np.random.uniform(0, T, n_bc_outlet)

    # IC
    x_ic = np.random.uniform(0, L, n_ic)
    t_ic = np.zeros(n_ic)

    def _t(a):
        return torch.tensor(a, dtype=torch.float32, device=DEVICE).reshape(-1, 1)

    return {
        'interior': {'x': _t(x_interior), 't': _t(t_interior)},
        'inlet_bc': {'x': _t(x_inlet), 't': _t(t_inlet)},
        'outlet_bc': {'x': _t(x_outlet), 't': _t(t_outlet)},
        'ic': {'x': _t(x_ic), 't': _t(t_ic)},
    }


# ============================================================================
# TRAINING FUNCTION WITH LOSS WEIGHT FLEXIBILITY
# ============================================================================

def train_model(seed, w_data=10.0, w_pde=1.0, w_ic=0.5, w_bc_in=0.5,
                w_bc_out=0.5, epochs=2000):
    """Train model with specified loss weights"""

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = ParameterizedCPPINN(hidden=64).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-3)

    loss_history = {
        'total': [], 'data': [], 'pde': [], 'ic': [], 'bc_inlet': [], 'bc_outlet': [],
        'test': [], 'r2': []
    }

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss_total = 0
        epoch_loss_data = 0
        epoch_loss_pde = 0
        epoch_loss_ic = 0
        epoch_loss_bc_inlet = 0
        epoch_loss_bc_outlet = 0
        batch_count = 0

        for case in train_data:
            optimizer.zero_grad()

            # DATA LOSS
            x_data = case['x'].clone().detach().requires_grad_(True)
            t_data = case['t'].clone().detach().requires_grad_(True)
            C_obs = case['C_obs']
            params_true = case['params_true']
            params_norm = case['params_normalized']

            alpha_t = torch.full_like(x_data, params_norm['alpha'])
            theta_m_t = torch.full_like(x_data, params_norm['theta_m'])
            theta_im_t = torch.full_like(x_data, params_norm['theta_im'])
            f_t = torch.full_like(x_data, params_norm['f'])
            rho_t = torch.full_like(x_data, params_norm['rho'])
            kd_t = torch.full_like(x_data, params_norm['kd'])
            D_xx_t = torch.full_like(x_data, params_norm['D_xx'])
            q_x_t = torch.full_like(x_data, params_norm['q_x'])

            C_m_data, _ = model(x_data, t_data, alpha_t, theta_m_t, theta_im_t,
                                f_t, rho_t, kd_t, D_xx_t, q_x_t)
            loss_data = torch.mean((C_m_data.squeeze() - C_obs) ** 2)

            # COLLOCATION & PDE LOSS
            colloc = generate_collocation_points(params_true, strategy='uniform')

            x_interior = colloc['interior']['x'].clone().detach().requires_grad_(True)
            t_interior = colloc['interior']['t'].clone().detach().requires_grad_(True)

            alpha_i = torch.full_like(x_interior, params_norm['alpha'])
            theta_m_i = torch.full_like(x_interior, params_norm['theta_m'])
            theta_im_i = torch.full_like(x_interior, params_norm['theta_im'])
            f_i = torch.full_like(x_interior, params_norm['f'])
            rho_i = torch.full_like(x_interior, params_norm['rho'])
            kd_i = torch.full_like(x_interior, params_norm['kd'])
            D_xx_i = torch.full_like(x_interior, params_norm['D_xx'])
            q_x_i = torch.full_like(x_interior, params_norm['q_x'])

            C_m_interior, C_im_interior = model(x_interior, t_interior, alpha_i, theta_m_i,
                                                theta_im_i, f_i, rho_i, kd_i, D_xx_i, q_x_i)

            dCm_dt = torch.autograd.grad(C_m_interior.sum(), t_interior, create_graph=True)[0]
            dCm_dx = torch.autograd.grad(C_m_interior.sum(), x_interior, create_graph=True)[0]
            d2Cm_dx2 = torch.autograd.grad(dCm_dx.sum(), x_interior, create_graph=True)[0]
            dCim_dt = torch.autograd.grad(C_im_interior.sum(), t_interior, create_graph=True)[0]

            D_xx = torch.tensor(params_true['D_xx'], device=DEVICE, dtype=torch.float32)
            q_x = torch.tensor(params_true['q_x'], device=DEVICE, dtype=torch.float32)
            alpha = torch.tensor(params_true['alpha'], device=DEVICE, dtype=torch.float32)
            theta_m = torch.tensor(params_true['theta_m'], device=DEVICE, dtype=torch.float32)
            theta_im = torch.tensor(params_true['theta_im'], device=DEVICE, dtype=torch.float32)
            rho = torch.tensor(params_true['rho'], device=DEVICE, dtype=torch.float32)
            kd = torch.tensor(params_true['kd'], device=DEVICE, dtype=torch.float32)
            f = torch.tensor(params_true['f'], device=DEVICE, dtype=torch.float32)
            tp = torch.tensor(params_true['tp'], device=DEVICE, dtype=torch.float32)


            coeff_m = theta_m + rho * kd * f
            pde_m = (coeff_m * dCm_dt
                     - theta_m * D_xx * d2Cm_dx2
                     + q_x * dCm_dx
                     + alpha * (C_m_interior - C_im_interior))

            coeff_im = theta_im + rho * kd * (1.0 - f)
            pde_im = coeff_im * dCim_dt - alpha * (C_m_interior - C_im_interior)

            loss_pde = torch.mean(pde_m ** 2) + torch.mean(pde_im ** 2)

            # IC LOSS
            x_ic = colloc['ic']['x'].clone().detach()
            t_ic = colloc['ic']['t'].clone().detach()

            alpha_ic = torch.full_like(x_ic, params_norm['alpha'])
            theta_m_ic = torch.full_like(x_ic, params_norm['theta_m'])
            theta_im_ic = torch.full_like(x_ic, params_norm['theta_im'])
            f_ic = torch.full_like(x_ic, params_norm['f'])
            rho_ic = torch.full_like(x_ic, params_norm['rho'])
            kd_ic = torch.full_like(x_ic, params_norm['kd'])
            D_xx_ic = torch.full_like(x_ic, params_norm['D_xx'])
            q_x_ic = torch.full_like(x_ic, params_norm['q_x'])

            C_m_ic, C_im_ic = model(x_ic, t_ic, alpha_ic, theta_m_ic, theta_im_ic,
                                    f_ic, rho_ic, kd_ic, D_xx_ic, q_x_ic)
            loss_ic = torch.mean(C_m_ic ** 2) + torch.mean(C_im_ic ** 2)

            # INLET BC LOSS
            x_inlet = colloc['inlet_bc']['x'].clone().detach()
            t_inlet = colloc['inlet_bc']['t'].clone().detach()

            alpha_in = torch.full_like(x_inlet, params_norm['alpha'])
            theta_m_in = torch.full_like(x_inlet, params_norm['theta_m'])
            theta_im_in = torch.full_like(x_inlet, params_norm['theta_im'])
            f_in = torch.full_like(x_inlet, params_norm['f'])
            rho_in = torch.full_like(x_inlet, params_norm['rho'])
            kd_in = torch.full_like(x_inlet, params_norm['kd'])
            D_xx_in = torch.full_like(x_inlet, params_norm['D_xx'])
            q_x_in = torch.full_like(x_inlet, params_norm['q_x'])

            C_m_inlet, _ = model(x_inlet, t_inlet, alpha_in, theta_m_in, theta_im_in,
                                 f_in, rho_in, kd_in, D_xx_in, q_x_in)

            C_inlet_target = torch.where(
                (t_inlet > 0.0) & (t_inlet <= tp),
                torch.ones_like(t_inlet),
                torch.zeros_like(t_inlet)
            )
            loss_bc_inlet = torch.mean((C_m_inlet.squeeze() - C_inlet_target.squeeze()) ** 2)

            # OUTLET BC LOSS
            x_outlet = colloc['outlet_bc']['x'].clone().detach().requires_grad_(True)
            t_outlet = colloc['outlet_bc']['t'].clone().detach()

            alpha_out = torch.full_like(x_outlet, params_norm['alpha'])
            theta_m_out = torch.full_like(x_outlet, params_norm['theta_m'])
            theta_im_out = torch.full_like(x_outlet, params_norm['theta_im'])
            f_out = torch.full_like(x_outlet, params_norm['f'])
            rho_out = torch.full_like(x_outlet, params_norm['rho'])
            kd_out = torch.full_like(x_outlet, params_norm['kd'])
            D_xx_out = torch.full_like(x_outlet, params_norm['D_xx'])
            q_x_out = torch.full_like(x_outlet, params_norm['q_x'])

            C_m_outlet, _ = model(x_outlet, t_outlet, alpha_out, theta_m_out, theta_im_out,
                                  f_out, rho_out, kd_out, D_xx_out, q_x_out)

            dCm_dx_outlet = torch.autograd.grad(C_m_outlet.sum(), x_outlet, create_graph=True)[0]
            loss_bc_outlet = torch.mean(dCm_dx_outlet ** 2)

            # TOTAL LOSS
            loss_total = (w_data * loss_data +
                          w_pde * loss_pde +
                          w_ic * loss_ic +
                          w_bc_in * loss_bc_inlet +
                          w_bc_out * loss_bc_outlet)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            batch_count += 1
            epoch_loss_total += loss_total.item()
            epoch_loss_data += loss_data.item()
            epoch_loss_pde += loss_pde.item()
            epoch_loss_ic += loss_ic.item()
            epoch_loss_bc_inlet += loss_bc_inlet.item()
            epoch_loss_bc_outlet += loss_bc_outlet.item()

        if batch_count > 0:
            epoch_loss_total /= batch_count
            epoch_loss_data /= batch_count
            epoch_loss_pde /= batch_count
            epoch_loss_ic /= batch_count
            epoch_loss_bc_inlet /= batch_count
            epoch_loss_bc_outlet /= batch_count

        # TESTING
        model.eval()
        test_loss = 0
        r2_scores = []

        with torch.no_grad():
            for case in test_data:
                C_obs_test = case['C_obs'].cpu().numpy()
                C_m_test, _ = model(*build_inputs(case))
                C_pred = C_m_test.squeeze().cpu().numpy()

                if not np.isnan(C_pred).any():
                    ss_res = np.sum((C_obs_test - C_pred) ** 2)
                    ss_tot = np.sum((C_obs_test - np.mean(C_obs_test)) ** 2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-10))
                    r2_scores.append(r2)
                    test_loss += np.mean((C_obs_test - C_pred) ** 2)

        model.train()

        if len(r2_scores) > 0:
            test_loss /= len(r2_scores)
            mean_r2 = np.mean(r2_scores)
        else:
            test_loss = np.nan
            mean_r2 = 0.0

        loss_history['total'].append(epoch_loss_total)
        loss_history['data'].append(epoch_loss_data)
        loss_history['pde'].append(epoch_loss_pde)
        loss_history['ic'].append(epoch_loss_ic)
        loss_history['bc_inlet'].append(epoch_loss_bc_inlet)
        loss_history['bc_outlet'].append(epoch_loss_bc_outlet)
        loss_history['test'].append(test_loss)
        loss_history['r2'].append(mean_r2)

        if (epoch + 1) % 200 == 0 or QUICK_TEST:
            elapsed = time.time() - start_time
            print(f"Seed {seed} | Epoch {epoch + 1:4d}/{epochs} | Loss: {epoch_loss_total:.4e} | "
                  f"R²: {mean_r2:.4f} | Time: {elapsed:.0f}s")

    training_time = time.time() - start_time
    return model, loss_history, training_time


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

print("=" * 90)
print("TRAINING COMPREHENSIVE PARAMETERIZED CP-PINN")
print("=" * 90 + "\n")

# Output directory (created up-front so every stage can write into it)
output_dir = f'CP_PINN_COMPREHENSIVE_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
test_dir = os.path.join(output_dir, 'test_results')
os.makedirs(test_dir, exist_ok=True)

# ---- Experiment 1: Seed robustness (3 seeds)
print("EXPERIMENT 1: Seed Robustness Testing (3 seeds)")
print("-" * 90)

seeds = [1234, 1334, 2024]
seed_results = {}
seed_times = {}

for seed in seeds:
    print(f"\nTraining with seed={seed}...")
    model, loss_history, train_time = train_model(seed, epochs=EPOCHS_FULL)
    seed_results[seed] = loss_history
    seed_times[seed] = train_time
    print(f"Seed {seed} training time: {train_time:.1f}s ({train_time / 60:.1f} min)")

timings['seed_robustness'] = sum(seed_times.values())
print(f"\nTotal seed robustness time: {timings['seed_robustness']:.1f}s ({timings['seed_robustness'] / 60:.1f} min)")

# ---- Experiment 2: Loss weight sensitivity
print("\n" + "=" * 90)
print("EXPERIMENT 2: Loss Weight Sensitivity Analysis")
print("-" * 90)

t_sensitivity_start = time.time()

sweep_values = [0.1, 1.0, 10.0]
sweep_specs = [('w_data', 'λ_Data'), ('w_pde', 'λ_PDE'), ('w_ic', 'λ_IC'),
               ('w_bc_in', 'λ_BC_inlet'), ('w_bc_out', 'λ_BC_outlet')]

sensitivity_results = {}
for key, label in sweep_specs:
    print(f"\nSweeping {label}...")
    sensitivity_results[key] = {}
    for val in sweep_values:
        print(f"  {label} = {val}...", end='')
        t_exp = time.time()
        _, lh, _ = train_model(1234, epochs=EPOCHS_SWEEP, **{key: val})
        elapsed = time.time() - t_exp
        sensitivity_results[key][val] = {
            'final_r2': lh['r2'][-1],
            'final_rmse': np.sqrt(lh['test'][-1]),
            'time': elapsed
        }
        print(f" R²={lh['r2'][-1]:.4f}, Time={elapsed:.1f}s")

timings['sensitivity_analysis'] = time.time() - t_sensitivity_start
print(f"\nTotal sensitivity analysis time: {timings['sensitivity_analysis']:.1f}s ({timings['sensitivity_analysis'] / 60:.1f} min)")

# ---- Final full training
print("\n" + "=" * 90)
print(f"FINAL FULL TRAINING: {EPOCHS_FULL} epochs with optimal weights")
print("-" * 90)

t_final_start = time.time()
final_model, final_loss_history, final_train_time = train_model(1234, epochs=EPOCHS_FULL)
timings['final_training'] = time.time() - t_final_start

print(f"\nFinal training complete in {final_train_time:.1f}s ({final_train_time / 60:.1f} min)")

# Persist model + normalisation stats so test BTCs can be regenerated without retraining
torch.save(final_model.state_dict(), os.path.join(output_dir, 'final_model.pt'))
with open(os.path.join(output_dir, 'param_stats.json'), 'w') as fh:
    json.dump({k: {m: float(v) for m, v in s.items()} for k, s in param_stats.items()}, fh, indent=2)
with open(os.path.join(output_dir, 'data_split.json'), 'w') as fh:
    json.dump({'train_cases': list(train_cases), 'test_cases': list(test_cases),
               'natural_sort': NATURAL_SORT}, fh, indent=2)


# ============================================================================
# [A] TEST-BTC PRODUCTION: PREDICTIONS + TIMING TABLE
# ============================================================================

print("\n" + "=" * 90)
print("TEST BTC GENERATION & TIMING TABLE")
print("-" * 90)

final_model.eval()
t_eval_start = time.time()

all_predictions, all_observations = {}, {}
timing_rows = []

with torch.no_grad():
    for case_name, case in zip(test_cases, test_data):
        n_pts = case['x'].shape[0]
        inputs = build_inputs(case)

        # first call (includes any lazy-init / allocator overhead)
        sync(); t0 = time.perf_counter()
        _ = final_model(*inputs)
        sync(); t_first = time.perf_counter() - t0

        # warm-up
        for _ in range(N_WARMUP):
            _ = final_model(*inputs)
        sync()

        # (i) pure forward pass
        fwd = []
        for _ in range(N_REPEAT):
            sync(); t0 = time.perf_counter()
            _ = final_model(*inputs)
            sync(); fwd.append(time.perf_counter() - t0)

        # (ii) end-to-end: build inputs -> forward -> numpy on CPU
        e2e = []
        for _ in range(N_REPEAT):
            sync(); t0 = time.perf_counter()
            C_m_out, _ = final_model(*build_inputs(case))
            C_pred = C_m_out.squeeze().cpu().numpy()
            sync(); e2e.append(time.perf_counter() - t0)

        all_predictions[case_name] = C_pred
        all_observations[case_name] = case['C_obs'].cpu().numpy()

        fwd_ms = np.array(fwd) * 1e3
        e2e_ms = np.array(e2e) * 1e3
        timing_rows.append({
            'Test case': case_name,
            'N points': n_pts,
            'First call (ms)': t_first * 1e3,
            'Forward pass mean (ms)': fwd_ms.mean(),
            'Forward pass std (ms)': fwd_ms.std(),
            'End-to-end mean (ms)': e2e_ms.mean(),
            'End-to-end std (ms)': e2e_ms.std(),
            'Throughput (pts/s)': n_pts / (fwd_ms.mean() / 1e3),
        })

timings['evaluation'] = time.time() - t_eval_start

timing_df = pd.DataFrame(timing_rows)
total_row = {
    'Test case': 'TOTAL (all test BTCs)',
    'N points': int(timing_df['N points'].sum()),
    'First call (ms)': timing_df['First call (ms)'].sum(),
    'Forward pass mean (ms)': timing_df['Forward pass mean (ms)'].sum(),
    'Forward pass std (ms)': np.sqrt((timing_df['Forward pass std (ms)'] ** 2).sum()),
    'End-to-end mean (ms)': timing_df['End-to-end mean (ms)'].sum(),
    'End-to-end std (ms)': np.sqrt((timing_df['End-to-end std (ms)'] ** 2).sum()),
    'Throughput (pts/s)': timing_df['N points'].sum() / (timing_df['Forward pass mean (ms)'].sum() / 1e3),
}
timing_df = pd.concat([timing_df, pd.DataFrame([total_row])], ignore_index=True)
timing_df.to_csv(os.path.join(output_dir, 'Table_test_BTC_timing.csv'), index=False)

print(f"\nDevice: {DEVICE.upper()} | warm-up={N_WARMUP} | repeats={N_REPEAT}\n")
print(timing_df.to_string(index=False, float_format=lambda v: f'{v:,.4f}'))
speedup_note = (f"Training time of the final model: {final_train_time:.1f} s  |  "
                f"predicting all {len(test_cases)} test BTCs: {total_row['End-to-end mean (ms)']:.2f} ms "
                f"(~{final_train_time * 1e3 / total_row['End-to-end mean (ms)']:,.0f}x shorter than training)")
print("\n" + speedup_note)

# PNG version of the table (paper-ready)
fig, ax = plt.subplots(figsize=(15, 0.6 * (len(timing_df) + 3)))
ax.axis('off')
col_labels = list(timing_df.columns)
cell_text = []
for _, r in timing_df.iterrows():
    cell_text.append([
        r['Test case'], f"{int(r['N points'])}", f"{r['First call (ms)']:.3f}",
        f"{r['Forward pass mean (ms)']:.3f}", f"{r['Forward pass std (ms)']:.3f}",
        f"{r['End-to-end mean (ms)']:.3f}", f"{r['End-to-end std (ms)']:.3f}",
        f"{r['Throughput (pts/s)']:,.0f}"])
tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.6)
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#d9e2f3')
    tbl[(0, j)].set_text_props(fontweight='bold')
    tbl[(len(timing_df), j)].set_facecolor('#f2f2f2')
    tbl[(len(timing_df), j)].set_text_props(fontweight='bold')
ax.set_title(f'Time to produce test BTCs ({DEVICE.upper()}, mean of {N_REPEAT} runs)',
             fontsize=13, fontweight='bold', pad=14)
fig.text(0.5, 0.02, speedup_note, ha='center', fontsize=9, style='italic')
plt.savefig(os.path.join(output_dir, '10_Table_Test_BTC_Timing.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\nTotal evaluation block time: {timings['evaluation']:.2f}s")


# ============================================================================
# [C] FULL TEST-RESULT EXPORT (per-case CSV + figures, metrics table)
# ============================================================================

print("\n" + "=" * 90)
print("EXPORTING ALL TEST RESULTS")
print("-" * 90)

t_export_start = time.time()


def compute_metrics(obs, pred):
    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    rmse = np.sqrt(np.mean((obs - pred) ** 2))
    rng = np.max(obs) - np.min(obs)
    return {
        'R2': 1 - ss_res / (ss_tot + 1e-10),
        'RMSE': rmse,
        'MAE': np.mean(np.abs(obs - pred)),
        'MSE': np.mean((obs - pred) ** 2),
        'NRMSE': rmse / rng if rng > 0 else np.nan,
    }


metric_rows = []
combined_frames = []

for idx, (case_name, case) in enumerate(zip(test_cases, test_data)):
    x_v = case['x'].squeeze().cpu().numpy()
    t_v = case['t'].squeeze().cpu().numpy()
    obs_n = all_observations[case_name]
    pred_n = all_predictions[case_name]
    c_max = case['C0']

    order = np.argsort(t_v)
    x_v, t_v, obs_n, pred_n = x_v[order], t_v[order], obs_n[order], pred_n[order]

    res_df = pd.DataFrame({
        'case': case_name, 'x': x_v, 'time': t_v,
        'C_obs_norm': obs_n, 'C_pred_norm': pred_n,
        'C_obs': obs_n * c_max, 'C_pred': pred_n * c_max,
        'abs_error_norm': np.abs(obs_n - pred_n),
        'residual_norm': pred_n - obs_n,
    })
    res_df.to_csv(os.path.join(test_dir, f'{case_name}_BTC_predictions.csv'), index=False)
    combined_frames.append(res_df)

    m = compute_metrics(obs_n, pred_n)
    i_o, i_p = int(np.argmax(obs_n)), int(np.argmax(pred_n))
    m.update({
        'Case': case_name,
        'N points': len(obs_n),
        'C0': c_max,
        'Peak C obs (norm)': obs_n[i_o],
        'Peak C pred (norm)': pred_n[i_p],
        'Peak value error (norm)': pred_n[i_p] - obs_n[i_o],
        'Time to peak obs': t_v[i_o],
        'Time to peak pred': t_v[i_p],
        'Peak time error': t_v[i_p] - t_v[i_o],
        'Inference ms (fwd)': timing_df.loc[idx, 'Forward pass mean (ms)'],
        'Inference ms (end-to-end)': timing_df.loc[idx, 'End-to-end mean (ms)'],
    })
    for k in PARAM_KEYS:
        m[k] = case['params_true'][k]
    metric_rows.append(m)

    # per-case figure: BTC | residuals | parity
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.8))
    ax = axs[0]
    ax.plot(t_v, obs_n * c_max, 'ko', markersize=4, alpha=0.6, label='Observed')
    ax.plot(t_v, pred_n * c_max, 'b-', linewidth=2.2, label='PINN prediction')
    ax.axvline(case['params_true']['tp'], color='gray', ls=':', lw=1.5, label='Pulse end $t_p$')
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Concentration, $C_m$', fontweight='bold')
    ax.set_title(f'{case_name}: BTC (R²={m["R2"]:.4f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axs[1]
    ax.plot(t_v, pred_n - obs_n, 'r-', linewidth=1.6)
    ax.axhline(0, color='k', lw=1)
    ax.set_xlabel('Time', fontweight='bold')
    ax.set_ylabel('Residual (pred − obs, normalised)', fontweight='bold')
    ax.set_title(f'Residuals (RMSE={m["RMSE"]:.4f})', fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax = axs[2]
    ax.scatter(obs_n, pred_n, s=14, alpha=0.6, color='royalblue')
    ax.plot([0, 1], [0, 1], 'r--', lw=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Observed $C_m/C_0$', fontweight='bold')
    ax.set_ylabel('Predicted $C_m/C_0$', fontweight='bold')
    ax.set_title(f'Parity (MAE={m["MAE"]:.4f})', fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(test_dir, f'{case_name}_BTC.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  {case_name}: R²={m['R2']:.4f}  RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  "
          f"peak-time err={m['Peak time error']:.3f}")

combined_df = pd.concat(combined_frames, ignore_index=True)
combined_df.to_csv(os.path.join(test_dir, 'ALL_test_predictions.csv'), index=False)

metrics_df = pd.DataFrame(metric_rows)
front = ['Case', 'N points', 'R2', 'RMSE', 'MAE', 'MSE', 'NRMSE']
metrics_df = metrics_df[front + [c for c in metrics_df.columns if c not in front]]
metrics_df.to_csv(os.path.join(test_dir, 'Table_test_metrics_per_case.csv'), index=False)

# Overall metrics (all test points pooled) -- used by Fig. 5 and the final summary
all_pred_flat = combined_df['C_pred_norm'].values
all_obs_flat = combined_df['C_obs_norm'].values
overall = compute_metrics(all_obs_flat, all_pred_flat)
r2_overall, rmse_overall = overall['R2'], overall['RMSE']
mae_overall, mse_overall, nrmse_overall = overall['MAE'], overall['MSE'], overall['NRMSE']
pd.DataFrame([{**overall, 'N': len(all_obs_flat)}]).to_csv(
    os.path.join(test_dir, 'Table_test_metrics_overall.csv'), index=False)

print("\nPer-case metrics:")
print(metrics_df[front].to_string(index=False, float_format=lambda v: f'{v:.4f}'))
print(f"\nOverall (pooled): R²={r2_overall:.4f}  RMSE={rmse_overall:.4f}  MAE={mae_overall:.4f}")

# The five test cases with the highest R2 (of the 20) are shown in the overview figures; all 20 are exported to the tables
SHOW = [r['Case'] for r in sorted(metric_rows, key=lambda r: -r['R2'])[:5]]
SHOW = sorted(SHOW, key=lambda n: int(n[4:]))
pd.DataFrame({'TC': [f'TC-{i+1}' for i in range(5)], 'case': SHOW}).to_csv(os.path.join(output_dir, 'shown_test_cases.csv'), index=False)
fig, axs = plt.subplots(1, 5, figsize=(4.2 * 5, 4.2), sharey=True)
axs = np.atleast_1d(axs)
for ax, case_name in zip(axs, SHOW):
    row = next(r for r in metric_rows if r['Case'] == case_name)
    d = combined_df[combined_df['case'] == case_name]
    ax.plot(d['time'], d['C_obs_norm'], 'ko', markersize=3.5, alpha=0.55, label='Obs')
    ax.plot(d['time'], d['C_pred_norm'], 'b-', linewidth=2, label='Pred')
    ax.set_title(f'{case_name}\nR²={row["R2"]:.4f}', fontweight='bold')
    ax.set_xlabel('Time', fontweight='bold')
    ax.grid(True, alpha=0.3)
axs[0].set_ylabel('C / C$_{max}$', fontweight='bold')
axs[0].legend(fontsize=9)
fig.suptitle('All test BTCs', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(test_dir, 'ALL_test_BTCs_overview.png'), dpi=300, bbox_inches='tight')
plt.close()

timings['test_export'] = time.time() - t_export_start
print(f"\nAll test results saved to: {test_dir}/  ({timings['test_export']:.1f}s)")


# ============================================================================
# [B] COLLOCATION POINTS IN THE x-t PLANE
# ============================================================================

print("\nGenerating collocation-point plots...")
t_colloc_start = time.time()

ref_case = train_data[0]
ref_params = ref_case['params_true']
ref_name = train_cases[0]
obs_t = ref_case['t'].squeeze().cpu().numpy()
obs_x = ref_case['x'].squeeze().cpu().numpy()


def _pts(c, key):
    return c[key]['x'].squeeze().cpu().numpy(), c[key]['t'].squeeze().cpu().numpy()


def draw_colloc(ax, c, title):
    xi, ti = _pts(c, 'interior')
    xin, tin = _pts(c, 'inlet_bc')
    xo, to = _pts(c, 'outlet_bc')
    xc, tc = _pts(c, 'ic')
    ax.scatter(xi, ti, s=26, c='tab:blue', alpha=0.75, label=f'Interior / PDE (n={len(xi)})')
    ax.scatter(xin, tin, s=34, c='tab:red', marker='s', label=f'Inlet BC, x=0 (n={len(xin)})')
    ax.scatter(xo, to, s=34, c='tab:orange', marker='^', label=f'Outlet BC, x=L (n={len(xo)})')
    ax.scatter(xc, tc, s=34, c='tab:green', marker='D', label=f'IC, t=0 (n={len(xc)})')
    ax.axhline(ref_params['tp'], color='gray', ls=':', lw=1.5, label='$t_p$ (pulse end)')
    ax.set_xlabel('x', fontweight='bold')
    ax.set_ylabel('t', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)


# Fig 08: one realisation of the training collocation set (strategy actually used: uniform)
np.random.seed(0)
c_uni = generate_collocation_points(ref_params, strategy='uniform')
fig, ax = plt.subplots(figsize=(9, 7))
draw_colloc(ax, c_uni, f'Collocation points in x–t plane ({ref_name}, strategy: uniform)')
ax.scatter(obs_x, obs_t, s=14, c='k', marker='x', alpha=0.7,
           label=f'Observed BTC data (n={len(obs_t)})')
ax.legend(fontsize=9, loc='center right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '08_Collocation_Points_xt.png'), dpi=300, bbox_inches='tight')
plt.close()

# Fig 09: all three strategies side by side
strategies = ['uniform', 'additional_less_cluster', 'invariable_less_cluster']
fig, axs = plt.subplots(1, 3, figsize=(20, 6.5), sharey=True)
for ax, strat in zip(axs, strategies):
    np.random.seed(0)
    draw_colloc(ax, generate_collocation_points(ref_params, strategy=strat), strat)
axs[0].legend(fontsize=8, loc='upper left')
fig.suptitle(f'Collocation strategies ({ref_name}; training uses "uniform", resampled every epoch)',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, '09_Collocation_Strategies_xt.png'), dpi=300, bbox_inches='tight')
plt.close()

# Raw collocation points of the plotted realisation, for reuse in the manuscript
rows = []
for key, lab in [('interior', 'interior'), ('inlet_bc', 'inlet_bc'),
                 ('outlet_bc', 'outlet_bc'), ('ic', 'ic')]:
    xs, ts = _pts(c_uni, key)
    rows += [{'type': lab, 'x': a, 't': b} for a, b in zip(xs, ts)]
pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'collocation_points_uniform_sample.csv'), index=False)

timings['collocation_plots'] = time.time() - t_colloc_start
print(f"Plots 08, 09 (collocation points): {timings['collocation_plots']:.2f}s")


# ============================================================================
# ORIGINAL FIGURES (01-07)
# ============================================================================

print("\nGenerating plots...")
t_plots_start = time.time()

# ---- FIGURE 1: Seed Robustness
fig, ax = plt.subplots(figsize=(10, 6))
for seed in seeds:
    ax.semilogy(np.arange(len(seed_results[seed]['total'])), seed_results[seed]['total'],
                label=f'seed={seed}', linewidth=2.5)
ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax.set_ylabel('Loss value', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
plt.tight_layout()
t_fig1_start = time.time()
plt.savefig(f'{output_dir}/01_Seed_Robustness.png', dpi=300, bbox_inches='tight')
timings['plot_01_seed_robustness'] = time.time() - t_fig1_start
print(f"Plot 01 (Seed Robustness): {timings['plot_01_seed_robustness']:.2f}s")
plt.close()

# ---- FIGURE 2: Loss Component Breakdown
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
epochs = np.arange(1, len(final_loss_history['total']) + 1)
t_fig2_start = time.time()

ax = axes[0, 0]
ax.semilogy(epochs, final_loss_history['data'], 'b-', linewidth=2.5)
ax.set_ylabel('Data Loss', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.semilogy(epochs, final_loss_history['pde'], 'g-', linewidth=2.5)
ax.set_ylabel('PDE Loss', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.semilogy(epochs, final_loss_history['ic'], 'purple', linewidth=2.5)
ax.set_ylabel('IC Loss', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.semilogy(epochs, final_loss_history['bc_inlet'], 'red', linewidth=2.5, label='Inlet')
ax.semilogy(epochs, final_loss_history['bc_outlet'], 'orange', linewidth=2.5, label='Outlet')
ax.set_ylabel('BC Loss', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.semilogy(epochs, final_loss_history['total'], 'k-', linewidth=2.5)
ax.set_ylabel('Total Loss', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(epochs, final_loss_history['r2'], 'steelblue', linewidth=2.5)
ax.set_ylabel('Test R²', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12, fontweight='bold'); ax.grid(True, alpha=0.3)

for _ax, _l in zip(axes.ravel(), 'abcdef'):
    _ax.set_title(f'({_l})', loc='left', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/02_Loss_Components.png', dpi=300, bbox_inches='tight')
timings['plot_02_loss_components'] = time.time() - t_fig2_start
print(f"Plot 02 (Loss Components): {timings['plot_02_loss_components']:.2f}s")
plt.close()

# ---- FIGURE 3: Loss & R² Convergence
fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))
t_fig3_start = time.time()

ax_left.semilogy(epochs, final_loss_history['data'], 'b-', linewidth=2.5, label='Train Data Loss')
ax_left.semilogy(epochs, final_loss_history['test'], 'r--', linewidth=2.5, label='Test Data Loss')
ax_left.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax_left.set_ylabel('MSE scaled concentration, log', fontsize=12, fontweight='bold')
ax_left.grid(True, alpha=0.3)
ax_left.legend(fontsize=11)
ax_left.set_title('(a)', loc='left', fontsize=13, fontweight='bold')

ax_right.plot(epochs, final_loss_history['r2'], 'go-', linewidth=2.5, markersize=3)
ax_right.set_xlabel('Epoch', fontsize=12, fontweight='bold')
ax_right.set_ylabel('Test R²', fontsize=12, fontweight='bold')
ax_right.grid(True, alpha=0.3)
ax_right.set_title('(b)', loc='left', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/03_Convergence_Analysis.png', dpi=300, bbox_inches='tight')
timings['plot_03_convergence'] = time.time() - t_fig3_start
print(f"Plot 03 (Convergence Analysis): {timings['plot_03_convergence']:.2f}s")
plt.close()

# ---- FIGURE 4: Loss Weight Sensitivity (baseline = measured value at multiplier 1.0)
fig, axes = plt.subplots(2, 5, figsize=(18, 10))
t_fig4_start = time.time()

multipliers = sweep_values
weight_names = [lab for _, lab in sweep_specs]
weight_keys = [k for k, _ in sweep_specs]

for idx, (weight_name, weight_key) in enumerate(zip(weight_names, weight_keys)):
    base_r2 = sensitivity_results[weight_key][1.0]['final_r2']
    ax = axes[0, idx]
    r2_values = [sensitivity_results[weight_key][w]['final_r2'] for w in multipliers]
    ax.plot(np.log10(multipliers), r2_values, 'bo-', linewidth=2.5, markersize=10)
    ax.axhline(y=base_r2, color='r', linestyle='--', linewidth=2, label='Baseline (×1)')
    ax.set_xlabel('Multiplier of baseline value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test R²', fontsize=11, fontweight='bold')
    ax.set_xticks(np.log10(multipliers))
    ax.set_xticklabels(['10⁻¹', '10⁰', '10¹'])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    ax.set_title(f'{weight_name} sweep', fontsize=12, fontweight='bold')

for idx, (weight_name, weight_key) in enumerate(zip(weight_names, weight_keys)):
    ax = axes[1, idx]
    rmse_values = [sensitivity_results[weight_key][w]['final_rmse'] for w in multipliers]
    ax.plot(np.log10(multipliers), rmse_values, 'o-', color='orange', linewidth=2.5, markersize=10)
    ax.set_xlabel('Multiplier of baseline value', fontsize=11, fontweight='bold')
    ax.set_ylabel('Test RMSE', fontsize=11, fontweight='bold')
    ax.set_xticks(np.log10(multipliers))
    ax.set_xticklabels(['10⁻¹', '10⁰', '10¹'])
    ax.grid(True, alpha=0.3)

fig.suptitle('Loss Weight Sensitivity Analysis (λ sweeps)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/04_Loss_Weight_Sensitivity.png', dpi=300, bbox_inches='tight')
timings['plot_04_sensitivity'] = time.time() - t_fig4_start
print(f"Plot 04 (Loss Weight Sensitivity): {timings['plot_04_sensitivity']:.2f}s")
plt.close()

pd.DataFrame([{'weight': k, 'multiplier': w, **v}
              for k, d in sensitivity_results.items() for w, v in d.items()]
             ).to_csv(os.path.join(output_dir, 'Table_loss_weight_sensitivity.csv'), index=False)

# ---- FIGURE 5: Predictions vs Observations Scatter Plot
fig, ax = plt.subplots(figsize=(10, 10))
t_fig5_start = time.time()

ax.scatter(all_obs_flat, all_pred_flat, alpha=0.5, s=20, color='blue')
ax.plot([0, 1], [0, 1], 'r--', linewidth=2.5, label='1:1 Line')
ax.set_xlabel('Observed $C_m/C_0$', fontsize=13, fontweight='bold')
ax.set_ylabel('Predicted $C_m/C_0$', fontsize=13, fontweight='bold')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3)

textstr = (f'R² = {r2_overall:.4f}\nRMSE = {rmse_overall:.4f}\nMAE = {mae_overall:.4f}\n'
           f'MSE = {mse_overall:.2e}\nNRMSE = {nrmse_overall:.4f}\nN = {len(all_obs_flat):,}')
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12, verticalalignment='top',
        bbox=dict(boxstyle='square', facecolor='white', edgecolor='0.5', alpha=1.0), family='monospace')
plt.tight_layout()
plt.savefig(f'{output_dir}/05_Predictions_vs_Observations.png', dpi=300, bbox_inches='tight')
timings['plot_05_scatter'] = time.time() - t_fig5_start
print(f"Plot 05 (Predictions vs Observations): {timings['plot_05_scatter']:.2f}s")
plt.close()

# ---- FIGURE 6: Test Predictions on Individual Cases
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
axes = np.atleast_1d(axes)
t_fig6_start = time.time()

test_r2_final = [compute_metrics(combined_df[combined_df['case'] == c]['C_obs_norm'].values, combined_df[combined_df['case'] == c]['C_pred_norm'].values)['R2'] for c in test_cases]
_unused = []
for idx, case_name in enumerate(SHOW):
    ax = axes[idx]
    d = combined_df[combined_df['case'] == case_name]
    t_sorted, C_obs_sorted, C_pred_sorted = d['time'].values, d['C_obs_norm'].values, d['C_pred_norm'].values

    r2_case = compute_metrics(C_obs_sorted, C_pred_sorted)['R2']
    _unused.append(r2_case)

    ax.plot(t_sorted, C_obs_sorted, 'ko', markersize=5, alpha=0.6, label='Obs')
    ax.plot(t_sorted, C_pred_sorted, 'b-', linewidth=2, label='Pred')
    ax.set_title(f'{case_name}\nR²={r2_case:.4f}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(fontsize=10)
    ax.set_ylabel('C (normalized)', fontsize=10, fontweight='bold')

fig.suptitle(f'Parameterized CP-PINN: Test Case Predictions ({", ".join(SHOW)})',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/06_Test_Predictions.png', dpi=300, bbox_inches='tight')
timings['plot_06_test_predictions'] = time.time() - t_fig6_start
print(f"Plot 06 (Test Predictions): {timings['plot_06_test_predictions']:.2f}s")
plt.close()

# ---- FIGURE 7: PDE Weight Effect (measured, from the λ_PDE sweep)
fig, ax = plt.subplots(figsize=(10, 6))
t_fig7_start = time.time()

pde_mult = sweep_values
r2_vals = [sensitivity_results['w_pde'][w]['final_r2'] for w in pde_mult]
ax.plot(np.log10(pde_mult), r2_vals, 'o-', color='orange', linewidth=2.5, markersize=12)
ax.axhline(y=sensitivity_results['w_pde'][1.0]['final_r2'], color='r', linestyle='--',
           linewidth=2, label='Baseline (×1)')
ax.set_xlabel('PDE weight multiplier (λ_PDE)', fontsize=13, fontweight='bold')
ax.set_ylabel('Test R²', fontsize=13, fontweight='bold')
ax.set_xticks(np.log10(pde_mult))
ax.set_xticklabels(['10⁻¹', '10⁰', '10¹'])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=12)
ax.set_title('Effect of PDE Weight on Model Performance (measured)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/07_PDE_Weight_Effect.png', dpi=300, bbox_inches='tight')
timings['plot_07_pde_weight'] = time.time() - t_fig7_start
print(f"Plot 07 (PDE Weight Effect): {timings['plot_07_pde_weight']:.2f}s")
plt.close()

timings['all_plots'] = time.time() - t_plots_start
print(f"\nTotal plotting time: {timings['all_plots']:.1f}s ({timings['all_plots'] / 60:.1f} min)")


# ============================================================================
# FINAL SUMMARY & TIMING REPORT
# ============================================================================

print("\n" + "=" * 90)
print("COMPREHENSIVE TIMING REPORT")
print("=" * 90)

total_time = (timings['data_loading'] + timings['seed_robustness'] + timings['sensitivity_analysis'] +
              timings['final_training'] + timings['evaluation'] + timings['test_export'] +
              timings['collocation_plots'] + timings['all_plots'])

print("\nCOMPONENT TIMINGS:")
print(f"  1. Data Loading:            {timings['data_loading']:8.2f}s")
print(f"  2. Seed Robustness (3x):    {timings['seed_robustness']:8.1f}s ({timings['seed_robustness'] / 60:5.1f} min)")
print(f"  3. Sensitivity Analysis:    {timings['sensitivity_analysis']:8.1f}s ({timings['sensitivity_analysis'] / 60:5.1f} min)")
print(f"  4. Final Training:          {timings['final_training']:8.1f}s ({timings['final_training'] / 60:5.1f} min)")
print(f"  5. Test BTC timing block:   {timings['evaluation']:8.2f}s")
print(f"  6. Test results export:     {timings['test_export']:8.2f}s")
print(f"  7. Collocation plots:       {timings['collocation_plots']:8.2f}s")
print(f"  8. Plotting (figs 01-07):   {timings['all_plots']:8.1f}s ({timings['all_plots'] / 60:5.1f} min)")
print(f"\n  TOTAL TIME:                 {total_time:8.1f}s ({total_time / 60:5.1f} min, {total_time / 3600:4.2f} hours)")

pd.DataFrame([{'component': k, 'seconds': v} for k, v in timings.items()]).to_csv(
    os.path.join(output_dir, 'Table_component_timings.csv'), index=False)

print("\nFINAL RESULTS:")
print(f"  Mean Test R² (per-case): {np.mean(test_r2_final):.4f} ± {np.std(test_r2_final):.4f}")
print(f"  Overall R² (pooled):     {r2_overall:.4f}")
print(f"  Overall RMSE:            {rmse_overall:.4f}")
print(f"  Test cases: {', '.join(test_cases)}")

print(f"\nResults saved to: {output_dir}/")
print(f"   test_results/ : per-case CSV + BTC figure, ALL_test_predictions.csv, metric tables")
print(f"\n{'=' * 90}\n")