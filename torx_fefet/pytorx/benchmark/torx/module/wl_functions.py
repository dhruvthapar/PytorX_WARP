import torch
import sys
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[6]))
from mapping_evaluation import *
import time
import json

def assign_wl_lsb_workload(crxb_col, crxb_row, crxb_size, wl_lsb,
                           alpha=(1.0, 1.0, 1.0), device=None, seed=1):
    assert 0 <= wl_lsb <= 8, "wl_lsb must be in [0, 8]"
    assert crxb_size % 8 == 0, "crxb_size must be divisible by 8"

    if seed is not None:
        torch.manual_seed(seed)

    wl_map = torch.zeros((crxb_col, crxb_row, crxb_size, crxb_size, 3),
                         device=device, dtype=torch.float32)

    if wl_lsb == 0:
        return wl_map

    dirichlet = torch.distributions.Dirichlet(torch.tensor(alpha, device=device))

    col = torch.arange(crxb_size, device=device)
    active_cols = (col % 8) >= (8 - wl_lsb)   # last wl_lsb cols in each 8-col block
    n_active_cols = int(active_cols.sum().item())

    N = crxb_col * crxb_row * crxb_size * n_active_cols
    samples = dirichlet.sample((N,))          # (N,3)

    wl_map[..., active_cols, :] = samples.view(crxb_col, crxb_row, crxb_size, n_active_cols, 3)
    return wl_map

def assign_wl_lsb_uniform_workload(crxb_col, crxb_row, crxb_size, wl_lsb,
                                   device, dtype=torch.float32, seed = None):
    """
    Active cells are the last wl_lsb columns in each 8-column block (per row).
    Every active cell gets the SAME wl_vec (len-3). Inactive cells are (0,0,0).
    """
    assert 0 <= wl_lsb <= 8, "wl_lsb must be in [0, 8]"
    assert crxb_size % 8 == 0, "crxb_size must be divisible by 8"
    wl_vec = (0.333333,0.333333,0.333333)

    wl_map = torch.zeros((crxb_col, crxb_row, crxb_size, crxb_size, 3),
                         device=device, dtype=dtype)
    if wl_lsb == 0:
        return wl_map

    wl_vec = torch.as_tensor(wl_vec, device=device, dtype=dtype).view(1, 1, 1, 1, 3)

    col = torch.arange(crxb_size, device=device)
    active_cols = (col % 8) >= (8 - wl_lsb)   # last wl_lsb cols in each 8-col block

    wl_map[..., active_cols, :] = wl_vec      # broadcasts to all active positions
    return wl_map

import torch
import torch.nn.functional as F
import math

def generate_region_map(wl_map, row_cells_per_region, col_cells_per_region):
    crxb_col, crxb_row, crxb_size_r, crxb_size_c, D = wl_map.shape
    assert crxb_size_r == crxb_size_c

    H = crxb_row * crxb_size_r
    W = crxb_col * crxb_size_c

    rpr = row_cells_per_region
    cpr = col_cells_per_region

    # number of regions (allow partial last region)
    R1 = (H + rpr - 1) // rpr   # ceil(H/rpr)
    R2 = (W + cpr - 1) // cpr   # ceil(W/cpr)

    H_pad = R1 * rpr
    W_pad = R2 * cpr
    pad_h = H_pad - H
    pad_w = W_pad - W

    # full_map: (H, W, D)
    full_map = wl_map.permute(1, 3, 0, 2, 4).reshape(H, W, D)

    # active mask per cell: (H, W)
    active = (full_map.abs().sum(dim=-1) > 0)

    # pad bottom/right with zeros / False so partial regions work
    if pad_h or pad_w:
        # pad format for F.pad on (H,W,D) is (D_left,D_right, W_left,W_right, H_left,H_right)
        full_map = F.pad(full_map, (0, 0, 0, pad_w, 0, pad_h), mode="constant", value=0)
        active   = F.pad(active,   (0, pad_w, 0, pad_h),      mode="constant", value=False)

    # reshape into regions: (R1, rpr, R2, cpr, D)
    cells = full_map.view(R1, rpr, R2, cpr, D)

    # mask: (R1, rpr, R2, cpr, 1) as float
    mask = active.view(R1, rpr, R2, cpr).unsqueeze(-1).to(full_map.dtype)

    # sum only active cells, count active cells
    sum_active = (cells * mask).sum(dim=(1, 3))              # (R1, R2, D)
    cnt_active = mask.sum(dim=(1, 3)).clamp_min(1.0)         # (R1, R2, 1)

    rmap = sum_active / cnt_active                           # (R1, R2, D)

    # regions with zero active cells -> exactly 0
    has_any = (mask.sum(dim=(1, 3)) > 0)                     # (R1, R2, 1) bool
    rmap = torch.where(has_any, rmap, torch.zeros_like(rmap))

    active_crop = active[:H, :W]
    active_mask = active_crop.view(crxb_row, crxb_size_r, crxb_col, crxb_size_c).permute(2, 0, 3, 1).contiguous()
    # shape: (crxb_col, crxb_row, crxb_size, crxb_size)
    return rmap, active_mask

sigma_dict = {
    "I0": {
        1000:      1.77e-07,
        10000:     1.77e-07,
        100000:    8.50e-08,
        1000000:   3.58e-08,
        10000000:  4.40e-08,
    },
    "I1": {
        1000:      1.26e-06,
        10000:     1.28e-06,
        100000:    1.22e-06,
        1000000:   9.74e-07,
        10000000:  7.17e-07,
    },
    "I2": {
        1000:      1.02e-06,
        10000:     1.10e-06,
        100000:    1.22e-06,
        1000000:   1.20e-06,
        10000000:  9.12e-07,
    },
    "I3": {
        1000:      6.85e-07,
        10000:     6.76e-07,
        100000:    6.80e-07,
        1000000:   7.33e-07,
        10000000:  7.60e-07,
    }
}

def get_sigma_I(sigma_dict, num_cycles):
    """
    Returns sigma_I vector in the fixed order:
    [I0, I1, I2, I3]
    """
    return (
        sigma_dict["I0"][num_cycles],
        sigma_dict["I1"][num_cycles],
        sigma_dict["I2"][num_cycles],
        sigma_dict["I3"][num_cycles],
    )

def predict_mean_I_from_fits(fits, X_new=None, C=None):
    n01, n02, n03 = X_new
    I_lists = ("I0_list", "I1_list", "I2_list", "I3_list")
    mu_dict = {}
    for I in I_lists:
        I_fit = fits[I][C] # using the fit function at that point
        mu = I_fit(n02, n03).item()  # scalar
        mu_dict[I] = max(mu, 1e-18)
    return mu_dict["I0_list"], mu_dict["I1_list"], mu_dict["I2_list"], mu_dict["I3_list"]

def degraded_conductances_per_cell(
    wl_map, vread, fits, num_cycles,
    base_mean_G, base_sigma_G,
    device, var_case
):
    new_shape = wl_map.shape[:-1]
    base_mean_G  = torch.as_tensor(base_mean_G,  device=device, dtype=torch.float32)  # (4,)
    base_sigma_G = torch.as_tensor(base_sigma_G, device=device, dtype=torch.float32)  # (4,)
    vread = torch.as_tensor(vread, device=device, dtype=torch.float32)

    mean_G = base_mean_G.view(*([1]*len(new_shape)), 4).expand(*new_shape, 4).clone()
    sig_G  = base_sigma_G.view(*([1]*len(new_shape)), 4).expand(*new_shape, 4).clone()

    active = (wl_map.abs().sum(dim=-1) > 0)                      # (...,)
    if not active.any():
        return mean_G, sig_G

    # Grab ALL active workload vectors at once: (N,3)
    X = wl_map[active].float()
    n02 = X[:, 1]
    n03 = X[:, 2]

    I_lists = ("I0_list", "I1_list", "I2_list", "I3_list")

    # Predict ALL means in one shot per I (4 calls total, not N calls)
    mean_I_cols = []
    for I in I_lists:
        I_fit = fits[I][num_cycles]                              # torch-native callable
        mean_I_cols.append(I_fit(n02, n03).reshape(-1))
    mean_I = torch.stack(mean_I_cols, dim=-1)                    # (N,4)

    mean_G_active = mean_I / vread                               # (N,4)

    # Sigma (vectorized)
    if var_case == "default":
        # expect sigma_dict[I][num_cycles] or sigma_dict[I][C] — adapt this line to your getter
        # If your get_sigma_I returns 4 scalars, do:
        sigma_I0, sigma_I1, sigma_I2, sigma_I3 = get_sigma_I(sigma_dict, num_cycles)
        sigma_I = torch.tensor([sigma_I0, sigma_I1, sigma_I2, sigma_I3],
                               device=device, dtype=torch.float32).view(1,4).expand_as(mean_I)
    elif var_case == "var1":
        sigma_I = 0.01 * mean_I
    elif var_case == "var2":
        sigma_I = 0.05 * mean_I
    else:
        raise ValueError(f"Unknown var_case: {var_case}")

    sig_G_active = sigma_I / vread                               # (N,4)

    # Assign back in one shot (no loop)
    mean_G[active] = mean_G_active
    sig_G[active]  = sig_G_active

    return mean_G, sig_G


def compute_iread_increase(mean_I0, mean_I1, mean_I2, mean_I3): # for nlf = 0
    delta_i1 = (mean_I0 + (mean_I3 - mean_I0)/3)   - mean_I1
    delta_i2 = (mean_I0 + 2*(mean_I3 - mean_I0)/3) - mean_I2
    return delta_i1, delta_i2

with open('/home/dthapar1/CHIMES/aspdac/saved_dicts/rep_info.pkl', 'rb') as f:
    rep_info = pickle.load(f)

def warp_conductances_per_cell(
    wl_map, vread, num_cycles, fits,
    rmap, row_cells_per_region, col_cells_per_region,
    base_mean_G, base_sigma_G, device, var_case, config_path, layer_count
):
    crxb_col, crxb_row, crxb_size_r, crxb_size_c, D = wl_map.shape
    assert D == 3
    assert crxb_size_r == crxb_size_c
    crxb_size = crxb_size_r

    base_mean_G  = torch.as_tensor(base_mean_G,  device=device, dtype=torch.float32)  # (4,)
    base_sigma_G = torch.as_tensor(base_sigma_G, device=device, dtype=torch.float32)  # (4,)
    vread = torch.as_tensor(vread, device=device, dtype=torch.float32)

    new_shape = (crxb_col, crxb_row, crxb_size, crxb_size)
    mean_G = base_mean_G.view(1,1,1,1,4).expand(*new_shape, 4).clone()
    sig_G  = base_sigma_G.view(1,1,1,1,4).expand(*new_shape, 4).clone()

    active = (wl_map.abs().sum(dim=-1) > 0)
    if not active.any():
        return mean_G, sig_G

    # ---- active wl vectors: (N,3)
    X = wl_map[active].float()
    n02 = X[:, 1]
    n03 = X[:, 2]

    # ---- baseline mean_I for active cells: (N,4)
    I_lists = ("I0_list", "I1_list", "I2_list", "I3_list")
    mean_I_cols = []
    for I in I_lists:
        I_fit = fits[I][num_cycles]          # must be torch-native
        mean_I_cols.append(I_fit(n02, n03).reshape(-1))
    mean_I = torch.stack(mean_I_cols, dim=-1)  # (N,4)

    # ---- region indices for each active cell
    # active_idx in tensor form (NO tolist): (N,4) [i_cb, j_cb, r, c]
    active_idx = active.nonzero(as_tuple=False)
    i_cb = active_idx[:, 0]
    j_cb = active_idx[:, 1]
    r    = active_idx[:, 2]
    c    = active_idx[:, 3]

    global_row = j_cb * crxb_size + r
    global_col = i_cb * crxb_size + c
    ri = torch.div(global_row, row_cells_per_region, rounding_mode="floor")
    rj = torch.div(global_col, col_cells_per_region, rounding_mode="floor")

    # ---- build rep_map (prefer caching outside if possible)
    R1, R2, _ = rmap.shape
    rep_map = torch.empty((R1, R2, 3), device=device, dtype=torch.float32)

    default_nc = 10**7
    default_I = "I2_list"
    rep_wls = rep_info[default_nc][default_I]["rep_wl"]
    i3_rep_by_cycles = rep_info[default_nc][default_I]["i3_rep"]   # dict: cycles -> array(len=R)

    rep_dict = {wl: {nc: float(i3_rep_by_cycles[nc][k]) for nc in i3_rep_by_cycles.keys()} for k, wl in enumerate(rep_wls)}
    gmax_values = []
    for _ri in range(R1):
        for _rj in range(R2):
            rep_wl = map_any_wl_to_wlrep(rmap[_ri, _rj], rep_info, fits[default_I][default_nc], default_nc, default_I)
            i3_rep = rep_dict[rep_wl][num_cycles]
            gmax_values.append(i3_rep/vread) # get rep_wl[num_cycles][I3_list] for all regions
            rep_map[_ri, _rj] = torch.as_tensor(rep_wl, device=device, dtype=torch.float32)
    gmax_values = torch.stack(gmax_values) 
    gmax = gmax_values.max() # or np.max(gmax_values)
    with open(f"{config_path}/config_{layer_count}.json", 'r') as f:
        config = json.load(f)
    config['gmax'] = gmax.item()
    with open(f"{config_path}/config_{layer_count}.json", 'w') as f:
        json.dump(config, f, indent=4)
    #update config_layer file
    # during forward use that config for each layer. -> recompute warp_accuracy

    # ---- representative wl_vec per active cell: (N,3)
    rep_X = rep_map[ri, rj]     # advanced indexing

    rep_n02 = rep_X[:, 1]
    rep_n03 = rep_X[:, 2]

    # ---- rep mean_I per active cell region: (N,4)
    rep_mean_I_cols = []
    for I in I_lists:
        I_fit = fits[I][num_cycles]
        rep_mean_I_cols.append(I_fit(rep_n02, rep_n03).reshape(-1))
    rep_mean_I = torch.stack(rep_mean_I_cols, dim=-1)  # (N,4)

    # ---- compute delta_i1, delta_i2 (vectorized)
    # You MUST ensure compute_iread_increase is torch-elementwise.
    delta_i1, delta_i2 = compute_iread_increase(
        rep_mean_I[:, 0], rep_mean_I[:, 1], rep_mean_I[:, 2], rep_mean_I[:, 3]
    )  # each (N,)

    # adjust only I1 and I2
    mean_I_adj = mean_I.clone()
    mean_I_adj[:, 1] = mean_I_adj[:, 1] + delta_i1
    mean_I_adj[:, 2] = mean_I_adj[:, 2] + delta_i2

    # ---- sigma (N,4)
    if var_case == "default":
        sigma_I0, sigma_I1, sigma_I2, sigma_I3 = get_sigma_I(sigma_dict, num_cycles)
        sigma_I = torch.tensor([sigma_I0, sigma_I1, sigma_I2, sigma_I3],
                               device=device, dtype=torch.float32).view(1,4).expand_as(mean_I_adj)
    elif var_case == "var1":
        sigma_I = 0.01 * mean_I_adj
    elif var_case == "var2":
        sigma_I = 0.05 * mean_I_adj
    else:
        raise ValueError(f"Unknown var_case: {var_case}")

    # ---- I -> G
    mean_G_active = mean_I_adj / vread
    sig_G_active  = sigma_I    / vread

    # ---- write back in one shot
    mean_G[active] = mean_G_active
    sig_G[active]  = sig_G_active

    return mean_G, sig_G

# def assign_wl_lsb_workload(
#     crxb_col,
#     crxb_row,
#     crxb_size,
#     wl_lsb,
#     alpha=(5.0, 2.5, 1.0),
#     device="cpu",
#     seed=1,
# ):
#     """
#     Returns:
#         wl_tensor of shape:
#         (crxb_col, crxb_row, crxb_size, crxb_size, 3)

#     Workload is assigned only to wl_lsb cells out of every
#     8-column group in each row of each crossbar.
#     """

#     assert 0 <= wl_lsb <= 8, "wl_lsb must be in [1, 8]"
#     assert crxb_size % 8 == 0, "crxb_size must be divisible by 8"

#     if seed is not None:
#         torch.manual_seed(seed)

#     # Initialize workload tensor
#     wl_map = torch.zeros(
#         (crxb_col, crxb_row, crxb_size, crxb_size, 3),
#         device=device,
#         dtype=torch.float32,
#     )
    
#     # Dirichlet distribution for (n_01, n_02, n_03)
#     dirichlet = torch.distributions.Dirichlet(
#         torch.tensor(alpha, device=device)
#     )
#     print(crxb_col*crxb_row*crxb_size*crxb_size)
#     for i in range(crxb_col):
#         for j in range(crxb_row):
#             for r in range(crxb_size):
#                 # iterate over 8-column blocks
#                 for c0 in range(0, crxb_size, 8):
#                     # last wl_lsb columns in this block
#                     start = c0 + (8 - wl_lsb)
#                     end = c0 + 8

#                     for c in range(start, end):
#                         wl_map[i, j, r, c, :] = dirichlet.sample()
#     print('done')
#     return wl_map

# def generate_region_map(wl_map, row_cells_per_region, col_cells_per_region):
#     """
#     Args:
#         wl_map: Tensor of shape
#             (crxb_col, crxb_row, crxb_size, crxb_size, D)
#         row_cells_per_region: number of cells per region along rows
#         col_cells_per_region: number of cells per region along columns

#     Returns:
#         rmap: Tensor of shape (R1, R2, D)
#               averaged only over active cells
#     """
#     crxb_col, crxb_row, crxb_size_r, crxb_size_c, D = wl_map.shape
#     assert crxb_size_r == crxb_size_c

#     H = crxb_row * crxb_size_r   # total rows
#     W = crxb_col * crxb_size_c   # total columns
#     assert H % row_cells_per_region == 0
#     assert W % col_cells_per_region == 0

#     R1 = H // row_cells_per_region
#     R2 = W // col_cells_per_region

#     # reshape into full 2D grid of cells
#     full_map = wl_map.permute(1, 3, 0, 2, 4).reshape(H, W, D)
#     # (row, col, D)

#     # active mask: True where workload != (0,0,0)
#     active = (full_map.abs().sum(dim=-1) > 0)  # (H, W)
#     rmap = wl_map.new_zeros((R1, R2, D))
#     for i in range(R1):
#         for j in range(R2):
#             r_start = i * row_cells_per_region
#             r_end   = (i + 1) * row_cells_per_region
#             c_start = j * col_cells_per_region
#             c_end   = (j + 1) * col_cells_per_region
#             region_cells = full_map[r_start:r_end, c_start:c_end, :]
#             region_active = active[r_start:r_end, c_start:c_end]
#             if region_active.any():
#                 # average only over active cells
#                 rmap[i, j] = region_cells[region_active].mean(dim=0)
#             else:
#                 # no active cells → keep zero (or set a default if you prefer)
#                 rmap[i, j] = 0.0
#     return rmap