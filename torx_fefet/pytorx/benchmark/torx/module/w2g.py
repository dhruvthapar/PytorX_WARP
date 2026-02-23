# Copyright 2019 The PytorX Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F

from .wl_functions import *
from .fault_injection import Fault

def x_relu(input):
    return input.clamp(min=0)
       
def bitslicer(input_tensor):
    # Ensure the new tensors created are on the same device as the input tensor
    device = input_tensor.device
    
    size = input_tensor.size()
    tensor_int16 = input_tensor.to(torch.int16)

    # Create a list to hold the 2-bit segments
    segments = []
    for index in range(7, -1, -1):
        shifted_tensor = tensor_int16 >> (2 * index)
        segment = shifted_tensor & 0b11
        # Make sure the segment is created on the correct device
        segments.append(segment.to(device))

    # Stack the segments along a new dimension and ensure it's on the correct device
    segments_tensor = torch.stack(segments, dim=-1).to(device).float()

    # Reshape the segments tensor to the desired shape
    reshaped_tensor = segments_tensor.transpose(3, 4).reshape(size[0], size[1], -1, size[3])

    return reshaped_tensor

class w2g(nn.Module):
    '''
    perfrom the weight conversion within this function, which convert the 
    post-quantization fixed point weight (weight_hat) into a pair of
    conductance values. output[0] is the G_pos and output[1] is the G_neg
    '''
    def __init__(self, Gmin, G00_mu, G01_mu, G10_mu, G11_mu, G00_sigma, G01_sigma, G10_sigma, G11_sigma, weight_shape, device=None, layer_count=0, wl_lsb = 1, num_cycles=10**3, fits = None, warp = None, var_case = None, config_path = None, G00_f_mu=None, G00_f_sigma=None, G01_f_mu=None, G01_f_sigma=None, G10_f_mu=None, G10_f_sigma =None, G11_f_mu=None, G11_f_sigma=None, fault_model=None, enable_fault=None, fault_rate=None, fault_dist=None, faulty_layer_idx=None, num_crossbars=None):
        super(w2g, self).__init__()
        #self.device =  device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_path = config_path
        self.Gmin = Gmin
        self.G00_mu = G00_mu
        self.G01_mu = G01_mu
        self.G10_mu = G10_mu
        self.G11_mu = G11_mu
        if var_case == "default":
            self.G00_sigma = G00_sigma
            self.G01_sigma = G01_sigma
            self.G10_sigma = G10_sigma
            self.G11_sigma = G11_sigma
        if var_case == "var1":
            self.G00_sigma = 0.01*G00_mu
            self.G01_sigma = 0.01*G01_mu
            self.G10_sigma = 0.01*G10_mu
            self.G11_sigma = 0.01*G11_mu
        if var_case == "var2":
            self.G00_sigma = 0.05*G00_mu
            self.G01_sigma = 0.05*G01_mu
            self.G10_sigma = 0.05*G10_mu
            self.G11_sigma = 0.05*G11_mu

        self.device =device
        self.config_path = config_path
        # fault injection properties
        self.enable_fault = enable_fault
        self.fault_dist = fault_dist
        self.fault_rate = fault_rate
        self.layer_count = layer_count
        self.faulty_layer_idx = faulty_layer_idx
        self.num_crossbars = num_crossbars

        self.active_mask = None
        shape_list = list(weight_shape)  # Convert the tuple to a list
        shape_list[2] = weight_shape[2] * 8 # Modify the desired element at index 2
        
        self.new_shape = torch.Size(shape_list)  # Convert the list back to a tuple
        self.G_pos = torch.zeros(self.new_shape).to(self.device)
        self.G_neg = torch.zeros(self.new_shape).to(self.device)
        #new_shape = (crxb_col, crxb_row, crxb_size, crxb_size) #8,1,64,64
        crxb_col = self.new_shape[0];crxb_row = self.new_shape[1];crxb_size = self.new_shape[3]
        self.fault_model = fault_model
        if self.fault_model == "charge_trapping":
            self.wl_lsb = wl_lsb
            self.wl_map = assign_wl_lsb_workload(
                crxb_col=crxb_col,
                crxb_row=crxb_row,
                crxb_size=crxb_size,
                wl_lsb=self.wl_lsb,
                device=device,
                seed=1,
            )
            base_mean_G = [self.G00_mu, self.G01_mu, self.G10_mu, self.G11_mu]
            base_sigma_G = [self.G00_sigma, self.G01_sigma, self.G10_sigma, self.G11_sigma]
            if not warp:
                self.mean_G, self.sig_G = degraded_conductances_per_cell(
                    wl_map=self.wl_map,               # (crxb_col, crxb_row, crxb_size, crxb_size, 3)
                    vread=0.75,                # self.vread
                    fits=fits,            # callable: mean_I0,mean_I1,mean_I2,mean_I3 = predict_I(wl_vec, C)
                    num_cycles=num_cycles,           # dict to fetch sigmas for I0..I3 (see sigma_key_fn below)
                    base_mean_G=base_mean_G,          # (4,) tensor/list: [mean_G00, mean_G01, mean_G10, mean_G11] for wl=(0,0,0)
                    base_sigma_G=base_sigma_G,         # (4,) tensor/list: [sig_G00,  sig_G01,  sig_G10,  sig_G11]  for wl=(0,0,0)
                    device=self.device,
                    var_case=var_case
                )
            else: #WRITE ELSE FUNCTION WITH THE PSEUDO CODE
                row_cells_per_region = min(int(64), crxb_row*crxb_size)
                col_cells_per_region = min(int(64*8/wl_lsb), crxb_col*crxb_size)
                self.rmap, self.active_mask = generate_region_map(self.wl_map, row_cells_per_region, col_cells_per_region)
                self.mean_G, self.sig_G = warp_conductances_per_cell(
                    wl_map=self.wl_map,                # (crxb_col, crxb_row, crxb_size, crxb_size, 3)
                    vread=0.75,
                    num_cycles=num_cycles,            # e.g., 1000/10000/...
                    fits=fits,             # callable: (I0,I1,I2,I3)=predict_I(wl_vec, C)
                    rmap=self.rmap,                  # (R1, R2, 3) active-only region averages (from generate_region_map)
                    row_cells_per_region=row_cells_per_region,
                    col_cells_per_region=col_cells_per_region,
                    base_mean_G=base_mean_G,           # length-4: [mean_G00, mean_G01, mean_G10, mean_G11] for inactive cells
                    base_sigma_G=base_sigma_G,          # length-4: [sig_G00,  sig_G01,  sig_G10,  sig_G11]  for inactive cells
                    device=self.device,
                    var_case=var_case,
                    config_path = self.config_path,
                    layer_count = layer_count
                )
                assert self.active_mask.shape == self.mean_G[..., 0].shape
        else:
            self.G00_f_mu = G00_f_mu
            self.G00_f_sigma = G00_f_sigma
            self.G01_f_mu = G01_f_mu
            self.G01_f_sigma = G01_f_sigma
            self.G10_f_mu = G10_f_mu
            self.G10_f_sigma = G10_f_sigma
            self.G11_f_mu = G11_f_mu
            self.G11_f_sigma = G11_f_sigma

            self.Fault = Fault(self.new_shape, 
                           G0_f_mu=self.G0_f_mu, G0_f_sigma = self.G0_f_sigma, G1_f_mu=self.G1_f_mu, G1_f_sigma = self.G1_f_sigma, G2_f_mu=self.G2_f_mu, G2_f_sigma = self.G2_f_sigma, G3_f_mu=self.G3_f_mu, G3_f_sigma = self.G3_f_sigma, dist = self.fault_dist, fault_rate = self.fault_rate, device=self.device, 
                            config_path=self.config_path, num_crossbars = self.num_crossbars)

    def forward(self, input):
        # x_relu() function is Critical
        input = input.to(self.device)

        # Split weights into positive/negative magnitudes
        positive_tensor = x_relu(input)
        negative_tensor = x_relu(-input)

        # 2-bit slicing: expected values in {0,1,2,3}, same shape as input
        input_scaled_pos = bitslicer(positive_tensor)
        input_scaled_neg = bitslicer(negative_tensor)

        if self.fault_model == "charge_trapping":
            # Per-cell shared Gaussian sample (correlates the 4 state conductances per cell)
            eps = torch.randn(self.new_shape, device=self.device, dtype=torch.float32)  # one eps per cell
            G_all = self.mean_G + eps.unsqueeze(-1) * self.sig_G  # (...,4)

            # Split into four tensors for your existing masking style
            G00 = G_all[..., 0].contiguous()
            G01 = G_all[..., 1].contiguous()
            G10 = G_all[..., 2].contiguous()
            G11 = G_all[..., 3].contiguous()

            # Map 2-bit states -> conductance (positive)
            self.G_pos[input_scaled_pos == 0] = G00[input_scaled_pos == 0]
            self.G_pos[input_scaled_pos == 1] = G01[input_scaled_pos == 1]
            self.G_pos[input_scaled_pos == 2] = G10[input_scaled_pos == 2]
            self.G_pos[input_scaled_pos == 3] = G11[input_scaled_pos == 3]

            # Map 2-bit states -> conductance (negative)
            self.G_neg[input_scaled_neg == 0] = G00[input_scaled_neg == 0]
            self.G_neg[input_scaled_neg == 1] = G01[input_scaled_neg == 1]
            self.G_neg[input_scaled_neg == 2] = G10[input_scaled_neg == 2]
            self.G_neg[input_scaled_neg == 3] = G11[input_scaled_neg == 3]

            # -------- NEW: split into active/inactive based on mask --------
            # active_mask must be broadcastable to input.shape
            # If missing, default: all active (so behavior matches old code)
            
            if self.active_mask is None:
                self.active_mask = torch.zeros_like(self.G_pos, dtype=torch.bool, device=self.device)
            else:
                self.active_mask = self.active_mask.to(self.device)
            if self.active_mask.dtype != torch.bool:
                    self.active_mask = self.active_mask.bool()

            # allow broadcast, but ensure it can broadcast to G_pos shape
                # (if you want strict checking, you can add asserts here)

            inactive_mask = ~self.active_mask

            # Active/inactive conductances for pos/neg
            G_pos_active = self.G_pos * self.active_mask
            G_pos_inactive = self.G_pos * inactive_mask
            G_neg_active = self.G_neg * self.active_mask
            G_neg_inactive = self.G_neg * inactive_mask

            # Return BOTH branches, each keeping the original (2, ...) format
            # so layer.py can keep "G[0] positive, G[1] negative" convention.
            G_active = torch.cat((G_pos_active.unsqueeze(0), G_neg_active.unsqueeze(0)), dim=0)
            G_inactive = torch.cat((G_pos_inactive.unsqueeze(0), G_neg_inactive.unsqueeze(0)), dim=0)
            return G_active, G_inactive
        else:
            mask_0_pos = input_scaled_pos == 0
            mask_0_neg = input_scaled_neg == 0
            mask_1_pos = input_scaled_pos == 1
            mask_1_neg = input_scaled_neg == 1
            mask_2_pos = input_scaled_pos == 2
            mask_2_neg = input_scaled_neg == 2
            mask_3_pos = input_scaled_pos == 3
            mask_3_neg = input_scaled_neg == 3
            #print(mask_0_pos.shape)
            #print(input_scaled_pos.shape)
            #exit()

            #self.G_pos = input_scaled_pos * self.delta_g
            #self.G_neg = input_scaled_neg * self.delta_g
            #inject process variations in all devices
            self.G0_dist = torch.normal(mean=self.G0_mu, std=self.G0_sigma, size=mask_0_pos.shape).to(self.device) #?
            self.G1_dist = torch.normal(mean=self.G1_mu, std=self.G1_sigma, size=mask_0_pos.shape).to(self.device)
            self.G2_dist = torch.normal(mean=self.G2_mu, std=self.G2_sigma, size=mask_0_pos.shape).to(self.device)
            self.G3_dist = torch.normal(mean=self.G3_mu, std=self.G3_sigma, size=mask_0_pos.shape).to(self.device)
            #self.G0_dist = torch.zeros(mask_0_pos.shape)
            #self.G0_dist = torch.ones(mask_0_pos.shape)
            
            self.G_pos = input_scaled_pos.clone()
            self.G_neg = input_scaled_neg.clone()
            self.G_pos[input_scaled_pos == 0] = mask_0_pos[input_scaled_pos == 0]*self.G0_dist[input_scaled_pos == 0]
            self.G_neg[input_scaled_neg == 0] = mask_0_neg[input_scaled_neg == 0]*self.G0_dist[input_scaled_neg == 0]
            self.G_pos[input_scaled_pos == 1] = mask_1_pos[input_scaled_pos == 1]*self.G1_dist[input_scaled_pos == 1]
            self.G_neg[input_scaled_neg == 1] = mask_1_neg[input_scaled_neg == 1]*self.G1_dist[input_scaled_neg == 1]
            self.G_pos[input_scaled_pos == 2] = mask_2_pos[input_scaled_pos == 2]*self.G2_dist[input_scaled_pos == 2]
            self.G_neg[input_scaled_neg == 2] = mask_2_neg[input_scaled_neg == 2]*self.G2_dist[input_scaled_neg == 2]
            self.G_pos[input_scaled_pos == 3] = mask_3_pos[input_scaled_pos == 3]*self.G3_dist[input_scaled_pos == 3]
            self.G_neg[input_scaled_neg == 3] = mask_3_neg[input_scaled_neg == 3]*self.G3_dist[input_scaled_neg == 3]
            
            if self.enable_fault:
                if self.layer_count == self.faulty_layer_idx:
                    output = torch.cat((self.Fault(self.G_pos, input_scaled_pos).unsqueeze(0),
                                    self.Fault(self.G_neg, input_scaled_neg).unsqueeze(0)),0)
                else:
                    output = torch.cat((self.G_pos.unsqueeze(0),
                                    self.G_neg.unsqueeze(0)), 0)
            else:
                output = torch.cat((self.G_pos.unsqueeze(0),
                                    self.G_neg.unsqueeze(0)), 0)      
            return output