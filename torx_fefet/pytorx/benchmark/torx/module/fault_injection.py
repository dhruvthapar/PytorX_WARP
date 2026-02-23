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
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from sklearn.mixture import GaussianMixture
from .exps_script import *
import pickle
import time


def inject_fault(input, G0_mu, G0_sigma, G1_mu, G1_sigma, G2_mu, G2_sigma, G3_mu, G3_sigma, input_scaled, fault_map): #this just assigns the appropriate fault conductance
    output = input.clone()
    G0_f_dist = torch.normal(mean=G0_mu, std=G0_sigma, size=input.shape, device=input.device)
    G1_f_dist = torch.normal(mean=G1_mu, std=G1_sigma, size=input.shape, device=input.device)
    G2_f_dist = torch.normal(mean=G2_mu, std=G2_sigma, size=input.shape, device=input.device)
    G3_f_dist = torch.normal(mean=G3_mu, std=G3_sigma, size=input.shape, device=input.device)
    output[(fault_map == 1) & (input_scaled == 0)] = G0_f_dist[(fault_map == 1) & (input_scaled == 0)]
    output[(fault_map == 1) & (input_scaled == 1)] = G1_f_dist[(fault_map == 1) & (input_scaled == 1)]
    output[(fault_map == 1) & (input_scaled == 2)] = G2_f_dist[(fault_map == 1) & (input_scaled == 2)]
    output[(fault_map == 1) & (input_scaled == 3)] = G3_f_dist[(fault_map == 1) & (input_scaled == 3)]
    return output

class Fault(nn.Module):
    def __init__(self, G_shape, G0_f_mu=3e-3, G0_f_sigma=1e-3, G1_f_mu=2e-3, G1_f_sigma=3e-6, G2_f_mu = 3e-3, G2_f_sigma = 1e-3, G3_f_mu = 2e-3, G3_f_sigma = 3e-6, dist = "cluster", fault_rate = 0.05, device=None, config_path=None, num_crossbars=None):
        super(Fault, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        #* IF layer count == fault list of [layercount] [[f1], [f2], [f3], [f4]]
        self.G0_f_mu = G0_f_mu
        self.G0_f_sigma = G0_f_sigma
        self.G1_f_mu = G1_f_mu
        self.G1_f_sigma = G1_f_sigma
        self.G2_f_mu = G2_f_mu
        self.G2_f_sigma = G2_f_sigma
        self.G3_f_mu = G3_f_mu
        self.G3_f_sigma = G3_f_sigma     

        self.dist = dist
        self.config_path = config_path
        self.fault_rate = fault_rate
        self.num_crossbars = num_crossbars
        self.G_shape = G_shape
        self.fault_map = torch.Tensor(self.G_shape).to(self.device)
        self.update_fault_profile(self.dist) 

    def forward(self, input, input_scaled):
        '''
        The forward function alter the elements that indexed by p_state to the defected conductance,
        and mask the gradient of those defect cells owing to the auto-differentiation.
        '''
        torch.manual_seed(42)
        output = inject_fault(input, self.G0_mu_f, self.G0_sigma_f, self.G1_mu_f, self.G1_sigma_f, self.G2_mu_f, self.G2_sigma_f, self.G3_mu_f, self.G3_sigma_f, input_scaled, self.fault_map)
        return output

    def dist_gen_uniform_faults(self):
        a, b, c, d = self.G_shape
        fault_map = torch.rand((a,b,c, d), device=self.device) < self.fault_rate
        #read the weight values at the locations where fault map injected the faults (is True)
        total_faults = fault_map.sum().item() #total_no_of_faults injected        
        return fault_map 

    # Generate the cluster distribution (Distributtion based Cluster Algorithm)
    def dist_gen_cluster(self):
        N, C, H, W = self.G_shape
        reshaped_dim = (N*H, C*W)
        # Randomly select a variance within the given range
        variance_range = (1, H)
        variance = np.random.uniform(*variance_range)
        # Generate a random center within the grid
        np.random.seed(int(time.time()))
        random_center = (np.random.randint(0, reshaped_dim[0]), np.random.randint(0, reshaped_dim[1]))
        # Generate grid points
        x, y = np.meshgrid(np.arange(reshaped_dim[0]), np.arange(reshaped_dim[1]), indexing='ij')
        
        # Calculate the Gaussian PDF for each point
        pdf = np.exp(-(((x - random_center[0])**2 + (y - random_center[1])**2) / (2 * variance**2)))
        pdf /= pdf.sum()  # Normalize the PDF
        
        # Determine fault states based on the fault rate
        fault_rate = min(self.fault_rate*self.num_crossbars/(N*C), 1)
        print(f'fault_rate = {fault_rate}')
        threshold = np.percentile(pdf, 100 - (fault_rate * 100))
        fault_state = pdf >= threshold
        fault_state_reshaped = fault_state.reshape(N, H, C, W).transpose(0, 2, 1, 3)
        fault_map = torch.tensor(fault_state_reshaped.astype(int), device=self.device)
        return fault_map
    

    def update_fault_profile(self, dist="uniform"):
        # Update this method to handle device placement correctly
        # Ensure that any tensor manipulation here uses self.device
        # For example, when generating random tensors or performing operations
        if dist == "uniform":
            self.fault_map = self.dist_gen_uniform_faults().to(self.device)
        elif dist == "cluster":
            self.fault_map = self.dist_gen_cluster().to(self.device)
