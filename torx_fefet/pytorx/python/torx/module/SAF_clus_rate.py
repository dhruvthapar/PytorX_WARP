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

def Inject_SAF(input, p_state, G_SA00, G_SA01, G_SA10, G_SA11):
    output = input.clone()
    output[p_state == 1] = G_SA00
    output[p_state == 2] = G_SA01
    output[p_state == 3] = G_SA10
    output[p_state == 4] = G_SA11
    return output

class SAF(nn.Module):

    def __init__(self, G_shape, p_SA00=0.1, p_SA01=0.1, p_SA10=0.1, p_SA11=0.1, G_SA00=3e-3, G_SA01=1e-3, G_SA10=2e-3, G_SA11=3e-6, dist = "cluster", fault_rate = 0.05, device=None):
        super(SAF, self).__init__()
        '''
        This module performs the Stuck-At-Fault (SAF) non-ideal effect injection.
            Args:
                G_shape (tensor.size): crossbar array size.
                p_SA00 (FP): Stuck-at-Fault rate at 00 (range from 0 to 1).
                p_SA11 (FP): Stuck-at-Fault rate at 11 (range from 0 to 1).
                p_SA01 (FP): Stuck-at-Fault rate at 01 (range from 0 to 1).
                p_SA10 (FP): Stuck-at-Fault rate at 10 (range from 0 to 1).
                G_SA00 (FP): Stuck-at-Fault conductance at 00 (in unit of S).
                G_SA01 (FP): Stuck-at-Fault conductance at 01 (in unit of S).
                G_SA10 (FP): Stuck-at-Fault conductance at 10 (in unit of S).
                G_SA11 (FP): Stuck-at-Fault conductance at 11 (in unit of S).
        '''
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # stuck at 00 leads to high conductance
       # Initialize tensors on the correct device
        self.p_SA00 = torch.tensor([p_SA00], device=self.device)
        self.p_SA01 = torch.tensor([p_SA01], device=self.device)
        self.p_SA10 = torch.tensor([p_SA10], device=self.device)
        self.p_SA11 = torch.tensor([p_SA11], device=self.device)
        
        self.G_SA00 = G_SA00
        self.G_SA01 = G_SA01
        self.G_SA10 = G_SA10
        self.G_SA11 = G_SA11

        self.dist = dist
        self.fault_rate = fault_rate
        self.G_shape = G_shape
        self.probabilities = torch.tensor([self.p_SA00[0],self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]], device=self.device)
        self.p_state = torch.Tensor(self.G_shape).to(self.device)
        # self.p_state = torch.empty(G_shape, device=self.device)  # Initialize p_state on the correct device
        self.update_SAF_profile(dist)  # Initialize SAF profile
        

        assert (
            self.p_SA00+self.p_SA11+self.p_SA01+self.p_SA10) <= 1, 'The sum of probability of SA00, SA01, SA10, SA11 is greater than 1 !!'
        

    def forward(self, input):
        '''
        The forward function alter the elements that indexed by p_state to the defected conductance,
        and mask the gradient of those defect cells owing to the auto-differentiation.
        '''
        output = Inject_SAF(input, self.p_state, self.G_SA00, self.G_SA01, self.G_SA10, self.G_SA11)
        return output

    def index_SA00(self):

        return self.p_state.eq(1)


    def index_SA01(self):

        return self.p_state.eq(2)


    def index_SA10(self):

        return self.p_state.eq(3)


    def index_SA11(self):

        return self.p_state.eq(4)

    # def dist_gen_uniform(self):
    #     final_fault_state = torch.zeros(self.G_shape, device=self.device)
    #     a, b, c, d = self.G_shape

    #     for i in range(a):
    #         for j in range(b):
    #             # Generate a fault map based on the fault rate
    #             fault_map = torch.rand((c, d), device=self.device) < self.fault_rate
    #             total_faults = fault_map.sum().item()
    #             print("total_faults", total_faults)

    #             if total_faults > 0:
    #                 # Distribute fault types according to the specified probabilities
    #                 fault_types = torch.zeros((c, d), device=self.device)
                    
    #                 # Generate fault type indices based on probabilities
    #                 fault_indices = torch.multinomial(self.probabilities, total_faults, replacement=True) + 1
                    
    #                 # Assign fault types to their locations
    #                 fault_types[fault_map] = fault_indices.float()
                    
    #                 # Update the final fault state with the generated fault types for this slice
    #                 final_fault_state[i, j] = fault_types
    #     print("final_fault_state", final_fault_state)
    #     # quit()

    #     return final_fault_state
    def dist_gen_uniform(self):
            final_fault_state = torch.zeros(self.G_shape, device=self.device)
            a, b, c, d = self.G_shape

            fault_map = torch.rand((a,b,c, d), device=self.device) < self.fault_rate
            total_faults = fault_map.sum().item()
            # print("total_faults", total_faults)

            if total_faults > 0:
                # Distribute fault types according to the specified probabilities
                fault_types = torch.zeros((a,b,c, d), device=self.device)
                
                # Generate fault type indices based on probabilities
                fault_indices = torch.multinomial(self.probabilities, total_faults, replacement=True) + 1
                
                # Assign fault types to their locations
                fault_types[fault_map] = fault_indices.float()
                
                # Update the final fault state with the generated fault types for this slice
                final_fault_state = fault_types
            # print("final_fault_state", final_fault_state)
            # quit()

            return final_fault_state

    
    # Generate the cluster distribution (Distributtion based Cluster Algorithm)
    def dist_gen_cluster(self):
        final_fault_state = torch.zeros(self.G_shape, device=self.device)
        
        max_points = int(self.fault_rate * self.p_state.numel())
        
        # Adjusting dimensions for meshgrid generation
        N, C, H, W = self.p_state.shape
        flat_dim_x, flat_dim_y = N * H, C * W
        

        # Correctly redefining the setup for Gaussian Mixture Model clustering
        c, d = flat_dim_x, flat_dim_y  # Grid dimensions for a manageable demonstration
        z = self.fault_rate * 100 # Fault percentage

        # Generate grid points again
        x, y = np.arange(c), np.arange(d)
        xx, yy = np.meshgrid(x, y)
        points = np.c_[xx.ravel(), yy.ravel()]

        # Select a random center from the points
        # np.random.seed(0)  # For reproducibility
        random_center_index = np.random.choice(len(points))
        random_center = points[random_center_index]

        # Initialize the Gaussian Mixture Model with the corrected means_init parameter
        gmm = GaussianMixture(n_components=1, means_init=[random_center]).fit(points)

        # Calculate the probabilities for each point
        probabilities = gmm.score_samples(points)

        # Find the threshold to include the desired fault percentage of points
        threshold = np.percentile(probabilities, 100 - z)

        # Determine clustered points based on the threshold
        is_clustered = np.where(probabilities >= threshold, 1, 0)

        # Reshape the clustered points indicator to match the grid's dimensions
        clustered_indicator_matrix = is_clustered.reshape(c, d)

        # Convert the clustered indicator matrix to a PyTorch tensor and reshape it back to the original shape
        clustered_indicator_tensor = torch.tensor(clustered_indicator_matrix, device=self.device)
        intermediate_tensor = clustered_indicator_tensor.view(N, H, C, W)
        # print(clustered_indicator_tensor)
        clustered_tensor = intermediate_tensor.permute(0, 2, 1, 3)
        
        # Distributing the faults based on the mask
        fault_types = torch.tensor([1, 2, 3, 4], device=self.device)
        probabilities = torch.tensor([self.p_SA00[0], self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]], device=self.device)
        
        for a in range(N):
            for b in range(C):
                true_indices = clustered_tensor[a, b].nonzero(as_tuple=True)
                num_true = true_indices[0].size(0)
                if num_true > 0:
                    faults = torch.multinomial(probabilities, num_true, replacement=True)
                    final_fault_state[a, b][true_indices] = fault_types[faults].to(final_fault_state.dtype)
                
        return final_fault_state

    def update_SAF_profile(self, dist="uniform"):
        # Update this method to handle device placement correctly
        # Ensure that any tensor manipulation here uses self.device
        # For example, when generating random tensors or performing operations
        if dist == "uniform":
            self.p_state.data = self.dist_gen_uniform().to(self.device)
        elif dist == "cluster":
            self.p_state.data = self.dist_gen_cluster().to(self.device)

    # def set_SAF_rate_tile(self, row_idx, col_idx, new_SA00_rate, new_SA01_rate, new_SA10_rate, new_SA11_rate):
    #     self.p_SA00[row_idx][col_idx].fill_(new_SA00_rate)
    #     self.p_SA01[row_idx][col_idx].fill_(new_SA01_rate)
    #     self.p_SA10[row_idx][col_idx].fill_(new_SA10_rate)
    #     self.p_SA11[row_idx][col_idx].fill_(new_SA11_rate)




# class _SAF(torch.autograd.Function):
#     '''
#     This autograd function performs the gradient mask for the weight
#     element with Stuck-at-Fault defects, where those weights will not
#     be updated during backprop through gradient masking.

#     Args:
#         input (Tensor): weight tensor in FP32
#         p_state (Tensor): probability tensor for indicating the SAF state
#         w.r.t the preset SA0/1 rate (i.e., p_SA00 and p_SA11).
#         p_SA00 (FP): Stuck-at-Fault rate at 0 (range from 0 to 1).
#         p_SA11 (FP): Stuck-at-Fault rate at 1 (range from 0 to 1).
#         G_SA0 (FP): Stuck-at-Fault conductance at 0 (in unit of S).
#         G_SA1 (FP): Stuck-at-Fault conductance at 1 (in unit of S).
#     '''

#     @staticmethod
#     def forward(ctx, input, p_state, G_SA00, G_SA01, G_SA10, G_SA11):
#         # p_state is the mask
        
#         output = input.clone()
#         output[p_state==1] = G_SA00
#         output[p_state==2] = G_SA01
#         output[p_state==3] = G_SA10
#         output[p_state==4] = G_SA11
#         ctx.save_for_backward(p_state)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         p_state = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         # mask the gradient of defect cells

#         grad_input[p_state==1] = 0
#         grad_input[p_state==2] = 0
#         grad_input[p_state==3] = 0
#         grad_input[p_state==4] = 0

#         #print("grad input from SAF:", grad_input, grad_input.size())
#         return grad_input, None, None, None, None, None


# Inject_SAF = _SAF.apply


############################################################
# Testbenchs
############################################################

# pytest
def test_SAF_update_profile():
    G_shape = torch.Size([1, 3, 3, 3])
    saf_module = SAF(G_shape)
    pre_index_SA00 = saf_module.index_SA00()
    pre_index_SA01 = saf_module.index_SA01()
    pre_index_SA10 = saf_module.index_SA10()
    pre_index_SA11 = saf_module.index_SA11()
    print("SAF AT 00", pre_index_SA00)
    print("SAF AT 01", pre_index_SA01)
    print("SAF AT 10", pre_index_SA10)
    print("SAF AT 11", pre_index_SA11)
    #saf_module.update_SAF_profile()
    #post_index_SA0 = saf_module.index_SA00()
    # print((pre_index_SA0-post_index_SA0).sum())
    #assert (pre_index_SA0 -
    #        post_index_SA0).sum().item() != 0, 'SAF profile is not updated!'
    # print(saf_module.index_SA0())
    return


def test_SA0_SA1_overlap():
    '''
    ensure there is no SAF state overlap between SA0 and SA1
    '''
    G_shape = torch.Size([3, 1, 3, 3])
    saf_module = SAF(G_shape)
    index_SA0 = saf_module.index_SA0()
    index_SA1 = saf_module.index_SA1()
    assert (index_SA0 * index_SA1).sum().item() == 0, 'exist element is 1 for both SA0/1 index!'
    return


#if __name__ == '__main__':
#     test_SAF_update_profile()
#     test_SA0_SA1_overlap()
