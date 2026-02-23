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

def Inject_SAF(input, p_state, G_SA00, G_SA01, G_SA10, G_SA11): #this just assigns the appropriate fault conductance
    output = input.clone()
    output[p_state == 1] = G_SA00
    output[p_state == 2] = G_SA01
    output[p_state == 3] = G_SA10
    output[p_state == 4] = G_SA11
    return output

class SAF(nn.Module):

    def __init__(self, G_shape, p_SA00=0.1, p_SA01=0.1, p_SA10=0.1, p_SA11=0.1, G_SA00=3e-3, G_SA01=1e-3, G_SA10=2e-3, G_SA11=3e-6, dist = "cluster", fault_rate = 0.05, device=None, config_path=None, layer_count=None, weight_mask=None):
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
        
        #* IF layer count == fault list of [layercount] [[f1], [f2], [f3], [f4]]
        
        self.G_SA00 = G_SA00
        self.G_SA01 = G_SA01
        self.G_SA10 = G_SA10
        self.G_SA11 = G_SA11

        self.dist = dist
        self.config_path = config_path
        self.weight_mask = weight_mask
        self.layer_count = layer_count
        self.fault_rate = fault_rate
        self.G_shape = G_shape
        #print(layer_count, G_shape)
        self.probabilities = torch.tensor([self.p_SA00[0],self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]], device=self.device)
        self.p_state = torch.Tensor(self.G_shape).to(self.device)
        # self.p_state = torch.empty(G_shape, device=self.device)  # Initialize p_state on the correct device
        # self.update_SAF_profile(dist)  # Initialize SAF profile
        
        self.update_SAF_profile(self.dist) 
        

        assert (
            self.p_SA00+self.p_SA11+self.p_SA01+self.p_SA10) <= 1, 'The sum of probability of SA00, SA01, SA10, SA11 is greater than 1 !!'
        

    def forward(self, input, mask):
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
            #read the weight values at the locations where fault map injected the faults (is True)
            
            total_faults = fault_map.sum().item() #total_no_of_faults injected
            # print("total_faults", total_faults)

            if total_faults > 0:
                # Distribute fault types according to the specified probabilities
                #fault_types = torch.zeros((a,b,c, d), device=self.device) #commented_dhruv
                
                # Generate fault type indices based on probabilities
                fault_indices = torch.multinomial(self.probabilities, total_faults, replacement=True) + 1 
                #fault_indices are the fault_type = 1,2,3, or 4 dependeding on which stuck-at
                
                # Assign fault types to their locations
                final_fault_state[fault_map] = fault_indices.float() 
                
                # Update the final fault state with the generated fault types for this slice
                #final_fault_state = fault_types #commented_dhruv
            # print("final_fault_state", final_fault_state)
            # quit()

            return final_fault_state

    def dist_gen_worst_case(self, weight_mask, config_path, layer_count):
            masks_list = torch.load(self.config_path+'/masks_list.pth')
            weight_mask = masks_list[self.layer_count].to(self.device)
            final_fault_state = torch.zeros(self.G_shape, device=self.device)
            # with open(config_path, 'r') as config_file:
            #     config = json.load(config_file)

            fault_map = torch.load(self.config_path+'/fault_map.pth')

            # print(fault_map)
            # print("layer_count", layer_count)   
            fault_map_layer = fault_map[layer_count].to(self.device)
            fault_map_layer = fault_map_layer.permute(1, 0, 3, 2)
            # print("fault_map_layer", fault_map_layer.shape)
            ##apply mask to fault_map_layer
            # print("input mask shape", mask.shape)
            masked_fault_map = fault_map_layer.to(self.device) * weight_mask.int()
            
            # print("masked_fault_map", masked_fault_map.shape)
            # p = max(0, min(p, 100))
            # fault_percentage = self.fault_rate * 100
            # Calculate the number of elements to select based on percentage p
            num_elements_to_select = int((masked_fault_map.size(2) * masked_fault_map.size(3) * self.fault_rate) )
            
            # Initialize the mask with zeros
            mask = torch.zeros_like(masked_fault_map, dtype=torch.float)
            # print("mask shape", mask.shape)
            # print('fault_state', final_fault_state.shape)
            for a in range(masked_fault_map.size(0)):
                for b in range(masked_fault_map.size(1)):
                    # Flatten the (c, d) dimensions
                    slice_flat = masked_fault_map[a, b].flatten().to(torch.int16)
                    
                    # Sort elements in descending order and get the indices
                    _, indices = torch.sort(slice_flat, descending=True)
                    
                    # Select indices corresponding to the top p%
                    selected_indices = indices[:num_elements_to_select]
                    
                    # Create a mask for the selected elements
                    slice_mask = torch.zeros_like(slice_flat, dtype=torch.float)
                    slice_mask[selected_indices] = 1
                    
                    # Reshape the mask back to the (c, d) dimensions and assign to the mask tensor
                    mask[a, b] = slice_mask.reshape(masked_fault_map.size(2), masked_fault_map.size(3))

            # Distributing the faults based on the mask
            fault_types = torch.tensor([1, 2, 3, 4], device=self.device)
            probabilities = torch.tensor([self.p_SA00[0], self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]], device=self.device)
            
            for a in range(mask.size(0)):
                for b in range(mask.size(1)):
                    # print("mask", mask[a, b])
                    true_indices = mask[a, b].nonzero(as_tuple=True)
                    num_true = true_indices[0].size(0)
                    # print("num_true", true_indices)
                    # print("shape", mask[a, b].shape)
                    # print("shape", final_fault_state[a, b].shape)
                    # Example of advanced indexing if final_fault_state[a, b] and mask[a, b] are 2D
                    if num_true > 0:
                        faults = torch.multinomial(probabilities, num_true, replacement=True)
                        fault_values = fault_types[faults]
                        # Assume true_indices is a tuple of two tensors (for 2D indexing)
                        final_fault_state[a, b][true_indices[0], true_indices[1]] = fault_values.to(final_fault_state.dtype)


            return final_fault_state    

    # Generate the cluster distribution (Distributtion based Cluster Algorithm)
    def dist_gen_cluster(self):
        final_fault_state = torch.zeros(self.G_shape, device=self.device)

        N, C, H, W = self.G_shape
        reshaped_dim = (N*H, C*W)
        
        # Randomly select a variance within the given range
        variance_range = (1, H)
        variance = np.random.uniform(*variance_range)
        
        # Generate a random center within the grid
        random_center = (np.random.randint(0, reshaped_dim[0]), np.random.randint(0, reshaped_dim[1]))
        
        # Generate grid points
        x, y = np.meshgrid(np.arange(reshaped_dim[0]), np.arange(reshaped_dim[1]), indexing='ij')
        
        # Calculate the Gaussian PDF for each point
        pdf = np.exp(-(((x - random_center[0])**2 + (y - random_center[1])**2) / (2 * variance**2)))
        pdf /= pdf.sum()  # Normalize the PDF
        
        # Determine fault states based on the fault rate
        threshold = np.percentile(pdf, 100 - (self.fault_rate * 100))
        fault_state = pdf >= threshold
        # print(fault_state.astype(int))
        # Reshape the fault state back to the original tensor shape
        fault_state_reshaped = fault_state.reshape(N, H, C, W).transpose(0, 2, 1, 3)
        # print(fault_state_reshaped.astype(int))
        final_fault_state = torch.tensor(fault_state_reshaped.astype(int), device=self.device)
        
        # Distributing the faults based on the mask
        fault_types = torch.tensor([1, 2, 3, 4], device=self.device)
        probabilities = torch.tensor([self.p_SA00[0], self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]], device=self.device)
        
        for a in range(N):
            for b in range(C):
                true_indices = final_fault_state[a, b].nonzero(as_tuple=True)
                num_true = true_indices[0].size(0)
                if num_true > 0:
                    faults = torch.multinomial(probabilities, num_true, replacement=True)
                    final_fault_state[a, b][true_indices] = fault_types[faults].to(final_fault_state.dtype)
                
        return final_fault_state

    def update_SAF_profile(self, dist="uniform", mask=None):
        # Update this method to handle device placement correctly
        # Ensure that any tensor manipulation here uses self.device
        # For example, when generating random tensors or performing operations
        if dist == "uniform":
            self.p_state.data = self.dist_gen_uniform().to(self.device)
        elif dist == "cluster":
            self.p_state.data = self.dist_gen_cluster().to(self.device)
        elif dist == "worst_case":
            self.p_state.data = self.dist_gen_worst_case(self.weight_mask, self.config_path, self.layer_count).to(self.device)

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
