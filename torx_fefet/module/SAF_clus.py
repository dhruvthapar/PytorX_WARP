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

class SAF(nn.Module):

    def __init__(self, G_shape, p_SA00=0.1, p_SA01=0.1, p_SA10=0.1, p_SA11=0.1, G_SA00=3e-3, G_SA01=1e-3, G_SA10=2e-3, G_SA11=3e-6, dist = "cluster"):
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
        # stuck at 00 leads to high conductance
        self.p_SA00 = nn.Parameter(torch.Tensor(
            [p_SA00]), requires_grad=False)  # probability of SA00
        self.G_SA00 = G_SA00
        # stuck at 01 leads to high conductance
        self.p_SA01 = nn.Parameter(torch.Tensor(
            [p_SA01]), requires_grad=False)  # probability of SA01
        self.G_SA01 = G_SA01
        # stuck at 10 leads to high conductance
        self.p_SA10 = nn.Parameter(torch.Tensor(
            [p_SA10]), requires_grad=False)  # probability of SA10
        self.G_SA10 = G_SA10
        # stuck at 11 leads to high conductance
        self.p_SA11 = nn.Parameter(torch.Tensor(
            [p_SA11]), requires_grad=False)  # probability of SA11
        self.G_SA11 = G_SA11
        self.dist = dist
        assert (
            self.p_SA00+self.p_SA11+self.p_SA01+self.p_SA10) <= 1, 'The sum of probability of SA00, SA01, SA10, SA11 is greater than 1 !!'

        # initialize a random mask
        # TODO: maybe change the SAF profile to uint8 format to avoid calculating the SAF defect
        # state on-the-fly, for simulation speedup. However the current setup has higher configurability
        # to simulate the real-time SAF state if there is run-time change .
        self.p_state = nn.Parameter(torch.Tensor(G_shape), requires_grad=False)
        # # print(self.p_state.shape)
        # self.p_SA00 = self.p_SA00.new_full(G_shape, p_SA00) # Allow tile-wise fault rate
        # self.p_SA11 = self.p_SA11.new_full(G_shape, p_SA11)
        # self.p_SA01 = self.p_SA01.new_full(G_shape, p_SA01) # Allow tile-wise fault rate
        # self.p_SA10 = self.p_SA10.new_full(G_shape, p_SA10)

        # self.fault_tensor = torch.zeros((self.p_state.shape))
        self.total_faults_per_crossbar = torch.zeros((self.p_state.shape[0],self.p_state.shape[1]))
        self.update_SAF_profile(self.dist)  # init the SAF distribution profile
        

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

    
    def dist_gen_uniform(self, fault_rate = 0.05):
        self.p_state.data.uniform_()
        fault_map = self.p_state.le(fault_rate) # initialize fault rate as tensor
        # true_indeces = torch.nonzero(fault_map)
        probabilities = torch.tensor([self.p_SA00[0],self.p_SA01[0], self.p_SA10[0], self.p_SA11[0]])
        count_true = torch.sum(fault_map).item()
        print(count_true)
        
        selected_values = (torch.multinomial(probabilities, count_true, replacement=True).squeeze() + 1).float()  # Convert to float        
        # Assign selected values to the corresponding True locations in p_state
        fault_state = torch.zeros_like(self.p_state)
        fault_state.masked_scatter_(fault_map, selected_values)
        self.p_state = fault_state 
       
        # self.p_state = torch.where(fault_map,torch.multinomial(probabilities,1).squeeze() + 1 , torch.zeros(1))
        # self.p_state.data.fill_(0)
        # for index in true_indeces:
        #     random_fault_type = torch.randint(len(values), size=(1,)).item()
        #     self.p_state[index] = values[random_fault_type]
        return self.p_state

    
    # Generate the cluster distribution (Distributtion based Cluster Algorithm)
    def dist_gen_cluster(self):
        # Randomly pick cluster centers
        # Generate coordinates for a center
        # Iterate all the element, increase the probability of sa0 or sa1
        # by increase/decrease the prob. distribution with threshold p_th
        # TODO: For now assume a 4-dimension weight matrix, make it more flexible

        crossbars_data = []
        for a in range(0, self.p_state.shape[0]):
            crossbars_data_row = []
            for b in range(0, self.p_state.shape[1]):
                #random picking of cluster centers
                centers_x = torch.randint(1, self.p_state.shape[2],size=(1,))
                centers_y = torch.randint(1, self.p_state.shape[3],size=(1,))
                # print("centers", centers_x, centers_y)
                variances = torch.randint(1, self.p_state.shape[3],size=(1,)) # spread along x and y- axis is same
                grid_x, grid_y = torch.meshgrid(torch.arange(0, self.p_state.shape[2]), torch.arange(0, self.p_state.shape[3]))
                mean = torch.tensor([centers_x, centers_y], dtype=torch.float32)
                covariance_matrix = torch.diag(torch.tensor([variances, variances], dtype=torch.float32))
                mvn = MultivariateNormal(mean, covariance_matrix)
                # print(torch.stack((grid_x, grid_y), dim=-1), torch.stack((grid_x, grid_y), dim=-1).shape)
                probabilities = mvn.log_prob(torch.stack((grid_x, grid_y), dim=-1)).exp()
                crossbars_data_row.append(probabilities)
                # Convert the list of data to a PyTorch tensor
            # crossbars_data.append(crossbars_data_row)
        
            self.p_state[a,:,:,:] = torch.stack(crossbars_data_row)

        #print(self.p_state, self.p_state.shape)

        
        #Identify the fault and distribute the faults(sa 00/01/10/11). Each fault type has equal prob of 0.25
        for a in range(0, self.p_state.shape[0]):
            for b in range(0, self.p_state.shape[1]):
                max_value = self.p_state[a,b,:,:].max().item()
                # print("max", max_value)
                Nr =  torch.rand(1).item() * 1.35 * max_value  #random value, if P(x,y) <
                # print("Nr",Nr)
                fault_map = Nr <= self.p_state[a,b,:,:]
                self.total_faults_per_crossbar[a,b] = torch.sum(fault_map).item()
                # Create a tensor of random values {1, 2, 3, 4} for the True values
                true_values = np.random.choice([1, 2, 3, 4], size=fault_map.shape, p=[0.25, 0.25, 0.25, 0.25])
                true_tensor = torch.tensor(true_values)
                #print(true_tensor)
                # Replace False values with zeros
                # false_tensor = torch.zeros_like(fault_map, dtype=torch.int)
                # Combine true and false tensors to get the final assigned tensor
                self.p_state[a,b,:,:] = torch.where(fault_map, true_tensor, 0)

        # self.p_state = self.fault_tensor
        # print(self.fault_tensor[0,0,65:75,115:120])
        print("total number of faults per layer:",self.total_faults_per_crossbar.sum().item())
        # quit(0)
        # print( self.total_faults_per_crossbar.shape)
        # print(self.total_faults_per_crossbar.sum().item())
        # print(self.total_faults_per_crossbar.sum().item() * 100/(self.p_state.shape[1]*self.p_state.shape[2]*self.p_state.shape[3]*self.p_state.shape[0]))
        
        return self.p_state, self.total_faults_per_crossbar

    def update_SAF_profile(self, dist='uniform'):
        if dist == 'uniform':
            self.p_state = self.dist_gen_uniform(fault_rate=0.005)  # update the SAF distribution.
        if dist == "cluster":
            self.p_state, self.total_faults_per_crossbar = self.dist_gen_cluster()
        return

    # def set_SAF_rate_tile(self, row_idx, col_idx, new_SA00_rate, new_SA01_rate, new_SA10_rate, new_SA11_rate):
    #     self.p_SA00[row_idx][col_idx].fill_(new_SA00_rate)
    #     self.p_SA01[row_idx][col_idx].fill_(new_SA01_rate)
    #     self.p_SA10[row_idx][col_idx].fill_(new_SA10_rate)
    #     self.p_SA11[row_idx][col_idx].fill_(new_SA11_rate)

    def dyn_injection(self, dyn_fault_rate=0.001, dyn_xb_rate=1, prob_faults = [0.1,0.2,0.3,0.4]):
        # Find the cells where there are no faults and mask the cells that are faulty
        # mask = self.p_state == 0
        # new_fault_data = torch.ones_like(self.p_state).float()

        # Generate uniform random values and apply the mask
        for j in range(self.p_state.shape[0]):  # row
            for k in range(self.p_state.shape[1]):  # column
                if torch.rand(1).item() < dyn_xb_rate:
                    # shape_tensor = torch.empty(self.p_state.size(2),self.p_state.size(3))
                    uniform_values = torch.rand(self.p_state.size(2),self.p_state.size(3))
                    mask_crxb = (self.p_state[j, k, :, :] == 0)
                    new_fault_data = torch.ones(self.p_state.size(2),self.p_state.size(3)).float()
                    uniform_values = uniform_values.to(self.p_state.device)
                    new_fault_data = new_fault_data.to(self.p_state.device)
                    new_fault_data[mask_crxb] = uniform_values[mask_crxb]

                    # Get the new fault map based on the dynamic_fault_rate
                    new_fault_map = new_fault_data.le(dyn_fault_rate).to(self.p_state.device)

                    probabilities = torch.tensor([prob_faults[0],prob_faults[1], prob_faults[2], prob_faults[3]]).to(self.p_state.device)
                    count_true = torch.sum(new_fault_map).item()
                    # print(count_true)
                    if count_true != 0:
                        selected_values = (torch.multinomial(probabilities, count_true, replacement=True).squeeze() + 1).float().to(self.p_state.device)
                        # Assign selected values to the corresponding True locations in p_state
                        fault_state = torch.zeros(self.p_state.size(2),self.p_state.size(3)).float().to(self.p_state.device)
                        fault_state.masked_scatter_(new_fault_map, selected_values).to(self.p_state.device)
                        # Distribute the fault equally among these new fault equally. Go over each XB.
                        self.p_state[j, k, :, :] = torch.where(new_fault_map,
                                                               fault_state,
                                                               self.p_state[j, k, :, :])
        return self.p_state


class _SAF(torch.autograd.Function):
    '''
    This autograd function performs the gradient mask for the weight
    element with Stuck-at-Fault defects, where those weights will not
    be updated during backprop through gradient masking.

    Args:
        input (Tensor): weight tensor in FP32
        p_state (Tensor): probability tensor for indicating the SAF state
        w.r.t the preset SA0/1 rate (i.e., p_SA00 and p_SA11).
        p_SA00 (FP): Stuck-at-Fault rate at 0 (range from 0 to 1).
        p_SA11 (FP): Stuck-at-Fault rate at 1 (range from 0 to 1).
        G_SA0 (FP): Stuck-at-Fault conductance at 0 (in unit of S).
        G_SA1 (FP): Stuck-at-Fault conductance at 1 (in unit of S).
    '''

    @staticmethod
    def forward(ctx, input, p_state, G_SA00, G_SA01, G_SA10, G_SA11):
        # p_state is the mask
        
        output = input.clone()
        output[p_state==1] = G_SA00
        output[p_state==2] = G_SA01
        output[p_state==3] = G_SA10
        output[p_state==4] = G_SA11
        ctx.save_for_backward(p_state)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        p_state = ctx.saved_tensors
        grad_input = grad_output.clone()
        # mask the gradient of defect cells

        grad_input[p_state==1] = 0
        grad_input[p_state==2] = 0
        grad_input[p_state==3] = 0
        grad_input[p_state==4] = 0

            #print("grad input from SAF:", grad_input, grad_input.size())
        return grad_input, None, None, None, None, None


Inject_SAF = _SAF.apply


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
