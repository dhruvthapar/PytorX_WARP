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

# from .SAF import SAF
# from .SAF_clus import SAF
from .SAF_clus_rate import SAF

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

    def __init__(self, delta_g, Gmin, G_SA00, G_SA01, G_SA10, G_SA11, weight_shape, p_SA00, p_SA01, p_SA10, p_SA11,  
                 enable_rand=True, enable_SAF=False, fault_rate = 0.5, fault_dist = "cluster", msb_only_ec = True,  device=None):
        super(w2g, self).__init__()
        self.device =  device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.delta_g = delta_g
        self.Gmin = Gmin
        self.G_SA00 = G_SA00
        self.G_SA01 = G_SA01
        self.G_SA10 = G_SA10
        self.G_SA11 = G_SA11
        self.p_SA00 = p_SA00
        self.p_SA01 = p_SA01
        self.p_SA10 = p_SA10
        self.p_SA11 = p_SA11
        self.enable_rand = enable_rand
        self.enable_SAF = enable_SAF
        self.device =device
        self.fault_dist = fault_dist
        self.fault_rate = fault_rate
        self.msb_only_ec = msb_only_ec
        shape_list = list(weight_shape)  # Convert the tuple to a list
        shape_list[2] = weight_shape[2] * 8 # Modify the desired element at index 2
        new_shape = torch.Size(shape_list)  # Convert the list back to a tuple
        self.SAF_pos = SAF(new_shape, p_SA00=self.p_SA00, p_SA01=self.p_SA01, p_SA10=self.p_SA10, p_SA11=self.p_SA11,
                            G_SA00=self.G_SA00, G_SA01=self.G_SA01, G_SA10=self.G_SA10, G_SA11=self.G_SA11, dist = self.fault_dist, fault_rate = self.fault_rate, device=self.device)
        self.SAF_neg = SAF(new_shape, p_SA00=self.p_SA00, p_SA01=self.p_SA01, p_SA10=self.p_SA10, p_SA11=self.p_SA11,
                            G_SA00=self.G_SA00, G_SA01=self.G_SA01, G_SA10=self.G_SA10, G_SA11=self.G_SA11, dist = self.fault_dist, fault_rate = self.fault_rate, device=self.device)
    
    def forward(self, input):
        # x_relu() function is Critical
        input = input.to(self.device)
        
        # keep positive values and set negative values as zero
        #print(input[:,:,0:2,:], input.size())
        positive_tensor = x_relu(input)
        #print("positive weights",positive_tensor[:,:,0,:], positive_tensor.size())
        # Set positive values as zero and store negative values in their absolute form
        negative_tensor = x_relu(-input)

        # Convert the weights to such that 2-bits are mapped to thae array
        #input_scaled_pos = bitslicer(positive_tensor,saf_enable, pos_SA00, pos_SA01, pos_SA10, pos_SA11)
        input_scaled_pos = bitslicer(positive_tensor)
        #print("scaled weights",input_scaled_pos)
        #input_scaled_neg = bitslicer(negative_tensor,saf_enable, neg_SA00, neg_SA01, neg_SA10, neg_SA11)
        input_scaled_neg = bitslicer(negative_tensor)
        #print("scaled weights",input_scaled_neg)
        
        #diff = (input_scaled_pos - input_scaled_neg)
        #print("diff", diff[:,:,0,:].to(torch.float64))

        self.G_pos = self.G_SA00 + input_scaled_pos * self.delta_g
        #postive_crxb = self.G_pos.view().transpose(2,3)
        #print(self.G_pos, type(self.G_pos), self.G_pos.size())
        self.G_neg = self.G_SA00 + input_scaled_neg * self.delta_g
        #print("diff crxb",self.G_pos)
        #quit(0)
        # the following two steps will update the SAF masking if enable_rand is True
        if self.enable_SAF:
            output = torch.cat((self.SAF_pos(self.G_pos).unsqueeze(0),
                                self.SAF_neg(self.G_neg).unsqueeze(0)),0)
        else:
            output = torch.cat((self.G_pos.unsqueeze(0),
                                self.G_neg.unsqueeze(0)), 0)

        return output
    
    
    def error_compensation(self):
        
        if self.msb_only_ec :
            # create a mask for lsb bits in each crossbar array. 
            # The mask is used to set the lsb bits to zero.
            dims = list(self.G_pos.shape)
            msb_mask = torch.ones_like(self.G_pos, dtype=torch.bool).to(self.device)
            range_list = list(range(0, dims[2]//8)) # divide by no of cells per weight bit (8)
            for a in range(0, dims[0]) :
                for b in range(0, dims[1]):
                        for i in range(0, len(range_list)):
                            msb_mask[a,b, i*8+5:8*i+8] = False
            
            pos_SA00 = (self.SAF_pos.index_SA00() & msb_mask).float().to(self.device)      
            pos_SA01 = (self.SAF_pos.index_SA01() & msb_mask).float().to(self.device)
            pos_SA10 = (self.SAF_pos.index_SA10() & msb_mask).float().to(self.device)
            pos_SA11 = (self.SAF_pos.index_SA11() & msb_mask).float().to(self.device)
            
            neg_SA00 = (self.SAF_neg.index_SA00() & msb_mask).float().to(self.device)
            neg_SA01 = (self.SAF_neg.index_SA01() & msb_mask).float().to(self.device)
            neg_SA10 = (self.SAF_neg.index_SA10() & msb_mask).float().to(self.device)
            neg_SA11 = (self.SAF_neg.index_SA11() & msb_mask).float().to(self.device)
        else :
            pos_SA00 = self.SAF_pos.index_SA00().float().to(self.device)      
            pos_SA01 = self.SAF_pos.index_SA01().float().to(self.device)
            pos_SA10 = self.SAF_pos.index_SA10().float().to(self.device)
            pos_SA11 = self.SAF_pos.index_SA11().float().to(self.device)
            
            neg_SA00 = self.SAF_neg.index_SA00().float().to(self.device)
            neg_SA01 = self.SAF_neg.index_SA01().float().to(self.device)
            neg_SA10 = self.SAF_neg.index_SA10().float().to(self.device)
            neg_SA11 = self.SAF_neg.index_SA11().float().to(self.device)

        G_pos_diff = ((self.G_pos-self.G_SA00)*pos_SA00 + \
        (self.G_pos-self.G_SA01)*pos_SA01 + \
        (self.G_pos-self.G_SA10)*pos_SA10 + \
        (self.G_pos-self.G_SA11)*pos_SA11)
        #print("input", self.G_pos)
        #print("pos", G_pos_diff[0,0,0,:]/self.delta_g,  torch.count_nonzero(G_pos_diff == 0))
        #quit(0)
        
        G_neg_diff = (self.G_neg-self.G_SA00)*neg_SA00 + \
        (self.G_neg-self.G_SA01)*neg_SA01 + \
        (self.G_neg-self.G_SA10)*neg_SA10 + \
        (self.G_neg-self.G_SA11)*neg_SA11
        #print("neg", G_neg_diff[0,0,0,:]/self.delta_g)
        #print("diff", (G_pos_diff - G_neg_diff)/self.delta_g)
        return G_pos_diff, G_neg_diff

    


# class _newrelu(torch.autograd.Function):
#     '''
#     This self-define function is used for mapping weight on positive 
#     and negative array. It will prevent close to zero weights trapped 
#     within the region that quantized into zero, which will never be 
#     updated by back-propagation, thus degrades the accuracy. 
#     ''' 
#     @staticmethod
#     def forward(ctx, input):
#         ctx.save_for_backward(input)
#         return input.clamp(min=0)
    
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, = ctx.saved_tensors
#         #print("backward_for xrelu", grad_output, grad_output.size())
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         #print("backward_from xrelu", grad_input, grad_input.size())
#         return grad_input
    
# x_relu = _newrelu.apply

# class _bitslicer(torch.autograd.Function):

    
#     @staticmethod
#     def forward(ctx, input):
#         size = input.size()
#         ctx.size = size
#         # Convert tensor to int16 data type
#         tensor_int16 = input.to(torch.int16)
#         # Split each element into 8 segments of 2 bits each
#         segments = []
#         for index in range(7, -1, -1):
#             # Shift the tensor by 2 bits for each segment index
#             shifted_tensor = tensor_int16 >> (2 * index)
#             # Apply bitwise AND operation with 0b11 to extract the 2-bit segment
#             segment = shifted_tensor & 0b11
#             segments.append(segment)
#         # Create a tensor by stacking the segments along a new dimension
#         segments_tensor = torch.stack(segments, dim=-1).float()
#         # Reshape the segments tensor to split along the third dimension
#         reshaped_tensor = segments_tensor.transpose(3, 4).reshape(size[0], size[1], -1, size[3])
#         return reshaped_tensor

#     @staticmethod
#     def backward(ctx, grad_output):
#         #print("backward_for bitslice", grad_output,grad_output.size())
#         grad_input = torch.zeros(ctx.size).cuda()
#         temp =  grad_output.clone().cuda()
#         verticalsize = ctx.size[3]
#         chunksize = int(verticalsize / 8)
#         # Reshape the tensor to split dimension 3 into groups of 8 elements
#         for i in range(chunksize):
#                 grad_input[:,:,i,:] = temp[:,:,8*i+7,:] 
            
#         #print("backward_from bitslice",grad_input, grad_input.size())
#         return grad_input,None, None, None, None, None

# bitslicer = _bitslicer.apply
############################################################
# Testbenchs
############################################################
# class _bitslicer(torch.autograd.Function):

    
#     @staticmethod
#     def forward(ctx, input, saf_enable, pos_SA00, pos_SA01, pos_SA10, pos_SA11):
#         size = input.size()
#         ctx.size = size
#         ctx.saf_enable = saf_enable
#         ctx.save_for_backward(pos_SA00, pos_SA01, pos_SA10, pos_SA11)
#         # Convert tensor to int16 data type
#         tensor_int16 = input.to(torch.int16).cuda()
#         # Split each element into 8 segments of 2 bits each
#         segments = []
#         for index in range(7, -1, -1):
#             # Shift the tensor by 2 bits for each segment index
#             shifted_tensor = tensor_int16 >> (2 * index)
#             # Apply bitwise AND operation with 0b11 to extract the 2-bit segment
#             segment = shifted_tensor & 0b11
#             segments.append(segment)
#         # Create a tensor by stacking the segments along a new dimension
#         segments_tensor = torch.stack(segments, dim=-1).float()
#         # Reshape the segments tensor to split along the third dimension
#         reshaped_tensor = segments_tensor.transpose(3, 4).reshape(size[0], size[1], -1, size[3]).cuda()
#         return reshaped_tensor

#     @staticmethod
#     def backward(ctx, grad_output):
#         #print("backward_for bitslice", grad_output,grad_output.size())
#         grad_input = torch.zeros(ctx.size).cuda()
#         pos_SA00, pos_SA01, pos_SA10, pos_SA11, = ctx.saved_tensors
#         temp =  grad_output.clone().cuda()
#         inverted_result = torch.logical_or(torch.logical_or(pos_SA00, pos_SA01), torch.logical_or(pos_SA10, pos_SA11)).cuda()
#         result = (~inverted_result).float()
#         verticalsize = ctx.size[3]
#         chunksize = int(verticalsize / 8)
#         # Reshape the tensor to split dimension 3 into groups of 8 elements
#         for i in range(chunksize):
#             if ctx.saf_enable == 1:
#                 masked_tensor = torch.zeros(temp.shape)
#                 masked_tensor[:,:,8*i,:] = (temp[:,:,8*i,:] * result[:,:,8*i,:])/ 2 ** 14 
#                 masked_tensor[:,:,8*i+1,:] = (temp[:,:,8*i+1,:] * result[:,:,8*i+1,:]) / 2 ** 12 
#                 masked_tensor[:,:,8*i+2,:] = (temp[:,:,8*i+2,:] * result[:,:,8*i+2,:]) / 2 ** 10 
#                 masked_tensor[:,:,8*i+3,:] = (temp[:,:,8*i+3,:] * result[:,:,8*i+3,:]) / 2 ** 8
#                 masked_tensor[:,:,8*i+4,:] = (temp[:,:,8*i+4,:] * result[:,:,8*i+4,:]) / 2 ** 6
#                 masked_tensor[:,:,8*i+5,:] = (temp[:,:,8*i+5,:] * result[:,:,8*i+5,:]) / 2 ** 4 
#                 masked_tensor[:,:,8*i+6,:] = (temp[:,:,8*i+6,:] * result[:,:,8*i+6,:]) / 2 ** 2 
#                 masked_tensor[:,:,8*i+7,:] = (temp[:,:,8*i+7,:] * result[:,:,8*i+7,:])
                
#                 for j in range(masked_tensor.size(3)):
#                     column = masked_tensor[:, :, :, j]
#                     #print("size", column.size())
#                     #sign = torch.sign(column)
#                     abs_tensor = torch.abs(column)
#                     max_values, index = torch.max(abs_tensor, dim=2)
#                     ##print("Max Values:", max_values)
#                     #print("Index of Max Values:", index)

#                     signs = torch.sign(column)
#                     max_signs = torch.gather(signs, dim=2, index=index.unsqueeze(-1))
#                     #print("Signs of Max Values:", max_signs)
                    
#                     max_values = max_values * max_signs.squeeze(-1)
#                     #print("column matrix", column, j)
#                     grad_input[:, :, i, j] = max_values
#             else:
#                 grad_input[:,:,i,:] = temp[:,:,8*i+7,:] 
            
#         #print("backward_from bitslice",grad_input, grad_input.size())
#         return grad_input,None, None, None, None, None
def test_w2g_module_output_conductance_range():
    '''
    ensure the w2g module has the correct output conductance range
    which is between G_min and G_max.
    '''

    return
