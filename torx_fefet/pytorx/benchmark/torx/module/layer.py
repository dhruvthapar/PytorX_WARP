import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import json

from .adc import adc
from .dac import quantize_dac
from .w2g import w2g

quantize_input = quantize_dac
quantize_weight = quantize_dac
adc = adc


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# Construct the path to the config.json file


# quantize = 8

class crxb_Conv2d(nn.Conv2d):
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        vdd(float): supply voltage.
        enable_stochastic_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True,groups=1,dilation=1,device=None, layer_count = None, fits=None, config_path=None):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        assert self.groups == 1, "currently not support grouped convolution for custom conv"
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(f"{config_path}/config.json", 'r') as f:
            config = json.load(f)
        self.config_path = config_path
        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        
        G00_mu = config['G00_mu']
        G01_mu = config['G01_mu']
        G10_mu = config['G10_mu']
        G11_mu = config['G11_mu']
        G00_sigma = config['G00_sigma']
        G01_sigma = config['G01_sigma']
        G10_sigma = config['G10_sigma']
        G11_sigma = config['G11_sigma']
        self.fault_model = config['fault_model']
        self.enable_fault = config['enable_fault']
        self.faulty_layer_idx = config['faulty_layer_idx']
        self.num_crossbars = config['num_crossbars']
        self.fault_rate = config['fault_rate']
        self.fault_dist = config['fault_dist']
        if self.fault_model != None and self.fault_model != "charge_trapping":
            G00_f_mu = config['G00_f_mu']
            G00_f_sigma = config['G00_f_sigma']
            G01_f_mu = config['G01_f_mu']
            G01_f_sigma = config['G01_f_mu']
            G10_f_mu = config['G10_f_mu']
            G10_f_sigma = config['G10_f_sigma']
            G11_f_mu = config['G11_f_mu']
            G11_f_sigma = config['G11_f_sigma']

        ################## Crossbar conversion #############################
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.col_size = int(self.crxb_size/(self.quantize_weights/2))

        self.nchout_index = torch.arange(self.out_channels).to(self.device)
        weight_flatten = self.weight.view(self.out_channels, -1)
        #print("weights_flatten size: ", weight_flatten.size())
        #print("weights_ size: ", self.weight.size())
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            weight_flatten.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad_col(
            weight_flatten.shape[0], self.crxb_size)
        self.h_out = None
        self.w_out = None
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, 0, 0, self.crxb_row_pads)
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)

        #print(f"conv2d layer {layer_count}: {weight_crxb.shape}")
        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2 ** self.input_quantize
        self.h_lvl = (self.n_lvl - 2) / 2
        self.n_lvl_w = 2 ** self.quantize_weights
        self.h_lvl_w = (self.n_lvl_w - 2) / 2
        self.n_lvl_adc = 2 ** (self.adc_resolution)
        self.h_lvl_adc = (self.n_lvl_adc - 2) / 2
        # ReRAM cells
        self.Gmax = config['gmax']  # max conductance
        self.Gmin = config['gmin']  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2 -1)  # conductance step
        self.layer_count = layer_count
        self.warp = config['warp']
        self.w2g = w2g(Gmin=self.Gmin, G00_mu=G00_mu, G01_mu=G01_mu, G10_mu=G10_mu, G11_mu = G11_mu, G00_sigma = G00_sigma, G01_sigma=G01_sigma, G10_sigma=G10_sigma, G11_sigma = G11_sigma,
        weight_shape=weight_crxb.shape, device=self.device, layer_count=layer_count, wl_lsb = config['wl_lsb'], num_cycles = config['num_cycles'], fits = fits, warp = self.warp, var_case = config['var_case'], config = config, config_path = config_path, G00_f_mu=G00_f_mu, G00_f_sigma=G00_f_sigma, G01_f_mu=G01_f_mu, G01_f_sigma=G01_f_sigma, G10_f_mu=G10_f_mu, G10_f_sigma =G10_f_sigma, G11_f_mu=G11_f_mu, G11_f_sigma=G11_f_sigma, fault_model=self.fault_model, enable_fault=self.enable_fault, fault_rate=self.fault_rate, fault_dist=self.fault_dist, faulty_layer_idx=self.faulty_layer_idx, num_crossbars=self.num_crossbars) 

        self.Gwire = config['gwire']
        self.Gload = config['gload']
        # DAC
        self.Vdd = config['vdd']  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        self.delta_v_adc = self.Vdd / (self.n_lvl_adc - 1)
        # self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.counter = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.scaler_dw = config['scaler_dw']


    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    def num_pad_col(self, source, target):
        #crxb_col_fit = target/(self.quantize_weights/2)
        crxb_index = math.ceil((source * (self.quantize_weights/2)) / target)
        num_padding =int( crxb_index * target / (self.quantize_weights/2) - source)
        return crxb_index, num_padding

    def shift_and_add(self, adc_out):
        # Define the dimensions
        tensorsize = adc_out.shape
        verticalsize = tensorsize[3]
        chunksize = int(verticalsize / 8)
        # Reshape the tensor to split dimension 3 into groups of 8 elements
        shift_and_added = torch.zeros(tensorsize[0], tensorsize[1], tensorsize[2], chunksize, tensorsize[4])#, device=adc_out.device, dtype=adc_out.dtype)
        for i in range(chunksize):
            shift_and_added[:,:,:,i,:] = adc_out[:,:,:,8*i,:] * 2 ** 14 + \
                                        adc_out[:,:,:,8*i+1,:] * 2 ** 12 + \
                                        adc_out[:,:,:,8*i+2,:] * 2 ** 10 + \
                                        adc_out[:,:,:,8*i+3,:]  * 2 ** 8 + \
                                        adc_out[:,:,:,8*i+4,:] * 2 ** 6 + \
                                        adc_out[:,:,:,8*i+5,:] * 2 ** 4 + \
                                        adc_out[:,:,:,8*i+6,:]* 2 ** 2 + \
                                        adc_out[:,:,:,8*i+7,:]
        return shift_and_added
    
    def _adc_shift_sum(self, output_crxb, delta_g):
        # output_crxb: same shape as before
        max_abs = output_crxb.abs().max()

        if max_abs.item() == 0.0:
            # Return a correctly-shaped ZERO output_sum for this branch.
            # We can infer output_sum shape by running shift_and_add on a tiny zero adc_out,
            # but simplest: build it from the standard pipeline shapes.
            # Here we do it robustly by short-circuiting after shift_and_add shape inference.
            # Create a fake adc_out with same shape as output_crxb
            adc_out = torch.zeros_like(output_crxb)
            out_add = self.shift_and_add(adc_out)
            out_sum = torch.sum(out_add, dim=2)
            return out_sum

        with torch.no_grad():
            delta_i = max_abs / self.h_lvl_adc
            delta_y = (self.delta_w * self.delta_x * delta_i) / (self.delta_v * delta_g)

        output_clip = F.hardtanh(
            output_crxb,
            min_val=-self.h_lvl_adc * delta_i.item(),
            max_val=self.h_lvl_adc * delta_i.item()
        )
        output_adc = adc(output_clip, delta_i, delta_y)
        output_add = self.shift_and_add(output_adc)
        output_sum = torch.sum(output_add, dim=2)
        return output_sum
    
    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl_w * self.scaler_dw
            self.delta_x = input.abs().max() / self.h_lvl
        input_clip = F.hardtanh(input, min_val=-self.h_lvl * self.delta_x.item(),
                                max_val=self.h_lvl * self.delta_x.item())
        input_quan = quantize_input(input_clip, self.delta_x) * self.delta_v  # convert to voltage
        weight_quan = quantize_weight(self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance
        if self.h_out is None and self.w_out is None:
            self.h_out = int((input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
            self.w_out = int((input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = weight_quan.view(self.out_channels, -1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad, mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad, mode='constant', value=0)

        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                    self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
                                        self.crxb_row, self.crxb_size).transpose(1, 2)

        # convert the floating point weight into conductance pair values

        if self.fault_model == "charge_trapping":
            # -------- NEW: w2g returns (G_active, G_inactive) --------
            G_crxb_active, G_crxb_inactive = self.w2g(weight_crxb)
            G_crxb_active = G_crxb_active.to(self.device)
            G_crxb_inactive = G_crxb_inactive.to(self.device)

            # Compute active/inactive analog MACs separately
            output_crxb_active = torch.matmul(G_crxb_active[0], input_crxb) - torch.matmul(G_crxb_active[1], input_crxb)
            output_crxb_inactive = torch.matmul(G_crxb_inactive[0], input_crxb) - torch.matmul(G_crxb_inactive[1], input_crxb)

            # 3. perform ADC operation (i.e., current to digital conversion)
            if self.warp == True:
                with open(f"{self.config_path}/config_active_{self.layer_count}.json", 'r') as f:
                    config_active = json.load(f)
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_inactive = json.load(f)
            else:
                # both config_active and config_inactive are same
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_active = json.load(f)
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_inactive = json.load(f)

            gmax_active = config_active['gmax']
            gmax_inactive = config_inactive['gmax']

            # 2-bit delta_g
            delta_g_active = (gmax_active - self.Gmin) / (2 ** 2 - 1)
            delta_g_inactive = (gmax_inactive - self.Gmin) / (2 ** 2 - 1)

            output_sum_active = self._adc_shift_sum(output_crxb_active, delta_g_active).to(self.device)
            output_sum_inactive = self._adc_shift_sum(output_crxb_inactive, delta_g_inactive).to(self.device)

            # Combine before reshape/index_select (keeps your original flow)
            output_sum =(output_sum_active + output_sum_inactive).to(self.device)

            output = output_sum.view(
                output_sum.shape[0],
                output_sum.shape[1] * output_sum.shape[2],
                self.h_out,
                self.w_out
            ).index_select(dim=1, index=self.nchout_index)

            if self.bias is not None:
                output += self.bias.unsqueeze(1).unsqueeze(1)

        else:
            # convert the floating point weight into conductance pair values
            G_crxb = self.w2g(weight_crxb).to(self.device)

            # 2.4. compute matrix multiplication followed by reshapes
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                            torch.matmul(G_crxb[1], input_crxb)
            with torch.no_grad():
                self.delta_i = (output_crxb.abs().max() / (self.h_lvl_adc))
                self.delta_y = (self.delta_w * self.delta_x * \
                            self.delta_i / (self.delta_v * self.delta_g))
            output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_adc * self.delta_i.item(),
                                    max_val=self.h_lvl_adc * self.delta_i.item())
            output_adc = adc(output_clip, self.delta_i, self.delta_y)       
            output_add = self.shift_and_add(output_adc)
            output_sum = torch.sum(output_add, dim=2).to(self.device)
            output = output_sum.view(output_sum.shape[0],
                                    output_sum.shape[1] * output_sum.shape[2],
                                    self.h_out,
                                    self.w_out).index_select(dim=1, index=self.nchout_index)
            if self.bias is not None:
                output += self.bias.unsqueeze(1).unsqueeze(1)
        return output.to(self.device)


class crxb_Linear(nn.Linear):
    """
    This is the custom linear layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        enable_stochastic_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, in_features, out_features, bias=True,device=None, layer_count = None, fits=None, config_path=None):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with open(f"{config_path}/config.json", 'r') as f:
            config = json.load(f)
        self.config_path = config_path
        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        G00_mu = config['G00_mu']
        G01_mu = config['G01_mu']
        G10_mu = config['G10_mu']
        G11_mu = config['G11_mu']
        G00_sigma = config['G00_sigma']
        G01_sigma = config['G01_sigma']
        G10_sigma = config['G10_sigma']
        G11_sigma = config['G11_sigma']
        self.fault_model = config['fault_model']
        self.enable_fault = config['enable_fault']
        self.faulty_layer_idx = config['faulty_layer_idx']
        self.num_crossbars = config['num_crossbars']
        self.fault_rate = config['fault_rate']
        self.fault_dist = config['fault_dist']
        if self.fault_model != None and self.fault_model != "charge_trapping":
            G00_f_mu = config['G00_f_mu']
            G00_f_sigma = config['G00_f_sigma']
            G01_f_mu = config['G01_f_mu']
            G01_f_sigma = config['G01_f_mu']
            G10_f_mu = config['G10_f_mu']
            G10_f_sigma = config['G10_f_sigma']
            G11_f_mu = config['G11_f_mu']
            G11_f_sigma = config['G11_f_sigma']
        ################## Crossbar conversion #############################
        self.col_size = int(self.crxb_size/(self.quantize_weights/2)) # is 2 for pos and neg?
        self.out_index = torch.arange(out_features).to(self.device)
        self.crxb_row, self.crxb_row_pads = self.num_pad(
            self.weight.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad_col(
            self.weight.shape[0], self.crxb_size)
        self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        self.input_pad = (0, self.crxb_row_pads)
        weight_padded = F.pad(self.weight, self.w_pad,
                              mode='constant', value=0)
        weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        #print(f"linear layer {layer_count}: {weight_crxb.shape}")
        ################# Hardware conversion ##############################
        # weight and input levels
        self.n_lvl = 2 ** self.input_quantize
        self.h_lvl = (self.n_lvl - 2) / 2
        self.n_lvl_w = 2 ** self.quantize_weights
        self.h_lvl_w = (self.n_lvl_w - 2) / 2
        self.n_lvl_adc = 2 ** (self.adc_resolution)
        self.h_lvl_adc = (self.n_lvl_adc - 2) / 2
        # ReRAM cells
        self.Gmax = config['gmax']  # max conductance
        self.Gmin = config['gmin']  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2 -1)  # conductance step
        self.layer_count = layer_count
        self.warp = config['warp']
        self.w2g = w2g(Gmin=self.Gmin, G00_mu=G00_mu, G01_mu=G01_mu, G10_mu=G10_mu, G11_mu = G11_mu, G00_sigma = G00_sigma, G01_sigma=G01_sigma, G10_sigma=G10_sigma, G11_sigma = G11_sigma,
        weight_shape=weight_crxb.shape, device=self.device, layer_count=layer_count, wl_lsb = config['wl_lsb'], num_cycles = config['num_cycles'], fits = fits, warp = self.warp, var_case = config['var_case'], config = config, config_path = config_path, G00_f_mu=G00_f_mu, G00_f_sigma=G00_f_sigma, G01_f_mu=G01_f_mu, G01_f_sigma=G01_f_sigma, G10_f_mu=G10_f_mu, G10_f_sigma =G10_f_sigma, G11_f_mu=G11_f_mu, G11_f_sigma=G11_f_sigma, fault_model=self.fault_model, enable_fault=self.enable_fault, fault_rate=self.fault_rate, fault_dist=self.fault_dist, faulty_layer_idx=self.faulty_layer_idx, num_crossbars=self.num_crossbars)  

        self.Gwire = config['gwire']
        self.Gload = config['gload']
        # DAC
        self.scaler_dw = config['scaler_dw']
        self.Vdd = config['vdd']  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        self.delta_v_adc = self.Vdd / (self.n_lvl_adc - 1)

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding
    def num_pad_col(self, source, target):
        crxb_col_fit = target/(self.quantize_weights/2)
        crxb_index = math.ceil((source * (self.quantize_weights/2)) / target)
        num_padding =int( crxb_index * target / (self.quantize_weights/2) - source)
        return crxb_index, num_padding

    def shift_and_add(self, adc_out):
        # Define the dimensions
        tensorsize = adc_out.shape
        verticalsize = tensorsize[3]
        chunksize = int(verticalsize / 8)
        # Reshape the tensor to split dimension 3 into groups of 8 elements
        shift_and_added = torch.zeros(tensorsize[0], tensorsize[1], tensorsize[2], chunksize, tensorsize[4])

        for i in range(chunksize):
            shift_and_added[:,:,:,i,:] = adc_out[:,:,:,8*i,:] * 2 ** 14 + \
                                        adc_out[:,:,:,8*i+1,:] * 2 ** 12 + \
                                        adc_out[:,:,:,8*i+2,:] * 2 ** 10 + \
                                        adc_out[:,:,:,8*i+3,:]  * 2 ** 8 + \
                                        adc_out[:,:,:,8*i+4,:] * 2 ** 6 + \
                                        adc_out[:,:,:,8*i+5,:] * 2 ** 4 + \
                                        adc_out[:,:,:,8*i+6,:]* 2 ** 2 + \
                                        adc_out[:,:,:,8*i+7,:]
        return shift_and_added

    def _adc_shift_sum(self, output_crxb, delta_g):
        # output_crxb: same shape as before
        max_abs = output_crxb.abs().max()

        if max_abs.item() == 0.0:
            # Return a correctly-shaped ZERO output_sum for this branch.
            # We can infer output_sum shape by running shift_and_add on a tiny zero adc_out,
            # but simplest: build it from the standard pipeline shapes.
            # Here we do it robustly by short-circuiting after shift_and_add shape inference.
            # Create a fake adc_out with same shape as output_crxb
            adc_out = torch.zeros_like(output_crxb)
            out_add = self.shift_and_add(adc_out)
            out_sum = torch.sum(out_add, dim=2)
            return out_sum

        with torch.no_grad():
            delta_i = max_abs / self.h_lvl_adc
            delta_y = (self.delta_w * self.delta_x * delta_i) / (self.delta_v * delta_g)

        output_clip = F.hardtanh(
            output_crxb,
            min_val=-self.h_lvl_adc * delta_i.item(),
            max_val=self.h_lvl_adc * delta_i.item()
        )
        output_adc = adc(output_clip, delta_i, delta_y)
        output_add = self.shift_and_add(output_adc)
        output_sum = torch.sum(output_add, dim=2)
        return output_sum

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl * self.scaler_dw
            self.delta_x = input.abs().max() / self.h_lvl

        input_clip = F.hardtanh(
            input,
            min_val=-self.h_lvl * self.delta_x.item(),
            max_val=self.h_lvl * self.delta_x.item()
        )
        input_quan = quantize_input(input_clip, self.delta_x) * self.delta_v
        weight_quan = quantize_weight(self.weight, self.delta_w)

        # 2. padding
        weight_padded = F.pad(weight_quan, self.w_pad, mode='constant', value=0)
        input_padded = F.pad(input_quan, self.input_pad, mode='constant', value=0)

        # 3. reshape to crxb
        input_crxb = input_padded.view(
            input.shape[0], 1, self.crxb_row, self.crxb_size, 1
        )
        weight_crxb = weight_padded.view(
            self.crxb_col, self.col_size, self.crxb_row, self.crxb_size
        ).transpose(1, 2)

        # -------- NEW: active / inactive split --------
        if self.fault_model == "charge_trapping":
            G_crxb_active, G_crxb_inactive = self.w2g(weight_crxb)
            G_crxb_active = G_crxb_active.to(self.device)
            G_crxb_inactive = G_crxb_inactive.to(self.device)

            # Analog MACs
            output_crxb_active = (
                torch.matmul(G_crxb_active[0], input_crxb)
                - torch.matmul(G_crxb_active[1], input_crxb)
            )

            output_crxb_inactive = (
                torch.matmul(G_crxb_inactive[0], input_crxb)
                - torch.matmul(G_crxb_inactive[1], input_crxb)
            )

            # 4. ADC
            if self.warp == True:
                with open(f"{self.config_path}/config_active_{self.layer_count}.json", 'r') as f:
                    config_active = json.load(f)
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_inactive = json.load(f)
            else:
                # both config_active and config_inactive are same
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_active = json.load(f)
                with open(f"{self.config_path}/config.json", 'r') as f:
                    config_inactive = json.load(f)

            gmax_active = config_active['gmax']
            gmax_inactive = config_inactive['gmax']

            delta_g_active = (gmax_active - self.Gmin) / (2 ** 2 - 1)
            delta_g_inactive = (gmax_inactive - self.Gmin) / (2 ** 2 - 1)

            output_sum_active   = self._adc_shift_sum(output_crxb_active,   delta_g_active).squeeze(dim=3)
            output_sum_inactive = self._adc_shift_sum(output_crxb_inactive, delta_g_inactive).squeeze(dim=3)

            # -------- MERGE --------
            output_sum = (output_sum_active + output_sum_inactive).to(self.device)

            output = output_sum.view(
                input.shape[0],
                output_sum.shape[1] * output_sum.shape[2]
            ).index_select(dim=1, index=self.out_index)

            if self.bias is not None:
                output += self.bias

        else:
            # convert the floating point weight into conductance pair values
            G_crxb = self.w2g(weight_crxb)
            output_crxb = torch.matmul(G_crxb[0], input_crxb) \
                            - torch.matmul(G_crxb[1], input_crxb)

            with torch.no_grad():
                self.delta_i = output_crxb.abs().max() / (self.h_lvl_adc)
                self.delta_y = (self.delta_w * self.delta_x * \
                            self.delta_i / (self.delta_v * self.delta_g))

            output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_adc * self.delta_i.item(),
                                    max_val=self.h_lvl_adc * self.delta_i.item())
            output_adc = adc(output_clip, self.delta_i, self.delta_y)
            output_add = self.shift_and_add(output_adc)
            output_sum = torch.sum(output_add, dim=2).squeeze(dim=3).to(self.device)
            output = output_sum.view(input.shape[0],
                                    output_sum.shape[1] * output_sum.shape[2]).index_select(dim=1, index=self.out_index).to(self.device)
            if self.bias is not None:
                output += self.bias

        return output.to(self.device)