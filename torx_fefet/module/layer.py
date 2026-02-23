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
# config_path = os.path.join(os.path.dirname(__file__), '../../../config/new_config.json')

# Load the JSON configuration


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

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True,groups=1,dilation=1,device=None, config_path=None, layer_count = 0):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        with open(config_path+'/config.json', 'r') as config_file:
            config = json.load(config_file)

       
        # self.device = config['device']
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.enable_ec_SAF = config['enable_ec_SAF']
        self.enable_ec_SAF_msb = config['enable_ec_SAF_msb']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        self.sa00_rate = config['sa00_rate']
        self.sa01_rate = config['sa01_rate']
        self.sa10_rate = config['sa10_rate']
        self.sa11_rate = config['sa11_rate']

        self.fault_rate = config['fault_rate']
        # print("fault_rate",self.fault_rate) 
        self.fault_dist = config['fault_dist']
        self.layer_count = layer_count
        self.config_path = config_path
             
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
        self.Gint1 = 6.15e-6
        self.Gint2 = 9.858e-5
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2 -1)  # conductance step
        # self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gmax-2*self.delta_g, G_SA10=self.Gmin+2*self.delta_g, G_SA11=self.Gmax,
        #                weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
        #                enable_rand=True, enable_SAF=config['enable_SAF'], fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb, 
        #                device=self.device, config_path=self.config_path, layer_count=self.layer_count)
        self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gint1, G_SA10=self.Gint2, G_SA11=self.Gmax,
                       weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
                       enable_rand=True, enable_SAF=config['enable_SAF'], fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb, 
                       device=self.device, config_path=self.config_path, layer_count=self.layer_count)

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

         ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_stochastic_noise = config['enable_noise']
        self.freq = config['freq']  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = config['temp']  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

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

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl_w * self.scaler_dw
            # if self.training:
            #     self.counter.data += 1
            #     self.delta_x = (input.abs().max() / self.h_lvl).to(self.device)
            #     self.delta_in_sum.data += self.delta_x
            # else:
            self.delta_x = input.abs().max() / self.h_lvl
        #print("input",input,input.size())
        input_clip = F.hardtanh(input, min_val=-self.h_lvl * self.delta_x.item(),
                                max_val=self.h_lvl * self.delta_x.item())
        #print("input_clip",input_clip,input_clip.size())
        input_quan = quantize_input(
            input_clip, self.delta_x) * self.delta_v  # convert to voltage

        #print("original_weights",self.weight,self.weight.size())
        weight_quan = quantize_weight(self.weight, self.delta_w)
        #print("quantized weight,",weight_quan)
        #print("Max Con",self.Gmax,"Min Con",self.Gmin)
        

        # 2. Perform the computation between input voltage and weight conductance
        if self.h_out is None and self.w_out is None:
            self.h_out = int(
                (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
            self.w_out = int(
                (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        # 2.1 flatten and unfold the weight and input
        #print("input_quan",input_quan,input_quan.size())
        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = weight_quan.view(self.out_channels, -1)

        # 2.2. add paddings
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_unfold, self.input_pad,
                             mode='constant', value=0)
        #print("input_padded", input_padded)
        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, input_padded.shape[2])
        weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        #print("crxb_weight",weight_crxb)
        G_crxb = self.w2g(weight_crxb).to(self.device)
        #print("crxb_weight",G_crxb[0], G_crxb.size())
        #quit(0)

        # 2.4. compute matrix multiplication followed by reshapes

        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_stochastic_noise:
            rand_p = nn.Parameter(torch.Tensor(G_crxb.shape),
                                   requires_grad=False)
            rand_g = nn.Parameter(torch.Tensor(G_crxb.shape),
                                  requires_grad=False)
            if device.type == "cuda":
                rand_p = rand_p.cuda()
                rand_g = rand_g.cuda()
            with torch.no_grad():
                input_reduced = (input_crxb.norm(p=2, dim=0).norm(p=2, dim=3).unsqueeze(dim=3)) / \
                                (input_crxb.shape[0] * input_crxb.shape[3])
                grms = torch.sqrt(
                    G_crxb * self.freq * (4 * self.kb * self.temp + 2 * self.q * input_reduced) / (input_reduced ** 2) \
                    + (self.delta_g / 3) ** 2)

                grms[torch.isnan(grms)] = 0
                grms[grms.eq(float('inf'))] = 0

                rand_p.uniform_()
                rand_g.normal_(0, 1)
                G_p = G_crxb * (self.b * G_crxb + self.a) / (G_crxb - (self.b * G_crxb + self.a))
                G_p[rand_p.ge(self.tau)] = 0
                G_g = grms * rand_g
            G_crxb += (G_g.cuda() + G_p)


        # this block is to calculate the ir drop of the crossbar
        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col, self.crxb_row, self.crxb_size,
                                                        input.shape[0],
                                                        input_padded.shape[2])

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                          torch.matmul(G_crxb[1], input_crxb)
        # print(output_crxb[coordinate])
        #print("G xrxb",G_crxb[:,:,:,0,:],G_crxb.size())
        #print("input_crxb",input_crxb[1,:,:,:,0],input_crxb.size())
        #print("ouput xrxb",output_crxb,output_crxb.size())
        
        
        #print("output",output_add)
        
        # 3. perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            # if self.training:
            #     self.delta_i = (output_crxb.abs().max() / (self.h_lvl_adc))
            #     self.delta_out_sum.data += self.delta_i
            # else:
            self.delta_i = (output_crxb.abs().max() / (self.h_lvl_adc))
            self.delta_y = (self.delta_w * self.delta_x * \
                           self.delta_i / (self.delta_v * self.delta_g))
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_adc * self.delta_i.item(),
                                 max_val=self.h_lvl_adc * self.delta_i.item())
        output_adc = adc(output_clip, self.delta_i, self.delta_y)

        #print("Before EC", output_adc[0,:,:,:,:],output_adc.size())
        #with torch.no_grad():
        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                # print(G_pos_diff.get_device())
                # print(G_neg_diff.get_device())
                ec_scale = (self.delta_y / self.delta_i)
                compensation = (torch.round((torch.matmul((G_pos_diff-G_neg_diff), input_crxb)) / self.delta_i) * self.delta_y).to(self.device)
                #print("compensation", compensation)
                output_adc += compensation            

        output_add = self.shift_and_add(output_adc)
        #print("After S & A", output_add[0,:,:,:,:], output_add.size())
        output_sum = torch.sum(output_add, dim=2).to(self.device)
        #print("final_ouput", output_sum[0,:,:,:])
        
        output = output_sum.view(output_sum.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2],
                                 self.h_out,
                                 self.w_out).index_select(dim=1, index=self.nchout_index)

        if self.bias is not None:
            output += self.bias.unsqueeze(1).unsqueeze(1)


        return output.to(self.device)

    # def _reset_delta(self):
    #     self.delta_in_sum.data[0] = 0
    #     self.delta_out_sum.data[0] = 0
    #     self.counter.data[0] = 0

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

    def __init__(self, in_features, out_features, bias=True,device=None,config_path=None, layer_count = 0):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)
        
        with open(config_path+'/config.json', 'r') as config_file:
            config = json.load(config_file)
        # self.device = config['device']
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ################## Crossbar conversion #############################
        self.crxb_size = config['crxb_size']
        self.enable_ec_SAF = config['enable_ec_SAF']
        self.enable_ec_SAF_msb = config['enable_ec_SAF_msb']
        self.ir_drop = config['ir_drop']
        self.quantize_weights =  config['quantize']
        self.adc_resolution =  config['adc_resolution']
        self.input_quantize =  config['input_quantize']
        self.sa00_rate = config['sa00_rate']
        self.sa01_rate = config['sa01_rate']
        self.sa10_rate = config['sa10_rate']
        self.sa11_rate = config['sa11_rate']
        self.config_path = config_path
        self.layer_count = layer_count
        self.fault_rate = config['fault_rate'] 
        self.fault_dist = config['fault_dist']
        ################## Crossbar conversion #############################
        self.col_size = int(self.crxb_size/(self.quantize_weights/2))

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
        self.Gint1 = 1.0252e-6
        self.Gint2 = 1.643e-5
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 2 -1)  # conductance step
        # self.w2g = w2g(self.delta_g,Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gmax-2*self.delta_g, G_SA10=self.Gmin+2*self.delta_g, G_SA11=self.Gmax,
        #                weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
        #                enable_rand=True, enable_SAF=config['enable_SAF'],fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb,
        #                device=self.device, config_path=self.config_path, layer_count=self.layer_count) 
        self.w2g = w2g(self.delta_g,Gmin=self.Gmin, G_SA00=self.Gmin, G_SA01=self.Gint1, G_SA10=self.Gint2, G_SA11=self.Gmax,
                       weight_shape=weight_crxb.shape, p_SA00=self.sa00_rate, p_SA01=self.sa01_rate, p_SA10=self.sa10_rate,p_SA11=self.sa11_rate, 
                       enable_rand=True, enable_SAF=config['enable_SAF'],fault_rate = self.fault_rate, fault_dist = self.fault_dist, msb_only_ec = self.enable_ec_SAF_msb,
                       device=self.device, config_path=self.config_path, layer_count=self.layer_count) 

        self.Gwire = config['gwire']
        self.Gload = config['gload']
        # DAC
        self.scaler_dw = config['scaler_dw']
        self.Vdd = config['vdd']  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        self.delta_v_adc = self.Vdd / (self.n_lvl_adc - 1)
        # self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.counter = nn.Parameter(torch.Tensor(1), requires_grad=False)

        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_stochastic_noise = config['enable_noise']
        self.freq = config['freq']  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = config['temp']  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

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

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl * self.scaler_dw
            # if self.training:
            #     self.counter.data += 1
            #     self.delta_x = input.abs().max() / self.h_lvl
            #     self.delta_in_sum.data += self.delta_x
            # else:
            self.delta_x = input.abs().max() / self.h_lvl

        input_clip = F.hardtanh(input, min_val=-self.h_lvl * self.delta_x.item(),
                                max_val=self.h_lvl * self.delta_x.item())
        input_quan = quantize_input(
            input_clip, self.delta_x) * self.delta_v  # convert to voltage

        weight_quan = quantize_weight(self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance
        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add padding
        weight_padded = F.pad(weight_quan, self.w_pad,
                              mode='constant', value=0)
        input_padded = F.pad(input_quan, self.input_pad,
                             mode='constant', value=0)
        # 2.3. reshape
        input_crxb = input_padded.view(
            input.shape[0], 1, self.crxb_row, self.crxb_size, 1)
        weight_crxb = weight_padded.view(self.crxb_col, self.col_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.w2g(weight_crxb)

        # 2.4. compute matrix multiplication
        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_stochastic_noise:
            rand_p = nn.Parameter(torch.Tensor(G_crxb.shape),
                                  requires_grad=False)
            rand_g = nn.Parameter(torch.Tensor(G_crxb.shape),
                                  requires_grad=False)

            if self.device.type == "cuda":
                rand_p = rand_p.cuda()
                rand_g = rand_g.cuda()

            with torch.no_grad():
                input_reduced = input_crxb.norm(p=2, dim=0).norm(p=2, dim=3).unsqueeze(dim=3) / (
                            input_crxb.shape[0] * input_crxb.shape[3])
                grms = torch.sqrt(
                    G_crxb * self.freq * (4 * self.kb * self.temp + 2 * self.q * input_reduced) / (input_reduced ** 2) \
                    + (self.delta_g / 3) ** 2)

                grms[torch.isnan(grms)] = 0
                grms[grms.eq(float('inf'))] = 0

                rand_p.uniform_()
                rand_g.normal_(0, 1)
                G_p = G_crxb * (self.b * G_crxb + self.a) / (G_crxb - (self.b * G_crxb + self.a))
                G_p[rand_p.ge(self.tau)] = 0
                G_g = grms * rand_g

            G_crxb += (G_g + G_p)


        # this block is to calculate the ir drop of the crossbar

        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            output_crxb = output_crxb.contiguous().view(self.crxb_col,
                                                        self.crxb_row,
                                                        self.crxb_size,
                                                        input.shape[0],
                                                        1)

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            output_crxb = torch.matmul(G_crxb[0], input_crxb) \
                          - torch.matmul(G_crxb[1], input_crxb)


        #quit(0)
        # 3. perform ADC operation (i.e., current to digital conversion)
        with torch.no_grad():
            # if self.training:
            #     self.delta_i = output_crxb.abs().max() / (self.h_lvl_adc)
            #     self.delta_out_sum.data += self.delta_i
            # else:
            self.delta_i = output_crxb.abs().max() / (self.h_lvl_adc)
            self.delta_y = (self.delta_w * self.delta_x * \
                           self.delta_i / (self.delta_v * self.delta_g))
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_adc * self.delta_i.item(),
                                 max_val=self.h_lvl_adc * self.delta_i.item())
        output_adc = adc(output_clip, self.delta_i, self.delta_y)

        # print(" adc output size", output_adc.size())
        # print("before EC adc output", output_adc[0, 0, 0, 0:8, 0])
#        with torch.no_grad():
        if self.w2g.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y / self.delta_i
                compensation = (torch.round((torch.matmul((G_pos_diff-G_neg_diff), input_crxb)) / self.delta_i) * self.delta_y).to(self.device)
                #print("compensation", compensation)
                output_adc += compensation 
        #print("After EC adc output", output_adc)
        # print("delta i{} and y{} for layer in ".format(self.delta_i, self.delta_y))
        # print("compensation_value" , compensation[0, 0, 0, 0:8, 0])
        # print("output adc after compensation", output_adc[0, 0, 0, 0:8, 0])
        # quit(0)
        #4. Perform Shift and ADD on the ouput crxb 
        output_add = self.shift_and_add(output_adc)
        #shift_and_add_out = self.shift_and_add(output_adc).cuda()
        #print("ouput_add",output_add)
        #quit(0)
        output_sum = torch.sum(output_add, dim=2).squeeze(dim=3).to(self.device)
        output = output_sum.view(input.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2]).index_select(dim=1, index=self.out_index).to(self.device)

        if self.bias is not None:
            output += self.bias
        #print("final output",output_sum)
        
        return output.to(self.device)

    # def _reset_delta(self):
    #     self.delta_in_sum.data[0] = 0
    #     self.delta_out_sum.data[0] = 0
    #     self.counter.data[0] = 0
