import torch

def adc(input, delta_i, delta_y):
    output = torch.round(input / delta_i) * delta_y
    return output

# class _adc(torch.autograd.Function):


#     @staticmethod
#     def forward(ctx, input, delta_i, delta_y):
#         ctx.delta_i = delta_i
#         ctx.delta_y = delta_y
#         #print("before adc", input)
#         # adc_out = torch.round(input / ctx.delta_i)
#         output = torch.round(input / ctx.delta_i) * ctx.delta_y
#         #print("shift_added", shift_and_added)
#         #print("after adc",output)
#         #quit(0)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         #print("backward", grad_output)
#         grad_input = ctx.delta_y * grad_output.clone() / ctx.delta_i, None, None
#         return grad_input

