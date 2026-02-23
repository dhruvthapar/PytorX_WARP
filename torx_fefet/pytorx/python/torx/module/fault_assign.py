import torch
import random

from .layer import crxb_Conv2d
from .layer import crxb_Linear

# Generate the SA0, SA1 fault rate from tile fault rate and assign to the model
def inject_fault_rate(model, dyn_fault_rate=0.001, dyn_xb_rate=1, prob_faults = [0.25,0.25,0.25,0.25]):

    # Assign the fault rate to each tile
    for name, module in model.named_modules():
        if isinstance(module, crxb_Conv2d) or isinstance(module, crxb_Linear):
            module.w2g.SAF_pos.dyn_injection(dyn_fault_rate=0.001, dyn_xb_rate=1, prob_faults = [0.25,0.25,0.25,0.25])
            module.w2g.SAF_neg.dyn_injection(dyn_fault_rate=0.001, dyn_xb_rate=1, prob_faults = [0.25,0.25,0.25,0.25])


