import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
from sklearn.mixture import GaussianMixture




def inject_pv(input, input_scaled, g_dict, pv_labels):
    output = input.clone()
    output[(input_scaled == 0) & (pv_labels == 0)] = g_dict["00"][0]
    output[(input_scaled == 1) & (pv_labels == 0)] = g_dict["00"][1]
    output[(input_scaled == 2) & (pv_labels == 0)] = g_dict["00"][2]
    output[(input_scaled == 3) & (pv_labels == 0)] = g_dict["00"][3]

    output[(input_scaled == 0) & (pv_labels == 1)] = g_dict["01"][0]
    output[(input_scaled == 1) & (pv_labels == 1)] = g_dict["01"][1]
    output[(input_scaled == 2) & (pv_labels == 1)] = g_dict["01"][2]
    output[(input_scaled == 3) & (pv_labels == 1)] = g_dict["01"][3]

    output[(input_scaled == 0) & (pv_labels == 2)] = g_dict["0-1"][0]
    output[(input_scaled == 1) & (pv_labels == 2)] = g_dict["0-1"][1]
    output[(input_scaled == 2) & (pv_labels == 2)] = g_dict["0-1"][2]
    output[(input_scaled == 3) & (pv_labels == 2)] = g_dict["0-1"][3]

    output[(input_scaled == 0) & (pv_labels == 3)] = g_dict["10"][0]
    output[(input_scaled == 1) & (pv_labels == 3)] = g_dict["10"][1]
    output[(input_scaled == 2) & (pv_labels == 3)] = g_dict["10"][2]
    output[(input_scaled == 3) & (pv_labels == 3)] = g_dict["10"][3]

    output[(input_scaled == 0) & (pv_labels == 4)] = g_dict["11"][0]
    output[(input_scaled == 1) & (pv_labels == 4)] = g_dict["11"][1]
    output[(input_scaled == 2) & (pv_labels == 4)] = g_dict["11"][2]
    output[(input_scaled == 3) & (pv_labels == 4)] = g_dict["11"][3]

    output[(input_scaled == 0) & (pv_labels == 5)] = g_dict["1-1"][0]
    output[(input_scaled == 1) & (pv_labels == 5)] = g_dict["1-1"][1]
    output[(input_scaled == 2) & (pv_labels == 5)] = g_dict["1-1"][2]
    output[(input_scaled == 3) & (pv_labels == 5)] = g_dict["1-1"][3]

    output[(input_scaled == 0) & (pv_labels == 6)] = g_dict["-10"][0]
    output[(input_scaled == 1) & (pv_labels == 6)] = g_dict["-10"][1]
    output[(input_scaled == 2) & (pv_labels == 6)] = g_dict["-10"][2]
    output[(input_scaled == 3) & (pv_labels == 6)] = g_dict["-10"][3]

    output[(input_scaled == 0) & (pv_labels == 7)] = g_dict["-11"][0]
    output[(input_scaled == 1) & (pv_labels == 7)] = g_dict["-11"][1]
    output[(input_scaled == 2) & (pv_labels == 7)] = g_dict["-11"][2]
    output[(input_scaled == 3) & (pv_labels == 7)] = g_dict["-11"][3]

    output[(input_scaled == 0) & (pv_labels == 8)] = g_dict["-1-1"][0]
    output[(input_scaled == 1) & (pv_labels == 8)] = g_dict["-1-1"][1]
    output[(input_scaled == 2) & (pv_labels == 8)] = g_dict["-1-1"][2]
    output[(input_scaled == 3) & (pv_labels == 8)] = g_dict["-1-1"][3]

    return output

def inject_fc1(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["000"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["000"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["000"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["000"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["010"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["010"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["010"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["010"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-10"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-10"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-10"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-10"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["100"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["100"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["100"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["100"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["110"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["110"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["110"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["110"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-10"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-10"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-10"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-10"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-100"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-100"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-100"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-100"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-110"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-110"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-110"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-110"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-10"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-10"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-10"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-10"][3]

    return output

def inject_fc2(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["001"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["001"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["001"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["001"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["011"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["011"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["011"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["011"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-11"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-11"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-11"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-11"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["101"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["101"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["101"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["101"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["111"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["111"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["111"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["111"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-11"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-11"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-11"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-11"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-101"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-101"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-101"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-101"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-111"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-111"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-111"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-111"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-11"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-11"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-11"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-11"][3]
    return output

def inject_fc3(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["002"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["002"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["002"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["002"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["012"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["012"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["012"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["012"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-12"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-12"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-12"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-12"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["102"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["102"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["102"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["102"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["112"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["112"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["112"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["112"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-12"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-12"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-12"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-12"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-102"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-102"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-102"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-102"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-112"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-112"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-112"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-112"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-12"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-12"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-12"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-12"][3]
    return output

def inject_fc4(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["003"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["003"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["003"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["003"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["013"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["013"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["013"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["013"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-13"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-13"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-13"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-13"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["103"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["103"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["103"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["103"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["113"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["113"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["113"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["113"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-13"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-13"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-13"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-13"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-103"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-103"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-103"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-103"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-113"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-113"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-113"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-113"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-13"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-13"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-13"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-13"][3]
    return output

def inject_sap0(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["004"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["004"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["004"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["004"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["014"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["014"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["014"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["014"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-14"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-14"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-14"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-14"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["104"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["104"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["104"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["104"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["114"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["114"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["114"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["114"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-14"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-14"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-14"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-14"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-104"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-104"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-104"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-104"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-114"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-114"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-114"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-114"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-14"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-14"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-14"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-14"][3]
    return output

def inject_sapp(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["005"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["005"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["005"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["005"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["015"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["015"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["015"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["015"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-15"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-15"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-15"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-15"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["105"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["105"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["105"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["105"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["115"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["115"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["115"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["115"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-15"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-15"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-15"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-15"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-105"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-105"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-105"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-105"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-115"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-115"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-115"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-115"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-15"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-15"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-15"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-15"][3]
    return output

def inject_sapn(input, input_scaled, g_dict, pv_labels, fault_labels):
    output = input.clone()
    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["006"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["006"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["006"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 0)] = g_dict["006"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["016"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["016"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["016"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 1)] = g_dict["016"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-16"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-16"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-16"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 2)] = g_dict["0-16"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["106"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["106"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["106"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 3)] = g_dict["106"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["116"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["116"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["116"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 4)] = g_dict["116"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-16"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-16"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-16"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 5)] = g_dict["1-16"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-106"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-106"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-106"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 6)] = g_dict["-106"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-116"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-116"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-116"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 7)] = g_dict["-116"][3]

    output[(input_scaled == 0) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-16"][0]
    output[(input_scaled == 1) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-16"][1]
    output[(input_scaled == 2) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-16"][2]
    output[(input_scaled == 3) & (fault_labels == 1) & (pv_labels == 8)] = g_dict["-1-16"][3]
    return output
