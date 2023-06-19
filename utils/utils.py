import jittor as jt
import random
import numpy as np
import torch


def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)

# def initialize_parameters(var):
#     var.assign(torch.rand(var.shape))