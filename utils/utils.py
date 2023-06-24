import jittor as jt
import random
import numpy as np



def set_random_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    jt.set_global_seed(seed)

