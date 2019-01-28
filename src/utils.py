import os
import sys
import torch
import numpy as np
import scipy.stats


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        print("Creating {}".format(dir_path))
        os.makedirs(dir_path)


def relative_to_src_dir(target_path):
    script_path = os.path.dirname(__file__)
    return os.path.abspath(os.path.join(script_path, target_path))


def have_gpu():
    return torch.cuda.is_available()

class OutputManager(object):
    def __init__(self, result_path):
        self.log_file = open(os.path.join(result_path,'log.txt'),'w')

    def say(self, s):
        self.log_file.write("{}".format(s))
        self.log_file.flush()
        sys.stdout.write("{}".format(s))
        sys.stdout.flush()

def negate_gradient(_optim):
    for entry in _optim.param_groups:
        params = entry['params']
        for tensor in params:
            tensor.grad.data *= -1.0

def get_trainable_params(obj):
    return [p for p in obj.parameters() if p.requires_grad]


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return [m, h]
