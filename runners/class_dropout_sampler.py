from base import get_runner # HAS TO BE FIRST OR SEGFAULT HAPPENS

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
from torch.utils.data.sampler import Sampler
from pytorch_metric_learning.utils import common_functions as c_f
from powerful_benchmarker.factories.hook_factory import HookFactory
from powerful_benchmarker.factories.base_factory import BaseFactory

from sklearn.metrics import f1_score
from pytorch_metric_learning.utils import stat_utils
from pytorch_metric_learning.utils.accuracy_calculator import get_lone_query_labels, get_label_counts



class ClassDropoutSampler(Sampler):
    def __init__(self, labels, m, d=0.1, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m = int(m)
        self.d = d

        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
    
        
        self.length_of_single_pass = self.m*len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = []
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        self.shuffled_labels = list(self.labels)
        for _ in range(num_iters):
            dropped_labels = np.random.choice(self.labels, size=int(len(self.labels)*self.d))
            c_f.NUMPY_RANDOM.shuffle(self.shuffled_labels)
            for label in self.shuffled_labels:
                if label in dropped_labels:
                    continue
                t = self.labels_to_indices[label]
                idx_list += list(c_f.safe_random_choice(t, size=self.m))
                i += self.m
        return iter(idx_list)



r = get_runner()
r.register("sampler", ClassDropoutSampler)
r.run()