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


from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

def dist_from_centroid_regularizer(embeddings, labels, threshold=0.1):
    centroids = {}
    
    reg_val = 0
    
    for label in torch.unique(labels):
        label_indices = torch.where(labels==label)
        label_embeddings = embeddings[label_indices]
        centroids[label] = torch.mean(label_embeddings, axis=0)
    
    all_centroids = torch.stack(list(centroids.values()))
    pairwise_centroid_dists = lmu.get_pairwise_mat(all_centroids, all_centroids, use_similarity=False, squared=False)
    for i in range(len(all_centroids)):
        for j in range(i+1, len(all_centroids)):
            reg_val += (threshold - pairwise_centroid_dists[i][j]).clamp(0, 1).sum()
    
    
    return reg_val
    
class RegularizerReducer(AvgNonZeroReducer):
    def __init__(self, weight=0.5, threshold=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = weight
        self.threshold = threshold
        
        self.add_to_recordable_attributes(name="reg_val")
        

    def forward(self, loss_dict, embeddings, labels):
        self.reg_val = 0
        loss_val = super().forward(loss_dict, embeddings, labels)
        self.reg_val = self.weight * dist_from_centroid_regularizer(embeddings, labels, threshold=self.threshold)
        
        return loss_val + self.reg_val


r = get_runner()
r.register("reducer", RegularizerReducer)
r.run()