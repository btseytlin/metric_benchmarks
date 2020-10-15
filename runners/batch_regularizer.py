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
from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
def dist_from_centroid_regularizer(embeddings, labels, reg_own_weight=1, reg_own_threshold=0.1, reg_other_weight=1, reg_other_threshold=0.2):
    uniq_labels = np.array(torch.unique(labels).cpu())
    label_indices = {}
    centroids = []
        
    for label in uniq_labels:
        label_indices[label] = torch.where(labels.flatten()==label)[0]#.cpu().numpy()
        label_embeddings = embeddings[label_indices[label]]
        centroids.append(torch.mean(label_embeddings, axis=0))
    
    centroids = torch.stack(centroids)
    
    dists = lmu.get_pairwise_mat(embeddings, 
                                     centroids,
                                     use_similarity=False, 
                                     squared=False)
    reg_vals_own_centroid = torch.empty((len(embeddings),))
    reg_vals_other_centroids = torch.empty((len(embeddings),))
    #print('Centroids', centroids)
    for i, label in enumerate(uniq_labels):
        label_embeddings = embeddings[label_indices[label]]
        #print('Label', label)
        #print(label_indices[label])
        dist_to_own_centroid = dists[label_indices[label], i]
        #print('To own centroid', dist_to_own_centroid)
        
        dist_other_centroids = dists[label_indices[label]][:, list(range(i)) + list(range(i+1, dists.shape[1]))]
#         print('To other centroids', dist_other_centroids)
        
        # penalty for large distance from own centroid
        
    
        own_centroid_reg = (dist_to_own_centroid - reg_own_threshold).clamp(0, 999)
        #print('Own centroid penalty', own_centroid_reg)
        
        
        other_centroid_reg, _ = (reg_other_threshold - dist_other_centroids).clamp(0, 999).max(dim=1)
        #print('Other centroid reg', other_centroid_reg)
        #print()
        reg_vals_own_centroid[label_indices[label]] = reg_own_weight*own_centroid_reg.cpu()
        reg_vals_other_centroids[label_indices[label]] = reg_other_weight*other_centroid_reg.cpu()
        #print(reg_vals)
    return reg_vals_own_centroid, reg_vals_other_centroids
    
class ContrastiveLossRegularized(ContrastiveLoss):
    def __init__(
        self,
        *args,
        reg_own_weight=0.1,
        reg_other_weight=0.1,
        reg_own_threshold=0.1,
        reg_other_threshold=0.2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reg_own_weight = reg_own_weight
        self.reg_other_weight = reg_other_weight
        self.reg_own_threshold = reg_own_threshold
        self.reg_other_threshold = reg_other_threshold
        
    def compute_loss(self, embeddings, labels, indices_tuple):
        loss_dict = super().compute_loss(embeddings, labels, indices_tuple)
        
        reg_own_centroid, reg_other_centroids = dist_from_centroid_regularizer(embeddings, labels, 
            reg_own_weight=self.reg_own_weight, 
            reg_other_weight=self.reg_other_weight,
            reg_own_threshold=self.reg_own_threshold,
            reg_other_threshold=self.reg_other_threshold,
            )
        reg_dict = {"losses": reg_own_centroid,
                    "indices": torch.tensor(list(range(len(embeddings)))),
                    "reduction_type": "element"}
        loss_dict['reg_own_centroid'] = reg_dict

        reg_dict = {"losses": reg_other_centroids,
                    "indices": torch.tensor(list(range(len(embeddings)))),
                    "reduction_type": "element"}
        loss_dict['reg_other_centroids'] = reg_dict
        return loss_dict
    
    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss", "reg_own_centroid", "reg_other_centroids"]

r = get_runner()
r.register("loss", ContrastiveLossRegularized)
r.run()