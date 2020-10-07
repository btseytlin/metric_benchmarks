from base import get_runner

import torch.nn.functional as F
import torch

from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer

# Warning: works only with distances, not similarities
class ContrastiveTripletLoss(ContrastiveLoss):
    def __init__(
        self,
        *args,
        triplet_margin=0.1,
        triplet_weight=0.5,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.triplet_margin = triplet_margin
        self.triplet_weight = triplet_weight
        
    def pair_based_loss(self, mat, labels, indices_tuple):
        #print('indices_tuple', indices_tuple)
        
        a1, p, a2, n = indices_tuple
        pos_pair, neg_pair = [], []
        if len(a1) > 0:
            pos_pair = mat[a1, p]
        if len(a2) > 0:
            neg_pair = mat[a2, n]
        loss_dict = self._compute_loss(pos_pair, neg_pair, indices_tuple)
        #print('contrastive losses', loss_dict)
        
        triplet_indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor='all')
        #print('triplet_indices_tuple', triplet_indices_tuple)
        anchor_idx, positive_idx, negative_idx = triplet_indices_tuple
        a_p_dist = mat[anchor_idx, positive_idx] ** self.power
        a_n_dist = mat[anchor_idx, negative_idx] ** self.power
        
        triplet_loss = self.triplet_weight * F.relu(a_p_dist - a_n_dist + self.triplet_margin)
        #print('triplet_loss', triplet_loss)
        loss_dict['triplet_loss'] = {"losses": triplet_loss, "indices": triplet_indices_tuple, "reduction_type": "triplet"}
        return loss_dict
    
    def get_default_reducer(self):
        return AvgNonZeroReducer()

    def sub_loss_names(self):
        return ["pos_loss", "neg_loss", "triplet_loss"]
    

r = get_runner()

r.register("loss", ContrastiveTripletLoss)

r.run()