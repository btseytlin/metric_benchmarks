from base import get_runner

from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer
import torch.nn.functional as F
import torch

from pytorch_metric_learning.losses.generic_pair_loss import GenericPairLoss
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer

# Warning: works only with distances, not similarities
class TripletContrastiveWeightedLoss(BaseMetricLossFunction):
    def __init__(
        self,
        pos_margin=0.5,
        neg_margin=0.5,
        triplet_margin=0.1,
        alpha=0.1,
        triplets_per_anchor="all",
        distance_norm=2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.triplet_margin = triplet_margin
        self.alpha = alpha
        self.triplets_per_anchor = triplets_per_anchor
        self.distance_norm = distance_norm
        
        self.add_to_recordable_attributes(list_of_names=["num_non_zero_pos_pairs", "num_non_zero_neg_pairs"])
        self.add_to_recordable_attributes(name="num_non_zero_triplets_triplet_loss_only")
        self.add_to_recordable_attributes(name="num_non_zero_triplets")
    
    def compute_loss(self, embeddings, labels, indices_tuple):
        self.num_non_zero_pos_pairs, self.num_non_zero_neg_pairs = 0, 0
        self.num_non_zero_triplets_triplet_loss_only = 0
        self.num_non_zero_triplets = 0
            
        indices_tuple = lmu.convert_to_triplets(indices_tuple, labels, t_per_anchor=self.triplets_per_anchor)
        
        anchor_idx, positive_idx, negative_idx = indices_tuple
        
        #print('Anchors, positives, negatives', anchor_idx, positive_idx, negative_idx)
        if len(anchor_idx) == 0:
            return self.zero_losses()
        
        anchors, positives, negatives = embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
        
        mat = lmu.get_pairwise_mat(embeddings, embeddings, use_similarity=False, squared=False)
        
        #print(mat)
        a_p_dist = mat[anchor_idx, positive_idx]
        a_n_dist = mat[anchor_idx, negative_idx]
        p_n_dist = mat[positive_idx, negative_idx]
        #print('An dist by mat', a_n_dist)
        
        #print('AP', a_p_dist, 'AN', a_n_dist, 'PN', p_n_dist)
        dist = a_p_dist - a_n_dist
        
        # Compute triplet loss
        triplet_loss = F.relu(dist + self.triplet_margin)
        self.num_non_zero_triplets_triplet_loss_only = (triplet_loss > 0).nonzero().size(0)
        #print('Triplet loss', triplet_loss)
        
        # Compute pos contrastive loss
        contrastive_pos = F.relu(a_p_dist - self.pos_margin)
        self.num_non_zero_pos_pairs = (contrastive_pos > 0).nonzero().size(0)
        #print('Contrastive pos', contrastive_pos)
        
        # Compute neg contrastive loss
        contrastive_neg = F.relu(self.neg_margin - a_n_dist)
        self.num_non_zero_neg_pairs = (contrastive_neg > 0).nonzero().size(0)
        #print('Contrastive neg', contrastive_neg)
        
        full_loss = self.alpha*triplet_loss + (contrastive_pos + contrastive_neg) 
        self.num_non_zero_triplets = (full_loss > 0).nonzero().size(0)
        
        #print(full_loss)
        loss_dict = {"loss": {"losses": full_loss, "indices": anchor_idx, "reduction_type": "element"}}
        return loss_dict
        
    def get_default_reducer(self):
        return AvgNonZeroReducer()
    

r = get_runner()

r.register("loss", TripletContrastiveWeightedLoss)

r.run()