from base import get_runner # HAS TO BE FIRST OR SEGFAULT HAPPENS

from collections import OrderedDict
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

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from pytorch_metric_learning.losses.contrastive_loss import ContrastiveLoss
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.reducers import AvgNonZeroReducer

def global_centroid_regularizer(embeddings, 
                                labels, 
                                centroids, 
                                reg_own_weight=1, 
                                reg_own_threshold=0.1, 
                                reg_other_weight=1, 
                                reg_other_threshold=0.2,
                                reg_warmup=5,
                                reg_warmup_current=0):
    if not centroids:
        return torch.zeros((len(embeddings), )), torch.zeros((len(embeddings), ))
    uniq_labels = sorted(np.array(torch.unique(labels).cpu()))

    print('Unique labels according to batch', uniq_labels)

    label_to_centroid_idx = {label: idx for idx, label in enumerate(centroids.keys())}
    centroid_vectors = torch.stack(list(centroids.values()))
    
    print('labels to centroids', label_to_centroid_idx)

    """
    labels to centroids {25: 0, 26: 1, 27: 2, 28: 3, 29: 4, 30: 5, 31: 6, 32: 7, 33: 8, 34: 9, 35: 10, 36: 11, 37: 12, 38: 13, 39: 14, 40: 15, 41: 16, 42: 17, 43: 18, 44: 19, 45: 20, 46: 21, 47: 22, 48: 23, 49: 24, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 32, 58: 33, 59: 34, 60: 35, 61: 36, 62: 37, 63: 38, 64: 39, 65: 40, 66: 41, 67: 42, 68: 43, 69: 44, 70: 45, 71: 46, 72: 47, 73: 48, 74: 49, 75: 50, 76: 51, 77: 52, 78: 53, 79: 54, 80: 55, 81: 56, 82: 57, 83: 58, 84: 59, 85: 60, 86: 61, 87: 62, 88: 63, 89: 64, 90: 65, 91: 66, 92: 67, 93: 68, 94: 69, 95: 70, 96: 71, 97: 72, 98: 73, 99: 74}
    
    labels to centroids {25: 0, 26: 1, 27: 2, 28: 3, 29: 4, 30: 5, 31: 6, 32: 7, 33: 8, 34: 9, 35: 10, 36: 11, 37: 12, 38: 13, 39: 14, 40: 15, 41: 16, 42: 17, 43: 18, 44: 19, 45: 20, 46: 21, 47: 22, 48: 23, 49: 24, 50: 25, 51: 26, 52: 27, 53: 28, 54: 29, 55: 30, 56: 31, 57: 32, 58: 33, 59: 34, 60: 35, 61: 36, 62: 37, 63: 38, 64: 39, 65: 40, 66: 41, 67: 42, 68: 43, 69: 44, 70: 45, 71: 46, 72: 47, 73: 48, 74: 49, 75: 50, 76: 51, 77: 52, 78: 53, 79: 54, 80: 55, 81: 56, 82: 57, 83: 58, 84: 59, 85: 60, 86: 61, 87: 62, 88: 63, 89: 64, 90: 65, 91: 66, 92: 67, 93: 68, 94: 69, 95: 70, 96: 71, 97: 72, 98: 73, 99: 74}


    """

    if embeddings.get_device() > -1:
        centroid_vectors = centroid_vectors.to(embeddings.get_device())
    
    reg_vals_own_centroid = torch.empty((len(embeddings),))
    reg_vals_other_centroids = torch.empty((len(embeddings),))
    
    #print(centroid_vectors)
    dists = lmu.get_pairwise_mat(embeddings, 
                                 centroid_vectors,
                                 use_similarity=False, 
                                 squared=False)
    
    warmup_coef = np.clip(np.log(reg_warmup_current)/np.log(reg_warmup), 0, 1)

    print('warmup', warmup_coef)
    for i, label in enumerate(uniq_labels):
        #print(label)
        label_indices = torch.where(labels.flatten()==label)[0]
        centroid_idx = label_to_centroid_idx[int(label)]
        
        dist_to_own_centroid = dists[label_indices, centroid_idx]
        #print('To own centroid', dist_to_own_centroid)
        
        dist_other_centroids = dists[label_indices][:, list(range(centroid_idx)) + list(range(centroid_idx+1, dists.shape[1]))]
        #print('To other centroids', dist_other_centroids)
        
        # penalty for large distance from own centroid
        
    
        own_centroid_reg = (dist_to_own_centroid - reg_own_threshold).clamp(0, 999)
        #print('Own centroid penalty', own_centroid_reg)
        
        
        other_centroid_reg, _ = (reg_other_threshold - dist_other_centroids).clamp(0, 999).max(dim=1)
        #print('Other centroid reg', other_centroid_reg)
        #print()
        reg_vals_own_centroid[label_indices] = warmup_coef * reg_own_weight*own_centroid_reg.cpu()
        reg_vals_other_centroids[label_indices] = warmup_coef * reg_other_weight*other_centroid_reg.cpu()
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
        reg_warmup = 5,
        centroids = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reg_own_weight = reg_own_weight
        self.reg_other_weight = reg_other_weight
        self.reg_own_threshold = reg_own_threshold
        self.reg_other_threshold = reg_other_threshold
        self.reg_warmup = reg_warmup
        self._warmup = 0
        self.centroids = centroids

    def compute_loss(self, embeddings, labels, indices_tuple):
        loss_dict = super().compute_loss(embeddings, labels, indices_tuple)
        
        reg_own_centroid, reg_other_centroids = global_centroid_regularizer(embeddings, labels, 
            self.centroids,
            reg_own_weight=self.reg_own_weight, 
            reg_other_weight=self.reg_other_weight,
            reg_own_threshold=self.reg_own_threshold,
            reg_other_threshold=self.reg_other_threshold,
            reg_warmup_current=self._warmup,
            reg_warmup=self.reg_warmup,
                                                                            
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


class CustomHookFactory(HookFactory):
    def create_end_of_epoch_hook(self,
                                _,
                                hooks, 
                                split_manager, 
                                splits_to_eval, 
                                tester,
                                model_folder, 
                                save_interval, 
                                patience, 
                                collate_fn, 
                                eval_assertions):
        dataset_dict = split_manager.get_dataset_dict("eval", inclusion_list=splits_to_eval)
        helper_hook = hooks.end_of_epoch_hook(tester=tester,
                                            dataset_dict=dataset_dict,
                                            model_folder=model_folder,
                                            test_interval=save_interval,
                                            patience=patience,
                                            test_collate_fn=collate_fn)

        def get_centroids(dataset, models):
            embeddings, labels = tester.get_all_embeddings(dataset, models['trunk'], models['embedder'], collate_fn)
            print(labels[:10])
            uniq_labels = sorted(np.unique(labels.flatten()))
            print('Unique labels according to hook factory', uniq_labels)
            centroids = {}            
            for label in uniq_labels:
                label_indices = np.where(labels.flatten() == label)[0]
                label_embeddings = embeddings[label_indices, :]
                centroids[int(label)] = torch.Tensor(np.mean(label_embeddings, axis=0))
            return centroids
        
        def end_of_epoch_hook(trainer):
            models = trainer.models
            train_dataset = split_manager.get_dataset("eval", "train")

            train_label_set = split_manager.get_label_set("eval", "train")
            print('Train label set', train_label_set)
            print('Test label set', split_manager.get_label_set("eval", "test"))
            print('Val label set', split_manager.get_label_set("eval", "val"))
            centroids = OrderedDict(get_centroids(train_dataset, models))
            print('Obtained centroids')
            trainer.loss_funcs['metric_loss'].centroids = centroids
            trainer.loss_funcs['metric_loss']._warmup = trainer.epoch

            return helper_hook(trainer)

        return end_of_epoch_hook

r = get_runner()
r.register("factory", CustomHookFactory)
r.register("loss", ContrastiveLossRegularized)
r.run()