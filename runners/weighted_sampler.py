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

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def normalize(v):
    return v / np.sum(v)

class ClassWeightedSampler(Sampler):
    """
    At every iteration, this will return m samples per class, weighted by the weights array.
    """
    def __init__(self, labels, m, mode='scores', min_per_class=2, max_per_class=6, reweight_interval = 2, weights=None, length_before_new_iter=100000):
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()
        self.m = int(m)
        self.min_per_class = int(min_per_class) 
        self.max_per_class = int(max_per_class) 
        self.reweight_interval = int(reweight_interval)
        
        assert mode in ('scores', 'errors')
        self.mode = mode

        self.labels_to_indices = c_f.get_labels_to_indices(labels)
        self.labels = list(self.labels_to_indices.keys())
        if weights is None:
            weights = [1/len(self.labels)]*len(self.labels)
        
        self.weights = np.array(weights)
        
        self.label_to_weight_idx = {label: i for label, i in zip(self.labels, range(len(weights)))}
        
        assert len(self.labels) == len(self.weights)
        
        self.length_of_single_pass = self.m*len(self.labels)
        self.list_size = length_before_new_iter
        if self.length_of_single_pass < self.list_size:
            self.list_size -= (self.list_size) % (self.length_of_single_pass)
    
    def update_with_scores(self, scores):
        if self.mode == 'scores':
            self.update_weights(scores)
        elif self.mode == 'errors':
            self.update_weights(1 - scores)

    def update_weights(self, weights):
        self.weights = normalize(softmax(weights))

    @property
    def m_per_class(self):
        return np.clip((self.length_of_single_pass*self.weights).astype(int), a_min=self.min_per_class, a_max=self.max_per_class)
    
    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = []
        i = 0
        num_iters = self.list_size // self.length_of_single_pass if self.length_of_single_pass < self.list_size else 1
        self.shuffled_labels = list(self.labels)
        for _ in range(num_iters):
            c_f.NUMPY_RANDOM.shuffle(self.shuffled_labels)
            for label in self.shuffled_labels:
                t = self.labels_to_indices[label]
                label_idx = self.label_to_weight_idx[label]
                m_per_label = self.m_per_class[label_idx]
                idx_list += list(c_f.safe_random_choice(t, size=m_per_label))
                i += m_per_label
        return iter(idx_list)


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

        def get_knn_f1_score(embeddings_and_labels, sampler):
            query_embeddings, query_labels, reference_embeddings, reference_labels = tester.set_reference_and_query(embeddings_and_labels, 'train')
        
            for L in tester.label_levels:
                curr_query_labels = query_labels[:, L]
                curr_reference_labels = reference_labels[:, L]
                label_counts, num_k = get_label_counts(curr_reference_labels)
                embeddings_come_from_same_source =  tester.embeddings_come_from_same_source(embeddings_and_labels)
                knn_indices, knn_distances = stat_utils.get_knn(reference_embeddings, query_embeddings, num_k, embeddings_come_from_same_source)
                knn_labels = curr_reference_labels[knn_indices]

                lone_query_labels = get_lone_query_labels(curr_query_labels, curr_reference_labels, label_counts, embeddings_come_from_same_source)
                not_lone_query_mask = ~np.isin(curr_query_labels, lone_query_labels)
                if not any(not_lone_query_mask):
                    print("Warning: None of the query labels are in the reference set and I barely know what that means.")

                f1_scores = f1_score(curr_reference_labels, knn_labels[:, :1].flatten(), labels=sampler.labels, average=None)
                return f1_scores


        def update_sampler_class_weights(ds_dict, models, sampler):
            models['trunk'].eval()
            models['embedder'].eval()
            embeddings_and_labels = tester.get_all_embeddings_for_all_splits(ds_dict, 
                                                                                models['trunk'], 
                                                                                models['embedder'], 
                                                                                ['train'], 
                                                                                collate_fn)
            f1_scores = get_knn_f1_score(embeddings_and_labels, sampler)

            sampler.update_with_scores(f1_scores)


        def end_of_epoch_hook(trainer):
            torch.cuda.empty_cache()
            eval_assertions(dataset_dict)

            if trainer.epoch % trainer.sampler.reweight_interval == 0: 
                models = trainer.models
                ds_dict = split_manager.get_dataset_dict("train")
                update_sampler_class_weights(ds_dict, models, trainer.sampler)

            return helper_hook(trainer)

        return end_of_epoch_hook

r = get_runner()
r.register("factory", CustomHookFactory)
r.register("sampler", ClassWeightedSampler)
r.run()