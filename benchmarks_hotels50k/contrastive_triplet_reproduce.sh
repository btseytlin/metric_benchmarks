python3 runners/contrastive_triplet.py --experiment_name hotels_contrastive_triplet \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'chains', root: $PWD/hotels50k}} \
--split_manager~SWAP~1 {ClosedSetSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--trainer~APPLY~2 {dataloader_num_workers: 6, iterations_per_epoch: 500} \
--tester~APPLY~2 {dataloader_num_workers: 6} \
--bayes_opt_iters 0 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveTripletLoss: {\
triplet_margin: 0.134, \
pos_margin: 0.409, \
neg_margin: 0.134, \
triplet_weight: 0.736}}} 
