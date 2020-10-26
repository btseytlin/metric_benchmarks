python3 runners/basic.py --experiment_name contrastive_hotels \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'chains'}} \
--split_manager~SWAP~1 {IndexSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--trainer~APPLY~2 {dataloader_num_workers: 6, iterations_per_epoch: 500} \
--bayes_opt_iters 7 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 
