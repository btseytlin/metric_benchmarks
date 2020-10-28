python3 runners/batch_regularizer.py --experiment_name hotels_batch_regularizer_reproduce \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'chains', root: $PWD/hotels50k}} \
--trainer~APPLY~2 {dataloader_num_workers: 6, iterations_per_epoch: 500} \
--tester~APPLY~2 {dataloader_num_workers: 6} \
--bayes_opt_iters 0 \
--split_manager~SWAP~1 {IndexSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_own_weight: 0.197, reg_own_threshold: 0.066, \
reg_other_weight: 0.772, reg_other_threshold: 0.436, \
pos_margin: 0.23, \
neg_margin: 0.917}}} 
