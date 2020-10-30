python3 runners/batch_regularizer.py --experiment_name hotels_batch_regularizer \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'hotels', root: $PWD/hotels50k}} \
--trainer~APPLY~2 {dataloader_num_workers: 6, iterations_per_epoch: 500} \
--patience 4 \
--bayes_opt_iters 7 \
--split_manager~SWAP~1 {ClosedSetSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_own_weight~BAYESIAN~: [0, 1], reg_own_threshold~BAYESIAN~: [0, 1], \
reg_other_weight~BAYESIAN~: [0, 1], reg_other_threshold~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 
