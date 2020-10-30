python3 runners/global_embedding_regularizer.py --experiment_name hotels_global_embedding_regularizer \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'hotels', root: $PWD/hotels50k}} \
--patience 4 \
--trainer~APPLY~2 {set_min_label_to_zero: False, dataloader_num_workers: 6, iterations_per_epoch: 500} \
--tester~APPLY~2 {dataloader_num_workers: 6} \
--split_manager~SWAP~1 {ClosedSetSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--bayes_opt_iters 7 \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_own_weight~BAYESIAN~: [0, 1], reg_own_threshold~BAYESIAN~: [0, 1], \
reg_other_weight~BAYESIAN~: [0, 1], reg_other_threshold~BAYESIAN~: [0, 1], \
delay_epochs~INT_BAYESIAN~: [0, 5], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 