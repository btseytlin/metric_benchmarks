python3 runners/global_embedding_regularizer.py --experiment_name global_embedding_regularizer \
--trainer~APPLY~2 {set_min_label_to_zero: False, dataloader_num_workers: 6} \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_own_weight~BAYESIAN~: [0, 1], reg_own_threshold~BAYESIAN~: [0, 1], \
reg_other_weight~BAYESIAN~: [0, 1], reg_other_threshold~BAYESIAN~: [0, 1], \
delay_epochs~INT_BAYESIAN~: [0, 5], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 