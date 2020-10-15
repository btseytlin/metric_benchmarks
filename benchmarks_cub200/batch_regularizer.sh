python3 runners/batch_regularizer.py --experiment_name batch_regularizer \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_own_weight~BAYESIAN~: [0, 1], reg_own_threshold~BAYESIAN~: [0, 1], \
reg_other_weight~BAYESIAN~: [0, 1], reg_other_threshold~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 