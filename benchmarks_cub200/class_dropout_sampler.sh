python3 runners/class_dropout_sampler.py --experiment_name class_dropout_sampler \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--sampler~OVERRIDE~ {ClassDropoutSampler: {m: 4, d~BAYESIAN~: [0, 1]}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 