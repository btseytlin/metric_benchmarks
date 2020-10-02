#rm -r $PWD/experiments/class_dropout_sampler

python3 runners/class_dropout_sampler.py --experiment_name class_dropout_sampler \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 0 \
--sampler~OVERRIDE~ {ClassDropoutSampler: {m: 4, d: 0.244}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin: 0.3955, \
neg_margin: 0.69}}} 

#./cleanup.sh