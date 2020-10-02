#rm -r $PWD/experiments/weighted_sampler

python3 runners/weighted_sampler.py --experiment_name weighted_sampler_reproduce \
--bayes_opt_iters 0 \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--sampler~OVERRIDE~ {ClassWeightedSampler: {m: 6, mode: 'scores'}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin: 0.28, \
neg_margin: 0.376}}} 

#./cleanup.sh