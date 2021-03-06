#rm -r $PWD/experiments/weighted_sampler

python3 runners/weighted_sampler.py --experiment_name weighted_sampler \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--sampler~OVERRIDE~ {ClassWeightedSampler: {m: 6, mode: 'scores'}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 

#./cleanup.sh