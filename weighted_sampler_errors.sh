rm -r /data/thesis/benchmarks/experiments/weighted_sampler_errors

python3 runners/weighted_sampler.py --experiment_name weighted_sampler_errors \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--sampler~OVERRIDE~ {ClassWeightedSampler: {m: 4, mode: 'errors', min_per_class~INT_BAYESIAN~: [1, 3], max_per_class~INT_BAYESIAN~: [4, 6]}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 

./cleanup.sh