rm -r /data/thesis/benchmarks/experiments/weighted_sampler

python3 runners/weighted_sampler.py --experiment_name weighted_sampler \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--factories {hook~OVERRIDE~: {CustomHookFactory: {}}} \
--sampler~OVERRIDE~ {ClassWeightedSampler: {m: 4}}