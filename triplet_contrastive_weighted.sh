rm -r /data/thesis/benchmarks/experiments/triplet_contrastive_weighted

python3 runners/triplet_contrastive_weighted.py --experiment_name triplet_contrastive_weighted \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {TripletContrastiveWeightedLoss: {\
triplet_margin~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1], \
alpha~BAYESIAN~: [0, 1]}}} 