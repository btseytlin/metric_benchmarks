rm -r /data/thesis/benchmarks/experiments/triplet_contrastive

python3 runners/triplet_contrastive.py --experiment_name triplet_contrastive \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {TripletContrastiveLoss: {\
triplet_margin~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 