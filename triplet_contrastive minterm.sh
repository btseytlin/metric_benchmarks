rm -r /data/thesis/benchmarks/experiments/triplet_contrastive_minterm

python3 runners/triplet_contrastive_minterm.py --experiment_name triplet_contrastive_minterm \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {TripletContrastiveLossMinterm: {\
triplet_margin~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 