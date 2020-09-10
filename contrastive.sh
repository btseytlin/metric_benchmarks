rm -r /data/thesis/benchmarks/experiments/contrastive

python3 runners/basic.py --experiment_name contrastive \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 