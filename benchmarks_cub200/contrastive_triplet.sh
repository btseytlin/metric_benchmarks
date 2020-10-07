python3 runners/contrastive_triplet.py --experiment_name contrastive_triplet \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveTripletLoss: {\
triplet_margin~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1], \
triplet_mweight~BAYESIAN~: [0, 1]}}} 
