#rm -r $PWD/experiments/contrastive

python3 runners/basic.py --experiment_name contrastive_hotels \
--dataset~OVERRIDE~ {Hotels50KDataset: {download: False, root: hotels50k}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} \
--split_manager~SWAP~1 {IndexSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}}
