#rm -r $PWD/experiments/contrastive

python3 runners/basic.py --experiment_name contrastive_hotels \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, root: /data/thesis/Hotels-50K}} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} \
--split_manager~SWAP~1 {IndexSplitManager: {data_and_label_getter_keys: None}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}}
