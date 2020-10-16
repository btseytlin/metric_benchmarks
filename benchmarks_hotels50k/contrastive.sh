#rm -r $PWD/experiments/contrastive

python3 runners/basic.py --experiment_name contrastive_hotels \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, root: $PWD/hotels50k}} \
--trainer~APPLY~2 {dataloader_num_workers: 10, iterations_per_epoch: 1000} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} \
--split_manager~SWAP~1 {IndexSplitManager: {data_and_label_getter_keys: None}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}}
