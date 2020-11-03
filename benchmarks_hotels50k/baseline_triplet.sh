python3 runners/baseline_triplet.py --experiment_name baseline_triplet \
--dataset~OVERRIDE~ {Hotels50kDataset: {download: False, target: 'chains', root: $PWD/hotels50k}} \
--patience 4 \
--split_manager~SWAP~1 {ClosedSetSplitManager: {}} \
--split_manager~APPLY~2 {helper_split_manager: {UseOriginalTestSplitManager: {}}} \
--trainer~APPLY~2 {dataloader_num_workers: 6, iterations_per_epoch: 500} \
--tester~APPLY~2 {dataloader_num_workers: 6} \
--bayes_opt_iters 7 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {TripletLossMeanReducer: {\
margin~BAYESIAN~: [0, 1], \
}}} \
--mining_funcs~OVERRIDE~ \
{tuple_miner: {BatchHardMiner: {}}} \