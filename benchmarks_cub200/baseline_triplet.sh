python3 runners/baseline_triplet.py --experiment_name baseline_triplet \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--trainer~APPLY~2 {dataloader_num_workers: 6} \
--tester~APPLY~2 {dataloader_num_workers: 6} \
--loss_funcs~OVERRIDE~ \
{metric_loss: {TripletLossMeanReducer: {\
margin~BAYESIAN~: [0, 1], \
}}} \
--mining_funcs~OVERRIDE~ \
{tuple_miner: {BatchHardMiner: {}}} \