#rm -r $PWD/experiments/embedding_regularizer_dist_from_center

python3 runners/embedding_regularizer_dist_from_center.py --experiment_name embedding_regularizer_dist_from_center \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLossRegularized: {\
reg_weight~BAYESIAN~: [0, 1], reg_threshold~BAYESIAN~: [0, 1], \
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 

#./cleanup.sh