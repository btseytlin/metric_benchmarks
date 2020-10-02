#rm -r $PWD/experiments/embedding_regularizer

python3 runners/embedding_regularizer.py --experiment_name embedding_regularizer \
--dataset~OVERRIDE~ {CUB200: {download: True}} \
--bayes_opt_iters 25 \
--loss_funcs~OVERRIDE~ \
{metric_loss: {ContrastiveLoss: {\
reducer: {RegularizerReducer: {weight~BAYESIAN~: [0, 1], threshold~BAYESIAN~: [0, 1]} },
pos_margin~BAYESIAN~: [0, 1], \
neg_margin~BAYESIAN~: [0, 1]}}} 

#./cleanup.sh