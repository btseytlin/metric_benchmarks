from base import get_runner
from pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
from pytorch_metric_learning.reducers import MeanReducer

class TripletLossMeanReducer(TripletMarginLoss):
    def get_default_reducer(self):
        return MeanReducer()

r = get_runner()

r.register("loss", TripletLossMeanReducer)

r.run()