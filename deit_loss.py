from nntools.nnet import register_loss, SUPPORTED_LOSS
from nntools.utils.misc import call_with_filtered_kwargs
import torch.nn as nn


class DistillationLossViT(nn.Module):
    def __init__(self, loss_type='CrossEntropy', ignore_index=None, weights=-100, smooth_factor=0.1, **kwargs):
        super(DistillationLossViT, self).__init__()

        loss_func = SUPPORTED_LOSS[loss_type]

        kwargs = dict(ignore_index=ignore_index, weights=weights, smooth_factor=smooth_factor, **kwargs)
        self.loss = call_with_filtered_kwargs(loss_func, kwargs)

    def forward(self, token_class, token_dist, y, y_teacher):
        return 0.5*self.loss(token_class, y)+0.5*self.loss(token_dist, y_teacher)


register_loss('DistillationLossViT', DistillationLossViT)
