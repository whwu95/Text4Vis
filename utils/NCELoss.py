import torch.nn.functional as F
import torch.nn as nn

class NCELoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using NCE Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label):
        batch_size = len(prediction)
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss


class DualLoss(nn.Module):
    def __init__(self, error_metric=nn.KLDivLoss(reduction='mean')):
        super().__init__()
        print('=========using DS Loss==========')
        self.error_metric = error_metric

    def forward(self, prediction, label, temp=1000):
        batch_size = len(prediction)
        prediction = prediction * F.softmax(prediction/temp, dim=0) * batch_size
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss