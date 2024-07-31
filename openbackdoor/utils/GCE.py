from torch import nn
import torch
import torch.nn.functional as F
class GCELoss(nn.Module):
    def __init__(self, q=0.5, eps=1e-7, reduction = 'mean'):
        super(GCELoss, self).__init__()
        self.q = q
        self.eps = eps
        self.reduction = reduction

    def forward(self, input, target):
        # Compute probabilities
        input_soft = F.softmax(input, dim=1)
        # Apply GCE formula
        loss = (1 - torch.pow(input_soft, self.q)) / self.q
        # Mask out loss according to target
        loss = loss.gather(1, target.view(-1, 1))   # gather
        # Take mean of masked loss
        loss = loss.mean() if self.reduction == 'mean' else loss
        return loss