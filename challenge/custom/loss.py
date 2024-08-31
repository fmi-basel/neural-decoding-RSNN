import torch
import torch.nn as nn

from stork.loss_stacks import LossStack

class CSTLossStack(LossStack):
    def __init__(self, 
                 mask=None, 
                 density_weighting_func=False):
        super().__init__()
        self.mask = mask
        self.loss_fn = None  # to be defined in the child class
        self.density_weighting_func = density_weighting_func

    def get_R2(self, pred, target):
        # Julian Rossbroich
        # modified july 2024
        """
        Args:
            pred: Predicted series of the model (batch_size * timestep * nb_outputs),
            target: Ground truth series (batch_size * timestep * nb_outputs).

        Return:
            r2: R-squared between the inputs along consecutive axis, over a batch.
        """

        # For each feature, calculate R2
        # We use the mean across all samples to calculate sst
        ssr = torch.sum((target - pred) ** 2, dim=(0, 1))
        sst = torch.sum((target - torch.mean(target, dim=(0, 1))) ** 2, dim=(0, 1))
        r2 = (1 - ssr / sst).detach().cpu().numpy()

        return [float(r2[0].round(3)), float(r2[1].round(3)), float(r2.mean().round(3))]

    def get_metric_names(self):
        # Julian Rossbroich
        # modified july 2024
        return ["r2x", "r2y", "r2"]

    def compute_loss(self, output, target):
        """Computes MSQE loss between output and target."""

        if self.mask is not None:
            output = output * self.mask.expand_as(output)
            target = target * self.mask.expand_as(output)
            
        if self.density_weighting_func:
            weight = self.density_weighting_func(target)
        else:
            weight = None

        self.metrics = self.get_R2(output, target)
        return self.loss_fn(output, target, weight=weight)

    def predict(self, output):
        return output

    def __call__(self, output, targets):
        return self.compute_loss(output, targets)


class MeanSquareError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MSEloss
        
    def _weighted_MSEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * (output - target) ** 2)
        else:
            return torch.mean((output - target) ** 2)


class RootMeanSquareError(CSTLossStack):

    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_RMSEloss

    def _weighted_RMSEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.sqrt(torch.mean(weight * (output - target) ** 2))
        else:
            return torch.sqrt(torch.mean((output - target) ** 2))


class MeanAbsoluteError(CSTLossStack):
    def __init__(self, mask=None, density_weighting_func=False):
        super().__init__(mask=mask, density_weighting_func=density_weighting_func)
        self.loss_fn = self._weighted_MAEloss

    def _weighted_MAEloss(self, output, target, weight=None):
        if weight is not None:
            return torch.mean(weight * torch.abs(output - target))
        else:
            return torch.mean(torch.abs(output - target))


class HuberLoss(CSTLossStack):
    def __init__(self, delta=1.0, mask=None, density_weighting_func=False):
        
        if density_weighting_func:
            raise ValueError("Density weighting not supported for Huber loss.")
        
        super().__init__(mask=mask)
        self.loss_fn = nn.SmoothL1Loss(beta=delta)
        self.delta = delta
