import numpy as np
import time

import torch
import torch.nn as nn

from stork.models import RecurrentSpikingModel

class CustomRecurrentSpikingModel(RecurrentSpikingModel):
    
    """
    Custom model wrapper that implements an additional training loop with 
    a custom mask for weight parameters used for an iterative pruning
    strategy.
    """
    
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
    def train_epoch_masked(self, dataset, shuffle=True, mask=None):
        self.train(True)
        self.prepare_data(dataset)
        metrics = []
        for local_X, local_y in self.data_generator(dataset, shuffle=shuffle):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)

            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )

            # Use autograd to compute the backward pass.
            self.optimizer_instance.zero_grad()
            total_loss.backward()

            if mask is not None:
                for name, param in self.named_parameters():
                    if 'weight' in name:
                        param.grad.data.mul_(mask[name])

            self.optimizer_instance.step()
            self.apply_constraints()
            
        if self.scheduler_instance is not None:
            self.scheduler_instance.step()

        return np.mean(np.array(metrics), axis=0)
    
    
    def fit_validate_masked(
        self, dataset, valid_dataset, nb_epochs=10, verbose=True, wandb=None, mask=None
    ):
        self.hist_train = []
        self.hist_valid = []
        self.wall_clock_time = []
        for ep in range(nb_epochs):
            t_start = time.time()
            self.train()
            ret_train = self.train_epoch_masked(dataset, mask=mask)

            self.train(False)
            ret_valid = self.evaluate(valid_dataset)
            self.hist_train.append(ret_train)
            self.hist_valid.append(ret_valid)

            if self.wandb is not None:
                self.wandb.log(
                    {
                        key: value
                        for (key, value) in zip(
                            self.get_metric_names()
                            + self.get_metric_names(prefix="val_"),
                            ret_train.tolist() + ret_valid.tolist(),
                        )
                    }
                )

            if verbose:
                t_iter = time.time() - t_start
                self.wall_clock_time.append(t_iter)
                print(
                    "%02i %s --%s t_iter=%.2f"
                    % (
                        ep,
                        self.get_metrics_string(ret_train),
                        self.get_metrics_string(ret_valid, prefix="val_"),
                        t_iter,
                    )
                )

        self.hist = np.concatenate(
            (np.array(self.hist_train), np.array(self.hist_valid))
        )
        self.fit_runs.append(self.hist)
        dict1 = self.get_metrics_history_dict(np.array(self.hist_train), prefix="")
        dict2 = self.get_metrics_history_dict(np.array(self.hist_valid), prefix="val_")
        history = {**dict1, **dict2}
        return history

    def evaluate_continuos_testdata(self, test_dataset, train_mode=False):
        """
        Evaluates on test data that is one long sample.
        Also returns predictions and ground truth
        """
        self.train(train_mode)
        self.prepare_data(test_dataset)
        metrics = []
        loss_min_batch = np.inf
        for local_X, local_y in self.data_generator(test_dataset, shuffle=False):
            output = self.forward_pass(local_X, cur_batch_size=len(local_X))
            total_loss = self.get_total_loss(output, local_y)
            
            # store loss and other metrics
            metrics.append(
                [self.out_loss.item(), self.reg_loss.item()] + self.loss_stack.metrics
            )
            if self.out_loss < loss_min_batch:
                loss_min_batch = self.out_loss
                pre_batch_best = (
                    output.cpu().detach().numpy()[0, :, :]
                )  # ltj, Choose the first sample in the batch with minimal loss as an example
                gt_batch_best = (
                    local_y.cpu().detach().numpy()[0, :, :]
                )  # ltj, Choose the first sample in the batch with minimal loss as an example
        
        return np.mean(np.array(metrics), axis=0) , pre_batch_best, gt_batch_best  # ltj

