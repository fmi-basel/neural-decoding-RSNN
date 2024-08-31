from stork.nodes.base import CellGroup

import torch
from torch.nn import Parameter

import numpy as np


class AverageReadouts(CellGroup):
    """Average over different readout groups"""

    def __init__(self, parent_groups):
        self.parent_groups = parent_groups
        super(AverageReadouts, self).__init__(parent_groups[0].shape)

    def forward(self):
        x = []
        for pg in self.parent_groups:
            x.append(pg.out)

        x = torch.stack(x)
        x = torch.mean(x, dim=0)

        self.out = x
        

class CustomReadoutGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        weight_scale=1.0,
        initial_state=-1e-3,
        stateful=False,
        learn_timescales=False,
        het_timescales=False,
        mem_het_timescales=False,
        syn_het_timescales=False,
        TC_mem_het_init="Constant",
        TC_syn_het_init="Constant",
        memsyn_bandpass_high_ratio_cut=2,
        memsyn_het_forward_method="highpass",
        is_delta_syn=False,
    ):  # ltj
        super().__init__(shape, stateful=stateful, name="Readout")
        self.tau_mem = tau_mem
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.het_timescales = het_timescales  # ltj
        self.mem_het_timescales = mem_het_timescales  # ltj
        self.syn_het_timescales = syn_het_timescales  # ltj
        self.learn_timescales = learn_timescales  # ltj
        self.is_delta_syn = is_delta_syn  # ltj
        self.TC_mem_het_init = TC_mem_het_init
        self.TC_syn_het_init = TC_syn_het_init
        self.memsyn_bandpass_high_ratio_cut = memsyn_bandpass_high_ratio_cut
        self.memsyn_het_forward_method = memsyn_het_forward_method
        self.syn = None
        self.tau_syn = tau_syn
        # size_tc = self.shape[0] if self.het_timescales else 1 #ltj
        # self.mem_param_readout = torch.rand(size_tc) #ltj
        # self.syn_param_readout = torch.rand(size_tc) #ltj

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        if self.het_timescales:
            self.mem_het_timescales = True
            self.syn_het_timescales = True
        if not self.is_delta_syn:
            self.dcy_syn = float(np.exp(-time_step / self.tau_syn))
            self.scl_syn = (1.0 - self.dcy_syn) * self.weight_scale
        if self.learn_timescales:  # ltj
            size_tc_mem = self.shape[0] if self.mem_het_timescales else 1  # ltj
            size_tc_syn = self.shape[0] if self.syn_het_timescales else 1  # ltj
            if self.TC_mem_het_init == "Uniform":
                mem_param_readout = torch.rand(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True
                )  # ltj
            elif self.TC_mem_het_init == "Gaussian":
                mem_param_readout = torch.randn(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True
                )  # ltj
            elif self.TC_mem_het_init == "Constant":
                mem_param_readout = torch.ones(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True
                )  # ltj
            # elif self.TC_mem_het_init == 'XavierUniform':
            #     mem_param_readout = torch.empty(size_tc_mem, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_uniform_(mem_param_readout)  # Xavier uniform initialization
            # elif self.TC_mem_het_init == 'XavierGassian':
            #     mem_param_readout = torch.empty(size_tc_mem, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_normal_(mem_param_readout)  # Xavier gaussian initialization
            elif self.TC_mem_het_init == "logNormal":
                mem_param_readout = torch.empty(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True
                )
                mean = -4.3  # Example mean of the underlying normal distribution
                std = 2.5  # Example standard deviation of the underlying normal distribution
                torch.nn.init.normal_(mem_param_readout, mean=mean, std=std)
                mem_param_readout = (
                    mem_param_readout.exp()
                )  # Converting normal distribution to log-normal
            elif self.TC_mem_het_init == "logspace":
                mem_param_readout = np.logspace(
                    np.log10(1), np.log10(10), num=size_tc_mem
                )
                mem_param_readout = torch.tensor(
                    mem_param_readout, device=device, dtype=dtype
                )  # ltj
            self.mem_param_readout = Parameter(
                mem_param_readout, requires_grad=self.learn_timescales
            )
            if not self.is_delta_syn:
                if self.TC_syn_het_init == "Uniform":
                    syn_param_readout = torch.rand(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True
                    )  # ltj
                elif self.TC_syn_het_init == "Gaussian":
                    syn_param_readout = torch.randn(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True
                    )  # ltj
                elif self.TC_syn_het_init == "Constant":
                    syn_param_readout = torch.ones(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True
                    )  # ltj
                # elif self.TC_syn_het_init == 'XavierUniform':
                #     syn_param_readout = torch.empty(size_tc_syn, device=device, dtype=dtype, requires_grad=True)
                #     torch.nn.init.xavier_uniform_(syn_param_readout)  # Xavier uniform initialization
                # elif self.TC_syn_het_init == 'XavierGassian':
                #     syn_param_readout = torch.empty(size_tc_syn, device=device, dtype=dtype, requires_grad=True)
                # torch.nn.init.xavier_normal_(syn_param_readout)  # Xavier gaussian initialization
                elif self.TC_syn_het_init == "logNormal":
                    syn_param_readout = torch.empty(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True
                    )
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5  # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(syn_param_readout, mean=mean, std=std)
                    syn_param_readout = (
                        syn_param_readout.exp()
                    )  # Converting normal distribution to log-normal
                elif self.TC_syn_het_init == "logspace":
                    syn_param_readout = np.logspace(
                        np.log10(1), np.log10(10), num=size_tc_syn
                    )
                    syn_param_readout = torch.tensor(
                        syn_param_readout, device=device, dtype=dtype
                    )  # ltj
                self.syn_param_readout = Parameter(
                    syn_param_readout, requires_grad=self.learn_timescales
                )
        elif self.mem_het_timescales or self.syn_het_timescales:
            if self.mem_het_timescales:
                size_tc = self.shape[0]
                if self.TC_mem_het_init == "Uniform":
                    mem_param_readout = torch.rand(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_mem_het_init == "Gaussian":
                    mem_param_readout = torch.randn(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_mem_het_init == "Constant":
                    mem_param_readout = torch.ones(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_mem_het_init == "logNormal":
                    mem_param_readout = torch.empty(size_tc, device=device, dtype=dtype)
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5  # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(mem_param_readout, mean=mean, std=std)
                    mem_param_readout = (
                        mem_param_readout.exp()
                    )  # Converting normal distribution to log-normal
                elif self.TC_mem_het_init == "logspace":
                    mem_param_readout = np.logspace(
                        np.log10(1), np.log10(10), num=size_tc_mem
                    )
                    mem_param_readout = torch.tensor(
                        mem_param_readout, device=device, dtype=dtype
                    )  # ltj

                if self.memsyn_het_forward_method == "bandpass":
                    self.dcy_mem = torch.exp(
                        -time_step
                        / (
                            self.memsyn_bandpass_high_ratio_cut
                            * self.tau_mem
                            * torch.sigmoid(mem_param_readout)
                        )
                    )
                elif self.memsyn_het_forward_method == "highpass":
                    # Create an instance of the Softplus class
                    softplus = torch.nn.Softplus()
                    self.dcy_mem = torch.exp(
                        -time_step / (self.tau_mem * softplus(mem_param_readout))
                    )
                elif self.memsyn_het_forward_method == "original":
                    self.dcy_mem = torch.exp(
                        -time_step / (self.tau_mem * mem_param_readout)
                    )
                self.scl_mem = 1.0 - self.dcy_mem
            if (not self.is_delta_syn) and self.syn_het_timescales:
                size_tc = self.shape[0]
                if self.TC_syn_het_init == "Uniform":
                    syn_param_readout = torch.rand(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_syn_het_init == "Gaussian":
                    syn_param_readout = torch.randn(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_syn_het_init == "Constant":
                    syn_param_readout = torch.ones(
                        size_tc, device=device, dtype=dtype
                    )  # ltj
                elif self.TC_syn_het_init == "logNormal":
                    syn_param_readout = torch.empty(size_tc, device=device, dtype=dtype)
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5  # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(syn_param_readout, mean=mean, std=std)
                    syn_param_readout = (
                        syn_param_readout.exp()
                    )  # Converting normal distribution to log-normal
                elif self.TC_syn_het_init == "logspace":
                    syn_param_readout = np.logspace(
                        np.log10(1), np.log10(10), num=size_tc_syn
                    )
                    syn_param_readout = torch.tensor(
                        syn_param_readout, device=device, dtype=dtype
                    )  # ltj

                if self.memsyn_het_forward_method == "bandpass":
                    self.dcy_syn = torch.exp(
                        -time_step
                        / (
                            self.memsyn_bandpass_high_ratio_cut
                            * self.tau_syn
                            * torch.sigmoid(syn_param_readout)
                        )
                    )
                elif self.memsyn_het_forward_method == "highpass":
                    softplus = torch.nn.Softplus()
                    self.dcy_syn = torch.exp(
                        -time_step / (self.tau_syn * softplus(syn_param_readout))
                    )
                elif self.synsyn_het_forward_method == "original":
                    self.dcy_syn = torch.exp(
                        -time_step / (self.tau_syn * syn_param_readout)
                    )
                self.scl_syn = 1.0 - self.dcy_syn
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_timescales:  # ltj
            if self.memsyn_het_forward_method == "bandpass":
                self.dcy_mem = torch.exp(
                    -self.time_step
                    / (
                        self.memsyn_bandpass_high_ratio_cut
                        * self.tau_mem
                        * torch.sigmoid(self.mem_param_readout)
                    )
                )
            elif self.memsyn_het_forward_method == "highpass":
                softplus = torch.nn.Softplus()
                self.dcy_mem = torch.exp(
                    -self.time_step / (self.tau_mem * softplus(self.mem_param_readout))
                )
            elif self.memsyn_het_forward_method == "original":
                self.dcy_mem = torch.exp(
                    -self.time_step / (self.tau_mem * self.mem_param_readout)
                )
            self.scl_mem = 1.0 - self.dcy_mem
            if not self.is_delta_syn:
                if self.memsyn_het_forward_method == "bandpass":
                    self.dcy_syn = torch.exp(
                        -self.time_step
                        / (
                            self.memsyn_bandpass_high_ratio_cut
                            * self.tau_syn
                            * torch.sigmoid(self.syn_param_readout)
                        )
                    )
                elif self.memsyn_het_forward_method == "highpass":
                    softplus = torch.nn.Softplus()
                    self.dcy_syn = torch.exp(
                        -self.time_step
                        / (self.tau_syn * softplus(self.syn_param_readout))
                    )
                elif self.memsyn_het_forward_method == "original":
                    self.dcy_syn = torch.exp(
                        -self.time_step / (self.tau_syn * self.syn_param_readout)
                    )
                self.scl_syn = 1.0 - self.dcy_syn
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        if not self.is_delta_syn:
            self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self):
        # synaptic & membrane dynamics
        if not self.is_delta_syn:
            new_syn = self.dcy_syn * self.syn + self.input
            new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn
        else:
            new_mem = self.dcy_mem * self.out + self.scl_mem * self.input

        self.out = self.states["out"] = new_mem
        if not self.is_delta_syn:
            self.syn = self.states["syn"] = new_syn
        # self.out_seq.append(self.out)
