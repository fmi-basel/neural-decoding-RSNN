import numpy as np
from scipy.stats import beta
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class CustomLIFGroup(CellGroup):
    def __init__(self, shape, tau_mem=10e-3, tau_syn=5e-3, diff_reset=False, 
                 learn_timescales=False, het_timescales=False, 
                 mem_het_timescales=False, syn_het_timescales=False, 
                 clamp_mem=False, TC_mem_het_init='Uniform',
                 TC_syn_het_init='Uniform', memsyn_bandpass_high_ratio_cut=2, 
                 is_delta_syn = False, memsyn_het_forward_method='highpass',
                 activation=activations.SuperSpike, dropout_p=0.0, stateful=False, name="LIFGroup", regularizers=None, **kwargs):
        """
        Custom LIF group based on the stork LIF group.
        The big difference is that this group allows for heterogenous & learnable
        time constants for the membrane and synaptic time constants.
        
        DISCALIMER: 
        This group has been tested and is working as expected,
        but requires a more efficient implementation and will be added to the main `stork` library in the future.
        Currently introduces a problem in the model state dict, where time constants are saved multiple times.

        Args: 
            :param shape: The number of units in this group
            :type shape: int or tuple of int
            :param tau_mem: The membrane time constant in s, defaults to 10e-3
            :type tau_mem: float
            :param tau_syn: The synaptic time constant in s, defaults to 5e-3
            :type tau_syn: float
            :param diff_reset: Whether or not to differentiate through the reset term, defaults to False
            :type diff_reset: bool
            :param learn_timescales: Whether to learn the membrane and synaptic time constants, defaults to False
            :type learn_timescales: bool
            :param het_timescales: Whether to set different time constants for different neurons for learning, , defaults to False #ltj
            :type het_timescales: bool
            :param activation: The surrogate derivative enabled activation function, defaults to stork.activations.SuperSpike
            :type activation: stork.activations
            :param dropout_p: probability that some elements of the input will be zeroed, defaults to 0.0
            :type dropout_p: float
            :param stateful: Whether or not to reset the state of the neurons between mini-batches, defaults to False
            :type stateful: bool
            :param regularizers: List of regularizers
        """

        super().__init__(shape, dropout_p=dropout_p, stateful=stateful,
                         name=name, regularizers=regularizers, **kwargs)
        self.tau_mem = tau_mem
        self.spk_nl = activation.apply
        self.diff_reset = diff_reset
        self.learn_timescales = learn_timescales
        self.clamp_mem = clamp_mem
        self.mem = None
        self.het_timescales = het_timescales #ltj
        self.mem_het_timescales = mem_het_timescales #ltj
        self.syn_het_timescales = syn_het_timescales #ltj
        self.is_delta_syn = is_delta_syn #ltj
        self.TC_mem_het_init = TC_mem_het_init
        self.TC_syn_het_init = TC_syn_het_init
        self.memsyn_bandpass_high_ratio_cut = memsyn_bandpass_high_ratio_cut
        self.memsyn_het_forward_method = memsyn_het_forward_method
        self.syn = None       
        self.tau_syn = tau_syn

    def configure(self, batch_size, nb_steps, time_step, device, dtype): 
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        if self.het_timescales:
            self.mem_het_timescales = True
            self.syn_het_timescales = True
        if not self.is_delta_syn:
            self.dcy_syn = float(np.exp(-time_step / self.tau_syn))
            self.scl_syn = 1.0 - self.dcy_syn
        if self.learn_timescales:
            size_tc_mem = self.shape[0] if self.mem_het_timescales else 1 #ltj
            size_tc_syn = self.shape[0] if self.syn_het_timescales else 1 #ltj
            if self.TC_mem_het_init == 'Uniform':
                mem_param = torch.rand(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True) #ltj
            elif self.TC_mem_het_init == 'Gaussian':  # modify
                mem_param = torch.randn(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True) #ltj
            elif self.TC_mem_het_init == 'Constant':
                mem_param = torch.ones(
                    size_tc_mem, device=device, dtype=dtype, requires_grad=True) #ltj
            # elif self.TC_mem_het_init == 'XavierUniform':
            #     mem_param = torch.empty(size_tc_mem, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_uniform_(mem_param)  # Xavier uniform initialization
            # elif self.TC_mem_het_init == 'XavierGassian':
            #     mem_param = torch.empty(size_tc_mem, device=device, dtype=dtype, requires_grad=True)
            #     torch.nn.init.xavier_normal_(mem_param)  # Xavier gaussian initialization
            elif self.TC_mem_het_init == 'logNormal':
                mem_param = torch.empty(size_tc_mem, device=device, dtype=dtype, requires_grad=True)
                mean = -4.3  # Example mean of the underlying normal distribution
                std = 2.5   # Example standard deviation of the underlying normal distribution
                torch.nn.init.normal_(mem_param, mean=mean, std=std)
                mem_param = mem_param.exp()  # Converting normal distribution to log-normal
            elif self.TC_mem_het_init == 'logspace':
                mem_param = np.logspace(np.log10(1), np.log10(10), num=size_tc_mem)
                mem_param = torch.tensor(mem_param, device=device, dtype=dtype) #ltj
            self.mem_param = Parameter(mem_param, requires_grad=self.learn_timescales)
            if not self.is_delta_syn:
                if self.TC_syn_het_init == 'Uniform':
                    syn_param = torch.rand(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True) #ltj
                elif self.TC_syn_het_init == 'Gaussian':
                    syn_param = torch.randn(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True) #ltj
                elif self.TC_syn_het_init == 'Constant':
                    syn_param = torch.ones(
                        size_tc_syn, device=device, dtype=dtype, requires_grad=True) #ltj
                # elif self.TC_syn_het_init == 'XavierUniform':
                #     syn_param = torch.empty(size_tc_syn, device=device, dtype=dtype, requires_grad=True)
                #     torch.nn.init.xavier_uniform_(syn_param)  # Xavier uniform initialization
                # elif self.TC_syn_het_init == 'XavierGassian':
                #     syn_param = torch.empty(size_tc_syn, device=device, dtype=dtype, requires_grad=True)
                #     torch.nn.init.xavier_normal_(syn_param)  # Xavier gaussian initialization
                elif self.TC_syn_het_init == 'logNormal':
                    syn_param = torch.empty(size_tc_syn, device=device, dtype=dtype, requires_grad=True)
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5   # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(syn_param, mean=mean, std=std)
                    syn_param = syn_param.exp()  # Converting normal distribution to log-normal
                elif self.TC_syn_het_init == 'logspace':
                    syn_param = np.logspace(np.log10(1), np.log10(10), num=size_tc_syn)
                    syn_param = torch.tensor(syn_param, device=device, dtype=dtype) #ltj
                self.syn_param = Parameter(
                    syn_param, requires_grad=self.learn_timescales)
        elif self.mem_het_timescales or self.syn_het_timescales:
            if self.mem_het_timescales:
                size_tc = self.shape[0]
                if self.TC_mem_het_init == 'Uniform':
                    mem_param = torch.rand(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_mem_het_init == 'Gaussian':
                    mem_param = torch.randn(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_mem_het_init == 'Constant':
                    mem_param = torch.ones(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_mem_het_init == 'logNormal':
                    mem_param = torch.empty(size_tc, device=device, dtype=dtype)
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5   # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(mem_param, mean=mean, std=std)
                    mem_param = mem_param.exp()  # Converting normal distribution to log-normal
                elif self.TC_mem_het_init == 'logspace':
                    mem_param = np.logspace(np.log10(1), np.log10(10), num=size_tc)
                    mem_param = torch.tensor(mem_param, device=device, dtype=dtype) #ltj

                if self.memsyn_het_forward_method == 'bandpass':
                    self.dcy_mem = torch.exp(-time_step /
                                            (self.memsyn_bandpass_high_ratio_cut * self.tau_mem * torch.sigmoid(mem_param)))
                elif self.memsyn_het_forward_method == 'highpass':
                    # Create an instance of the Softplus class
                    softplus = torch.nn.Softplus()
                    self.dcy_mem = torch.exp(-time_step /
                                            (self.tau_mem * softplus(mem_param))) 
                elif self.memsyn_het_forward_method == 'original':
                    self.dcy_mem = torch.exp(-time_step /
                                            (self.tau_mem * mem_param))
                self.scl_mem = 1.0 - self.dcy_mem
            if (not self.is_delta_syn) and self.syn_het_timescales:
                size_tc = self.shape[0]
                if self.TC_syn_het_init == 'Uniform':
                    syn_param = torch.rand(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_syn_het_init == 'Gaussian':
                    syn_param = torch.randn(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_syn_het_init == 'Constant':
                    syn_param = torch.ones(size_tc, device=device, dtype=dtype) #ltj
                elif self.TC_syn_het_init == 'logNormal':
                    syn_param = torch.empty(size_tc, device=device, dtype=dtype)
                    mean = -4.3  # Example mean of the underlying normal distribution
                    std = 2.5   # Example standard deviation of the underlying normal distribution
                    torch.nn.init.normal_(syn_param, mean=mean, std=std)
                    syn_param = syn_param.exp()  # Converting normal distribution to log-normal
                elif self.TC_syn_het_init == 'logspace':
                    syn_param = np.logspace(np.log10(1), np.log10(10), num=size_tc_syn)
                    syn_param = torch.tensor(syn_param, device=device, dtype=dtype) #ltj

                if self.memsyn_het_forward_method == 'bandpass':
                    self.dcy_syn = torch.exp(-time_step /
                                            (self.memsyn_bandpass_high_ratio_cut * self.tau_syn * torch.sigmoid(syn_param)))
                elif self.memsyn_het_forward_method == 'highpass':
                    # Create an instance of the Softplus class
                    softplus = torch.nn.Softplus()
                    self.dcy_syn = torch.exp(-time_step /
                                            (self.tau_syn * softplus(syn_param)))
                elif self.synsyn_het_forward_method == 'original':
                    self.dcy_syn = torch.exp(-time_step /
                                            (self.tau_syn * syn_param))
                self.scl_syn = 1.0 - self.dcy_syn
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.learn_timescales:
            if self.memsyn_het_forward_method == 'bandpass':
                self.dcy_mem = torch.exp(-self.time_step /
                                        (self.memsyn_bandpass_high_ratio_cut * self.tau_mem * torch.sigmoid(self.mem_param)))
            elif self.memsyn_het_forward_method == 'highpass':
                # Create an instance of the Softplus class
                softplus = torch.nn.Softplus()
                self.dcy_mem = torch.exp(-self.time_step /
                                        (self.tau_mem * softplus(self.mem_param)))
            elif self.memsyn_het_forward_method == 'original':
                self.dcy_mem = torch.exp(-self.time_step /
                                        (self.tau_mem * self.mem_param))
            self.scl_mem = 1.0 - self.dcy_mem
            if not self.is_delta_syn:
                if self.memsyn_het_forward_method == 'bandpass': # watch TC distri
                    self.dcy_syn = torch.exp(-self.time_step /
                                            (self.memsyn_bandpass_high_ratio_cut * self.tau_syn * torch.sigmoid(self.syn_param)))
                elif self.memsyn_het_forward_method == 'highpass':
                    # Create an instance of the Softplus class
                    softplus = torch.nn.Softplus()
                    self.dcy_syn = torch.exp(-self.time_step /
                                            (self.tau_syn * softplus(self.syn_param)))
                elif self.memsyn_het_forward_method == 'original':
                    self.dcy_syn = torch.exp(-self.time_step /
                                            (self.tau_syn * self.syn_param))
                self.scl_syn = 1.0 - self.dcy_syn
        self.mem = self.get_state_tensor("mem", state=self.mem)
        if not self.is_delta_syn:
            self.syn = self.get_state_tensor("syn", state=self.syn)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype)

    def get_spike_and_reset(self, mem):
        mthr = mem - 1.0
        out = self.spk_nl(mthr)

        if self.diff_reset:
            rst = out
        else:
            # if differentiation should not go through reset term, detach it from the computational graph
            rst = out.detach()

        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        if not self.is_delta_syn:
            new_syn = self.dcy_syn * self.syn + self.input 
            new_mem = (self.dcy_mem * self.mem + self.scl_mem *
                    self.syn) * (1.0 - rst)  # multiplicative reset 
        else:
            new_mem = (self.dcy_mem * self.mem + self.scl_mem *
                    self.input) * (1.0 - rst)  # multiplicative reset

        # Clamp membrane potential
        if self.clamp_mem:
            new_mem = torch.clamp(new_mem, max=1.01)

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
        if not self.is_delta_syn:
            self.syn = self.states["syn"] = new_syn
