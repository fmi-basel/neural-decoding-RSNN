# Stork-like SNNTorch neurons
#
# Julian Rossbroich, July 2024
#
# ATTENTION:
# Only works for `init_hidden` = False and `output` = False
# If you want to change that, you need to make more changes in the forward pass
# and the _base_state_function related methods

import torch
import torch.nn as nn

import snntorch as snn
from snntorch._neurons.neurons import _SpikeTorchConv


class Leaky_storklike(snn.Leaky):

    def __init__(
        self,
        dt,
        tau_mem,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):
        super(Leaky_storklike, self).__init__(
            tau_mem,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self.dt = dt
        self.tau_mem = tau_mem
        self.dcy_mem = torch.exp(-self.dt / self.tau_mem)
        self.scl_mem = 1 - self.dcy_mem
    
    def _base_state_function(self, input_, mem):
        
        base_fn = self.dcy_mem * mem + input_ * self.scl_mem
        return base_fn

    def forward(self, input_, mem=False):

        if hasattr(mem, "init_flag"):  # only triggered on first-pass
            mem = _SpikeTorchConv(mem, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.mem = _SpikeTorchConv(self.mem, input_=input_)

        # CHANGED FOR STORKLIKE: Calculate output spike based on old mem, not updated mem
        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            new_mem = self._build_state_function(input_, mem)

            if self.state_quant:
                new_mem = self.state_quant(new_mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, new_mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._leaky_forward_cases(mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk
            

class Rleaky_storklike(snn.RLeaky):
    
    def __init__(
        self,
        dt,
        tau_mem,
        V=1.0,
        all_to_all=True,
        linear_features=None,
        conv2d_channels=None,
        kernel_size=None,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True,  # changed learn_V
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
    ):
        super(Rleaky_storklike, self).__init__(
            tau_mem,
            V,
            all_to_all,
            linear_features,
            conv2d_channels,
            kernel_size,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            learn_recurrent,
            reset_mechanism,
            state_quant,
            output,
        )
            
        self.dt = dt
        self.dcy_mem = torch.exp(-self.dt / tau_mem)
        self.scl_mem = 1 - self.dcy_mem
    
    
    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, 
                                   self.linear_features, 
                                   bias=False)

    def _base_state_function(self, input_, spk, mem):
        base_fn = self.dcy_mem * mem + self.scl_mem * (input_ + self.recurrent(spk))
        return base_fn
    
    def forward(self, input_, spk=False, mem=False):
        if hasattr(spk, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            spk, mem = _SpikeTorchConv(spk, mem, input_=input_)
        # init_hidden case
        elif mem is False and hasattr(self.mem, "init_flag"):
            self.spk, self.mem = _SpikeTorchConv(
                self.spk, self.mem, input_=input_
            )

        # CHANGED FOR STORKLIKE: Calculate output spike based on old mem, not updated mem
        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            new_mem = self._build_state_function(input_, spk, mem)

            if self.state_quant:
                new_mem = self.state_quant(new_mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)  # batch_size
            else:
                spk = self.fire(mem)

            return spk, new_mem

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._rleaky_forward_cases(spk, mem)
            self.reset = self.mem_reset(self.mem)
            self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:  # read-out layer returns output+states
                return self.spk, self.mem
            else:  # hidden layer e.g., in nn.Sequential, only returns output
                return self.spk


class Synaptic_storklike(snn.Synaptic):
    def __init__(
        self,
        dt,
        tau_mem,
        tau_syn,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        dtype=torch.float32
    ):
        super(Synaptic_storklike, self).__init__(
            False,
            False,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
        )
        
        self.dt = dt
        self.dtype=dtype
        self.dcy_mem = torch.exp(-self.dt / tau_mem).to(self.dtype)
        self.scl_mem = 1 - self.dcy_mem
        self.dcy_syn = torch.exp(-self.dt / tau_syn).to(self.dtype)

    def _base_state_function(self, input_, syn, mem):
        base_fn_syn = self.dcy_syn * syn + input_
        base_fn_mem = self.dcy_mem * mem + self.scl_mem * syn
        return base_fn_syn, base_fn_mem
    
    def _base_state_reset_zero(self, input_, syn, mem):
        base_fn_mem = self.dcy_mem * mem + self.scl_mem * syn
        return 0, base_fn_mem
    
    def forward(self, input_, syn=False, mem=False):

        if hasattr(syn, "init_flag") or hasattr(
            mem, "init_flag"
        ):  # only triggered on first-pass
            syn, mem = _SpikeTorchConv(syn, mem, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.syn, self.mem = _SpikeTorchConv(
                self.syn, self.mem, input_=input_
            )

        # CHANGED FOR STORKLIKE: Calculate output spike based on old mem, not updated mem
        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            new_syn, new_mem = self._build_state_function(input_, syn, mem)

            if self.state_quant:
                new_syn = self.state_quant(new_syn)
                new_mem = self.state_quant(new_mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk.to(self.dtype), new_syn.to(self.dtype), new_mem.to(self.dtype)

        # intended for truncated-BPTT where instance variables are
        # hidden states
        if self.init_hidden:
            self._synaptic_forward_cases(mem, syn)
            self.reset = self.mem_reset(self.mem)
            self.syn, self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.syn = self.state_quant(self.syn)
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn, self.mem
            else:
                return self.spk
    

class RSynaptic_storklike(snn.RSynaptic):
    def __init__(
        self,
        dt,
        tau_syn,
        tau_mem,
        V=1.0,
        all_to_all=True,
        linear_features=None,
        conv2d_channels=None,
        kernel_size=None,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=False,
        learn_beta=False,
        learn_threshold=False,
        learn_recurrent=True,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        dtype=torch.float32
    ):
        super(RSynaptic_storklike, self).__init__(
            False,
            False,
            V,
            all_to_all,
            linear_features,
            conv2d_channels,
            kernel_size,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_threshold,
            learn_recurrent,
            reset_mechanism,
            state_quant,
            output,
        )
        
        self.dt = dt
        self.dtype = dtype
        self.dcy_mem = torch.exp(-self.dt / tau_mem).to(self.dtype)
        self.scl_mem = 1 - self.dcy_mem
        self.dcy_syn = torch.exp(-self.dt / tau_syn).to(self.dtype)
        
    def _init_recurrent_linear(self):
        self.recurrent = nn.Linear(self.linear_features, 
                                   self.linear_features, 
                                   bias=False)
        
    def _base_state_function(self, input_, spk, syn, mem):
        base_fn_syn = self.dcy_syn * syn + input_ + self.recurrent(spk)
        base_fn_mem = self.dcy_mem * mem + self.scl_mem * syn
        return base_fn_syn, base_fn_mem
    
    def _base_state_reset_zero(self, input_, spk, syn, mem):
        base_fn_mem = self.dcy_mem * mem + self.scl_mem * syn
        return 0, base_fn_mem
    
    
    def forward(self, input_, spk=False, syn=False, mem=False):
        if (
            hasattr(spk, "init_flag")
            or hasattr(syn, "init_flag")
            or hasattr(mem, "init_flag")
        ):  # only triggered on first-pass
            spk, syn, mem = _SpikeTorchConv(spk, syn, mem, input_=input_)
        elif mem is False and hasattr(
            self.mem, "init_flag"
        ):  # init_hidden case
            self.spk, self.syn, self.mem = _SpikeTorchConv(
                self.spk, self.syn, self.mem, input_=input_
            )

        # CHANGED FOR STORKLIKE: Calculate output spike based on old mem, not updated mem
        if not self.init_hidden:
            self.reset = self.mem_reset(mem)
            new_syn, new_mem = self._build_state_function(input_, spk, syn, mem)

            if self.state_quant:
                new_syn = self.state_quant(new_syn)
                new_mem = self.state_quant(new_mem)

            if self.inhibition:
                spk = self.fire_inhibition(mem.size(0), mem)
            else:
                spk = self.fire(mem)

            return spk.to(self.dtype), new_syn.to(self.dtype), new_mem.to(self.dtype)

        # intended for truncated-BPTT where instance variables are hidden
        # states
        if self.init_hidden:
            self._rsynaptic_forward_cases(spk, mem, syn)
            self.reset = self.mem_reset(self.mem)
            self.syn, self.mem = self._build_state_function_hidden(input_)

            if self.state_quant:
                self.syn = self.state_quant(self.syn)
                self.mem = self.state_quant(self.mem)

            if self.inhibition:
                self.spk = self.fire_inhibition(self.mem.size(0), self.mem)
            else:
                self.spk = self.fire(self.mem)

            if self.output:
                return self.spk, self.syn, self.mem
            else:
                return self.spk