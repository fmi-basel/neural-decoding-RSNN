import torch
import torch.nn as nn

from snntorch import surrogate

from .neurons import *

def remove_double_keys(dict):
    # remove every entry whose key has `src` or `dst` in it
    # This is due to an oversight in the `stork` library, which leads to some parameters
    # appearing multiple times in the state_dict
    keys = list(dict.keys())
    for key in keys:
        if 'src' in key or 'dst' in key:
            dict.pop(key)
            
    return dict


def get_tinyRSNN_model(model_cfg, stork_state_dict, device):
    """
    Build an equivalent snnTorch model of the tinyRSNN model
    from the stork state_dict and the model cfg
    """
    
    # Remove double keys
    stork_state_dict = remove_double_keys(stork_state_dict)
    
    # Make sure everything is on the correct device
    for key in stork_state_dict.keys():
        stork_state_dict[key] = stork_state_dict[key].to(device)
            
    # Get initialization arguments for tinyRSNN model
    input_size = stork_state_dict['con1.op.weight'].shape[1]
    hidden_size = model_cfg.hidden_size[0]

    tau_mem_hidden = model_cfg.tau_mem
    tau_syn_hidden = model_cfg.tau_syn
    mem_param_hidden = stork_state_dict['group2.mem_param']
    syn_param_hidden = stork_state_dict['group2.syn_param']

    tau_mem_readout = model_cfg.tau_mem_readout
    tau_syn_readout = model_cfg.tau_syn_readout
    mem_param_readout = stork_state_dict['group3.mem_param_readout']
    syn_param_readout = stork_state_dict['group3.syn_param_readout']
    
    # build model
    model = tinyRSNN(
        input_size=input_size, 
        tau_mem_hidden=tau_mem_hidden, 
        tau_syn_hidden=tau_syn_hidden,
        mem_param_hidden=mem_param_hidden,
        syn_param_hidden=syn_param_hidden,
        tau_mem_readout=tau_mem_readout,
        tau_syn_readout=tau_syn_readout,
        mem_param_readout=mem_param_readout,
        syn_param_readout=syn_param_readout,
        hidden_size=hidden_size, 
        device=device)
    
    # Get SNNtorch state_dict
    # get state dict
    SNNtorch_state_dict = model.state_dict()

    # replace keys for weight matrices with the ones from the stork state dict
    # hidden weights
    SNNtorch_state_dict['fc_hidden.weight'] = stork_state_dict['con1.op.weight']
    SNNtorch_state_dict['lif_hidden.recurrent.weight'] = stork_state_dict['con2.op.weight']

    # output weights
    SNNtorch_state_dict['fc_out.weight'] = stork_state_dict['con3.op.weight']
    
    # load state dict
    model.load_state_dict(SNNtorch_state_dict)
    
    # Make sure float16 is used
    model = model.half()
    model = model.to(device)
    
    return model


class tinyRSNN(nn.Module):

    def __init__(self, 
                 input_size, 
                 tau_mem_hidden, 
                 tau_syn_hidden,
                 mem_param_hidden,
                 syn_param_hidden,
                 tau_mem_readout,
                 tau_syn_readout,
                 mem_param_readout,
                 syn_param_readout,
                 hidden_size=64, 
                 SG_beta=20,
                 dt=4e-3,
                 device='cpu',
                 dtype=torch.float16):
        
        super().__init__()
        
        self.dt = dt
        self.device = device
        self.dtype = dtype
        
        # Layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = 2
        
        # Surrogate gradient 
        self.surrogate = surrogate.fast_sigmoid(slope=SG_beta)

        # Input weights
        self.fc_hidden = nn.Linear(self.input_size, 
                             self.hidden_size, 
                             bias=False, 
                             device=device)

        # Output weights
        self.fc_out = nn.Linear(self.hidden_size,
                                 self.output_size, 
                                 bias=False, 
                                 device=device)
        
        # Recurrent hidden layer
        self.lif_hidden = RSynaptic_storklike(
            tau_syn=torch.nn.Softplus()(syn_param_hidden) * tau_syn_hidden,
            tau_mem=torch.nn.Softplus()(mem_param_hidden) * tau_mem_hidden,
            dt = dt,
            spike_grad=self.surrogate, 
            threshold=1, 
            learn_beta=False,
            learn_threshold=False, 
            reset_mechanism='zero',
            all_to_all=True,
            linear_features=self.hidden_size,
            dtype=self.dtype)
        
        
        # Readout layer
        self.lif_out = Synaptic_storklike(
                tau_syn=torch.nn.Softplus()(syn_param_readout) * tau_syn_readout,
                tau_mem=torch.nn.Softplus()(mem_param_readout) * tau_mem_readout,
                dt = dt,
                spike_grad=self.surrogate, 
                threshold=1, 
                learn_beta=False,
                learn_threshold=False, 
                reset_mechanism='none',
                dtype=self.dtype)
        
        # initialize state variables
        self.inp = None
        self.mem_hidden, self.syn_hidden, self.spk_hidden = None, None, None
        self.mem_out, self.syn_out = None, None
            
        # Init recording lists
        self.inp_rec = []
        self.spk_rec = []
        self.pred = []

    def reset(self):
        
        self.spk_hidden, self.syn_hidden, self.mem_hidden = self.lif_hidden.init_rsynaptic()
        self.syn_out, self.mem_out = self.lif_out.init_synaptic()

        self.spk_rec = []
        self.inp_rec = []
        self.pred = []
        self.inp = None

    def single_forward(self, x):

        # Input
        x = x.squeeze() # convert shape (1, input_dim) to (input_dim)
        
        # Input dtype
        x = x.to(self.dtype)
        x = x.to(self.device)
        
        if self.inp is None:
            self.inp = torch.zeros_like(x, device=self.device).to(self.dtype)
            
        new_inp = x

        # Input to hidden layer    
        inp_hidden = self.fc_hidden(self.inp)

        # Hidden layer

        if len(self.spk_hidden) == 0:
            self.spk_hidden = torch.zeros(self.hidden_size, device=self.device).to(self.dtype)
        
        new_spk_hidden, new_syn_hidden, new_mem_hidden = self.lif_hidden(
            inp_hidden,
            self.spk_hidden,
            self.syn_hidden,
            self.mem_hidden)
        
        # Input to readout layer
        inp_out = self.fc_out(self.spk_hidden)
        
        # Readout layer
        _, new_syn_out, new_mem_out = self.lif_out(
            inp_out, 
            self.syn_out,
            self.mem_out)
            
        self.spk_hidden = new_spk_hidden
        self.mem_hidden = new_mem_hidden
        self.syn_hidden = new_syn_hidden
        self.syn_out = new_syn_out
        self.mem_out = new_mem_out
        self.inp = new_inp

        return new_spk_hidden.clone(), new_mem_out.clone()

    def forward(self, x):
        # here x is expected to be shape (len_series, 1, input_dim)
        predictions = []
        spk_recording = []
        inp_recording = []

        for sample in range(x.shape[0]):
            spk, pred = self.single_forward(x[sample, ...])
            predictions.append(pred)
            spk_recording.append(spk)
            inp_recording.append(x[sample, ...])

        predictions = torch.stack(predictions)
        spk_recording = torch.stack(spk_recording)
        inp_recording = torch.stack(inp_recording)
        self.inp_rec.append(inp_recording)
        self.spk_rec.append(spk_recording)
        self.pred.append(predictions)
        return predictions