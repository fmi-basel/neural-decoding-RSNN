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
        if "src" in key or "dst" in key:
            dict.pop(key)

    return dict


def get_bigRSNN_model(model_cfg, stork_state_dict, device):
    """
    Build an equivalent snnTorch model of the bigRSNN model
    from the stork state_dict and the model cfg
    """

    # Remove double keys
    stork_state_dict = remove_double_keys(stork_state_dict)

    # Make sure everything is on the cpu
    for key in stork_state_dict.keys():
        stork_state_dict[key] = stork_state_dict[key].to(device)

    # Get initialization arguments for bigRSNN model
    input_size = stork_state_dict["con1.op.weight"].shape[1]
    hidden_size = model_cfg.hidden_size[0]

    tau_mem_hidden = model_cfg.tau_mem
    tau_syn_hidden = model_cfg.tau_syn
    mem_param_hidden = stork_state_dict["group2.mem_param"]
    syn_param_hidden = stork_state_dict["group2.syn_param"]

    tau_syn_readouts = []
    tau_mem_readouts = []

    for key, d in model_cfg.readouts.items():
        # check if tau_syn and tau_mem are keys of d
        if "tau_syn" in d.keys():
            tau_syn_readouts.append(d["tau_syn"])
        else:
            tau_syn_readouts.append(model_cfg.tau_syn_readout)

        if "tau_mem" in d.keys():
            tau_mem_readouts.append(d["tau_mem"])
        else:
            tau_mem_readouts.append(model_cfg.tau_mem_readout)

    mem_param_readouts = []
    syn_param_readouts = []

    for key in [key for key in stork_state_dict.keys() if "mem_param_readout" in key]:
        mem_param_readouts.append(stork_state_dict[key])

    for key in [key for key in stork_state_dict.keys() if "syn_param_readout" in key]:
        syn_param_readouts.append(stork_state_dict[key])

    # Build the model
    model = bigRSNN(
        input_size,
        tau_mem_hidden,
        tau_syn_hidden,
        mem_param_hidden,
        syn_param_hidden,
        tau_mem_readouts,
        tau_syn_readouts,
        mem_param_readouts,
        syn_param_readouts,
        hidden_size=hidden_size,
        device=device,
    )

    # Get SNNtorch state_dict
    # get state dict
    SNNtorch_state_dict = model.state_dict()

    # replace keys for weight matrices with the ones from the stork state dict
    # hidden weights
    SNNtorch_state_dict["fc_hidden.weight"] = stork_state_dict["con1.op.weight"]
    SNNtorch_state_dict["lif_hidden.recurrent.weight"] = stork_state_dict[
        "con2.op.weight"
    ]

    # readout weights
    for idx in range(len(tau_syn_readouts)):
        SNNtorch_state_dict[f"fc_outputs.{idx}.weight"] = stork_state_dict[
            f"con{idx+3}.op.weight"
        ]
        # print("Replaced {} with {}".format(f'fc_outputs.{idx}.weight', f'con{idx+3}.op.weight'))

    # load state dict
    model.load_state_dict(SNNtorch_state_dict)

    return model


class bigRSNN(nn.Module):
    """
    snnTorch equivalent of the bigRSNN model
    """

    def __init__(
        self,
        input_size,
        tau_mem_hidden,
        tau_syn_hidden,
        mem_param_hidden,
        syn_param_hidden,
        tau_mem_readouts,
        tau_syn_readouts,
        mem_param_readouts,
        syn_param_readouts,
        hidden_size=1024,
        SG_beta=20,
        dt=4e-3,
        device="cpu",
    ):
        super().__init__()

        self.device = device

        # Layer sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nb_readout_heads = len(tau_mem_readouts)
        self.output_size = 2

        # Surrogate gradient
        self.surrogate = surrogate.fast_sigmoid(slope=SG_beta)

        # Input weights
        self.fc_hidden = nn.Linear(
            self.input_size, self.hidden_size, bias=False, device=device
        )

        # Recurrent hidden layer
        self.lif_hidden = RSynaptic_storklike(
            tau_syn=torch.nn.Softplus()(syn_param_hidden) * tau_syn_hidden,
            tau_mem=torch.nn.Softplus()(mem_param_hidden) * tau_mem_hidden,
            dt=dt,
            spike_grad=self.surrogate,
            threshold=1,
            learn_beta=False,
            learn_threshold=False,
            reset_mechanism="zero",
            all_to_all=True,
            linear_features=self.hidden_size,
        )

        # Readout units
        self.lif_readouts = nn.ModuleList()
        self.fc_outputs = nn.ModuleList()

        for i in range(self.nb_readout_heads):
            self.lif_readouts.append(
                Synaptic_storklike(
                    tau_syn=torch.nn.Softplus()(syn_param_readouts[i])
                    * tau_syn_readouts[i],
                    tau_mem=torch.nn.Softplus()(mem_param_readouts[i])
                    * tau_mem_readouts[i],
                    dt=dt,
                    spike_grad=self.surrogate,
                    threshold=1,
                    learn_beta=False,
                    learn_threshold=False,
                    reset_mechanism="none",
                )
            )

            self.fc_outputs.append(
                nn.Linear(self.hidden_size, self.output_size, bias=False, device=device)
            )

        # initialize state variables
        self.inp = None
        self.mem_hidden, self.spk_hidden, self.syn_hidden = None, None, None
        self.mem_readouts, self.syn_readouts = (
            [None] * self.nb_readout_heads,
            [None] * self.nb_readout_heads,
        )
        self.out = None

        # Init recording lists
        self.inp_rec = []
        self.spk_rec = []
        self.pred = []

    def reset(self):
        self.spk_hidden, self.syn_hidden, self.mem_hidden = (
            self.lif_hidden.init_rsynaptic()
        )

        for i in range(self.nb_readout_heads):
            self.mem_readouts[i], self.syn_readouts[i] = self.lif_readouts[
                i
            ].init_synaptic()

        self.spk_rec = []
        self.inp_rec = []
        self.pred = []
        self.inp = None
        self.out = None

    def single_forward(self, x):
        # Input
        x = x.squeeze()  # convert shape (1, input_dim) to (input_dim)

        if self.inp is None:
            self.inp = torch.zeros_like(x, device=self.device)

        new_inp = x

        # Input to hidden layer
        inp_hidden = self.fc_hidden(self.inp)

        # Hidden layer
        if len(self.spk_hidden) == 0:
            self.spk_hidden = torch.zeros(self.hidden_size, device=self.device)

        new_spk_hidden, new_syn_hidden, new_mem_hidden = self.lif_hidden(
            inp_hidden, self.spk_hidden, self.syn_hidden, self.mem_hidden
        )

        # Readout layers
        new_syn_readouts = []
        new_mem_readouts = []

        new_output = torch.zeros(self.output_size, device=self.device)

        for i in range(self.nb_readout_heads):
            inp_readout = self.fc_outputs[i](self.spk_hidden)

            _, new_syn_out, new_mem_out = self.lif_readouts[i](
                inp_readout, self.syn_readouts[i], self.mem_readouts[i]
            )

            new_syn_readouts.append(new_syn_out)
            new_mem_readouts.append(new_mem_out)

            new_output += new_mem_out

        new_output = new_output / self.nb_readout_heads

        # Update state variables
        self.spk_hidden = new_spk_hidden
        self.mem_hidden = new_mem_hidden
        self.syn_hidden = new_syn_hidden

        for i in range(self.nb_readout_heads):
            self.syn_readouts[i] = new_syn_readouts[i]
            self.mem_readouts[i] = new_mem_readouts[i]

        self.inp = new_inp
        self.out = new_output

        return new_spk_hidden.clone(), new_output.clone()

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
