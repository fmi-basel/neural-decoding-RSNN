import torch
import stork
from stork.nodes import InputGroup
from stork.connections import Connection
from stork.initializers import (
    FluctuationDrivenCenteredNormalInitializer,
    DistInitializer,
)
from stork.layers import Layer

# Custom additions to stork
from .custom.readout import CustomReadoutGroup, AverageReadouts
from .custom.lif import CustomLIFGroup
from .custom.models import CustomRecurrentSpikingModel

from .data import compute_input_firing_rates

import logging

logger = logging.getLogger(__name__)


def get_regularizers(cfg):
    regs = []
    regLB = stork.regularizers.LowerBoundL2(
        cfg.training.LB_L2_strength, threshold=cfg.training.LB_L2_thresh, dims=False
    )
    regs.append(regLB)
    regUB = stork.regularizers.UpperBoundL2(
        cfg.training.UB_L2_strength, threshold=cfg.training.UB_L2_thresh, dims=1
    )
    regs.append(regUB)

    return regs


def get_actfn(cfg):
    act_fn = stork.activations.CustomSpike
    if cfg.model.stochastic:
        act_fn.escape_noise_type = "sigmoid"
    else:
        act_fn.escape_noise_type = "step"
    act_fn.escape_noise_params = {"beta": cfg.training.SG_beta}
    act_fn.surrogate_type = "SuperSpike"
    act_fn.surrogate_params = {"beta": cfg.training.SG_beta}

    return act_fn


def get_initializers(cfg, nu=None, dtype=torch.float32):

    if nu is None or cfg.initializer.compute_nu is False:
        nu = cfg.initializer.nu
    else:
        logger.info(f"Initializing with nu = {nu}")

    # Initializers
    hidden_init = FluctuationDrivenCenteredNormalInitializer(
        sigma_u=cfg.initializer.sigma_u,
        nu=nu,
        timestep=cfg.data.dt,
        alpha=cfg.initializer.alpha,
        dtype=dtype
    )

    readout_init = DistInitializer(
        dist=torch.distributions.Normal(0, 1), 
        scaling="1/sqrt(k)", 
        dtype=dtype
    )

    return hidden_init, readout_init


def get_model(cfg, nb_inputs, dtype, data=None):

    nb_time_steps = int(cfg.data.sample_duration / cfg.data.dt)
    nb_outputs = cfg.data.nb_outputs

    model = CustomRecurrentSpikingModel(
        cfg.training.batchsize,
        nb_time_steps=nb_time_steps,
        nb_inputs=nb_inputs,
        device=cfg.device,
        dtype=dtype,
    )

    # Activation function
    act_fn = get_actfn(cfg)

    # Regularizer list
    regs = get_regularizers(cfg)

    # Compute mean firing rates for initializer
    if data is not None:
        mean1, mean2 = compute_input_firing_rates(data, cfg)
    else:
        mean1 = None

    hidden_init, readout_init = get_initializers(cfg, mean1, dtype)

    # INPUT LAYER
    # # # # # # # #

    input_group = model.add_group(InputGroup(nb_inputs, 
                                             dropout_p=cfg.model.dropout_p))

    # HIDDEN LAYERS
    # # # # # # # #
    current_src_grp = input_group

    hidden_neuron_kwargs = {
        "tau_mem": cfg.model.tau_mem,
        "tau_syn": cfg.model.tau_syn,
        "activation": act_fn,
        "dropout_p": cfg.model.dropout_p,
        "het_timescales": cfg.model.het_timescales,
        "learn_timescales": cfg.model.learn_timescales,
        "is_delta_syn": cfg.model.delta_synapses,
    }

    for i in range(cfg.model.nb_hidden):

        hidden_layer = Layer(
            name="hidden",
            model=model,
            size=cfg.model.hidden_size[i],
            input_group=current_src_grp,
            recurrent=cfg.model.recurrent[i],
            regs=regs,
            neuron_class=CustomLIFGroup,
            neuron_kwargs=hidden_neuron_kwargs,
            connection_kwargs={},
        )

        current_src_grp = hidden_layer.output_group

        # initialize
        hidden_init.initialize(hidden_layer)

        if i == 0 and nb_inputs == 192 and data is not None:
            with torch.no_grad():
                hidden_layer.connections[0].op.weight[:, 96:] /= mean2 / mean1

    # READOUT LAYER
    # # # # # # # #

    if cfg.model.multiple_readouts:
        logger.info("Adding custom readout groups")
        custom_readouts = get_custom_readouts(cfg)
        for g in custom_readouts:
            model.add_group(g)
            con_ro = model.add_connection(Connection(current_src_grp, g, dtype=dtype))
            readout_init.initialize(con_ro)

        model.add_group(AverageReadouts(model.groups[-len(custom_readouts) :]))
    else:
        logger.info("Adding single readout group")
        readout_group = model.add_group(
            CustomReadoutGroup(
                nb_outputs,
                tau_mem=cfg.model.tau_mem_readout,
                tau_syn=cfg.model.tau_syn_readout,
                het_timescales=cfg.model.het_timescales_readout,
                learn_timescales=cfg.model.learn_timescales_readout,
                initial_state=-1e-2,
                is_delta_syn=cfg.model.delta_synapses,
            )
        )

        con_ro = model.add_connection(
            Connection(current_src_grp, readout_group, dtype=dtype)
        )

        readout_init.initialize(con_ro)


    return model


def get_custom_readouts(cfg):
    ro_list = []
    for ro, specs in cfg.model["readouts"].items():
        if "tau_mem" in specs:
            tau_mem = specs["tau_mem"]
        else:
            tau_mem = cfg.model.tau_mem_readout
        if "tau_syn" in specs:
            tau_syn = specs["tau_syn"]
        else:
            tau_syn = cfg.model.tau_syn_readout

        if specs["type"] == "default":
            ro_group = CustomReadoutGroup(
                cfg.data.nb_outputs,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                het_timescales=cfg.model.het_timescales_readout,
                learn_timescales=cfg.model.learn_timescales_readout,
                initial_state=-1e-2,
                is_delta_syn=False,
            )
        elif specs["type"] == "delta":
            ro_group = CustomReadoutGroup(
                cfg.data.nb_outputs,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                het_timescales=cfg.model.het_timescales_readout,
                learn_timescales=cfg.model.learn_timescales_readout,
                initial_state=-1e-2,
                is_delta_syn=True,
            )

        ro_group.set_name(ro)
        ro_list.append(ro_group)

    return ro_list
