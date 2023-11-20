from conditioned_ldm.src.python.training_and_testing.models.ddpm_v2 import DDPM
from conditioned_ldm.src.python.training_and_testing.models.ae_kl_v1 import AutoencoderKL
from conditioned_ldm.src.python.training_and_testing.models.aekl_no_attention_3d import AutoencoderKL as AutoencoderKL_3D
from conditioned_ldm.src.python.training_and_testing.models.ddpm_v2_conditioned import DDPM as DDPM_C
from conditioned_ldm.src.python.training_and_testing.models.ddpm_v2_3d import DDPM3D
from conditioned_ldm.src.python.training_and_testing.models.ddpm_v2_3d_conditioned import DDPM3D as DDPM3D_C
import torch

from omegaconf import OmegaConf

def define_VAE(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = AutoencoderKL(**config["stage1"]["params"])
    return model


def define_DDPM_unconditioned(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = DDPM(**config["ldm"]["params"])
    return model

def define_DDPM_conditioned(path_to_config):
    config = OmegaConf.load(path_to_config)
    model = DDPM_C(**config["ldm"].get("params", dict()))
    return model

def define_DDPM3D_unconditioned(path_to_config):

    config = OmegaConf.load(path_to_config)
    model = DDPM3D(**config["ldm"]["params"])
    return model

def define_DDPM3D_conditioned(path_to_config):
    config = OmegaConf.load(path_to_config)
    model = DDPM3D_C(**config["ldm"].get("params", dict()))
    return model

def define_VAE3D(path_to_config):
    config = OmegaConf.load(path_to_config)
    model = AutoencoderKL_3D(**config["stage1"]["params"])
    return model

def loadmodel(file_model, config_file, net_name = "VAE"):

    if net_name == "VAE":
        net = define_VAE(config_file)
    elif net_name=='VAE3D':
        net = define_VAE3D(config_file)
    elif net_name == "LDM_CONDITIONED":
        net = define_DDPM_conditioned(config_file)
    elif net_name == "LDM_UNCONDITIONED":
        net = define_DDPM_unconditioned(config_file)
    elif net_name == "LDM3D_CONDITIONED":
        net = define_DDPM3D_conditioned(config_file)
    elif net_name == "LDM3D_UNCONDITIONED":
        net = define_DDPM3D_unconditioned(config_file)

    weights = torch.load(file_model)
    net.load_state_dict(weights)

    return net