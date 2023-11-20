import sys
from generative.networks.nets.diffusion_model_unet import DiffusionModelUNet
from generative.networks.nets.autoencoderkl import AutoencoderKL
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler
from generative.inferers.inferer import LatentDiffusionInferer
import torch

from omegaconf import OmegaConf

def define_VAE(path_to_config, vae_model_path):

    config = OmegaConf.load(path_to_config)
    model = AutoencoderKL(**config["stage1"]["params"]["hparams"])
    model.load_state_dict(torch.load(vae_model_path))
    return model, config['stage1']['resolution']

def define_DDPM(path_to_config, ldm_model_path, scheduler_type = 'ddpm', num_inference_steps = 150):

    if scheduler_type not in ['ddpm', 'pndm', 'ddim']:
        raise ValueError("Scheduler not recognised.")
    config = OmegaConf.load(path_to_config)
    model = DiffusionModelUNet(**config['ldm']['params']['unet_config']['params'])
    model.load_state_dict(torch.load(ldm_model_path))
    if scheduler_type == 'ddpm':
        scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=config['ldm']['params']['ddpm']['beta_schedule'],
            beta_start=config['ldm']['params']['ddpm']['beta_start'],
            beta_end=config['ldm']['params']['ddpm']['beta_end'],
            prediction_type=config['ldm']['params']['ddpm']['prediction_type'],
            clip_sample=False,
        )
    elif scheduler_type == 'ddim':
        scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule=config['ldm']['params']['ddpm']['beta_schedule'],
            beta_start=config['ldm']['params']['ddpm']['beta_start'],
            beta_end=config['ldm']['params']['ddpm']['beta_end'],
            prediction_type=config['ldm']['params']['ddpm']['prediction_type'],
            clip_sample=False,
        )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    else:
        scheduler = PNDMScheduler(
            num_train_timesteps=1000,
            beta_schedule=config['ldm']['params']['ddpm']['beta_schedule'],
            beta_start=config['ldm']['params']['ddpm']['beta_start'],
            beta_end=config['ldm']['params']['ddpm']['beta_end'],
            prediction_type=config['ldm']['params']['ddpm']['prediction_type'],
        )
        scheduler.set_timesteps(num_inference_steps=num_inference_steps)
    return model, scheduler, config['ldm']['params']['scale_factor']



