# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Union
from generative.inferers import LatentDiffusionInferer
import torch
import torch.nn as nn
from monai.utils import optional_import
import monai.transforms as mt
from torch.types import _device
import monai
from label_ldm.label_generator.wrappers import Stage1Wrapper

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")


class SizeableInferer(LatentDiffusionInferer):
    """
    SizeableInferer is an extension of LatentDiffusionInferer, where the autoencoder and the LDM latent spaces
    dont need to be the same.
    Takes in:
    :param scheduler generative scheduler, to handle the diffusion process
    :param latent_shape_vae: list, shape of the VAE latent space (without batch dim)
    :param latent_shape_ldm: list, shape of the LDM latent space (without batch dim)
    :param wrapped_vae: VAE is wrapped in a DataParallel, Stage1Wrapper is used in __call__
    """

    def __init__(self, scheduler: nn.Module, scale_factor: float,
                 latent_shape_vae: list, latent_shape_ldm: list,
                 wrapped_vae: bool = False) -> None:
        super().__init__(scheduler=scheduler, scale_factor=scale_factor)

        self.scheduler = scheduler
        self.latent_shape_vae = latent_shape_vae
        self.latent_shape_ldm = latent_shape_ldm
        self.wrapped_vae = wrapped_vae

        if self.latent_shape_ldm != self.latent_shape_vae:
            self.tf_pad = mt.Compose([mt.SpatialPad(self.latent_shape_ldm),
                                      mt.ToTensor(dtype=torch.float32)])
            self.tf_crop = mt.Compose([mt.CenterSpatialCrop(self.latent_shape_vae),
                                       mt.ToTensor(dtype=torch.float32)])
        else:
            self.tf_pad = None
            self.tf_crop = None

    def encode_latents(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        device: _device,
    ) -> torch.Tensor:

        """
        Implements the forward pass for a supervised training iteration.
        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            device: CUDA or CPU device
        """

        with torch.no_grad():
            if self.wrapped_vae:
                latent = autoencoder_model(inputs)
                latent = latent * self.scale_factor
            else:
                latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor
            if self.tf_pad is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_pad(latent.numpy())
                latent = latent.type(latent_type).to(device)

        return latent

    def __call__(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        device: _device,
        condition: Optional[torch.Tensor] = None,
        focal_loss_weight: Optional[float] = 1.0
    ) -> torch.Tensor:

        """
        Implements the forward pass for a supervised training iteration.
        Args:
            inputs: input image to which the latent representation will be extracted and noise is added.
            autoencoder_model: first stage model.
            diffusion_model: diffusion model.
            noise: random noise, of the same shape as the latent representation.
            timesteps: random timesteps.
            device: CUDA or CPU device
            condition: conditioning for network input.
        """

        with torch.no_grad():
            if self.wrapped_vae:
                latent = autoencoder_model(inputs)
                latent = latent * self.scale_factor
            else:
                latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor
            if self.tf_pad is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_pad(latent.numpy())
                latent = latent.type(latent_type).to(device)

        noisy_image = self.scheduler.add_noise(original_samples=latent, noise=noise, timesteps=timesteps)
        prediction = super(LatentDiffusionInferer, self).__call__(
            inputs=latent,
            diffusion_model=diffusion_model,
            noise=noise,
            timesteps=timesteps,
            condition=condition,
        )

        if focal_loss_weight > 0.0:
            alpha_prod_t = torch.gather(self.scheduler.alphas_cumprod, 0, timesteps)
            beta_prod_t = 1 - alpha_prod_t
            if self.scheduler.prediction_type == "epsilon":
                pred_seg = (noisy_image - beta_prod_t[:, None, None, None, None].expand(prediction.shape) \
                            ** (0.5) * prediction) / alpha_prod_t[:, None, None, None, None].expand(prediction.shape) \
                            ** (0.5)
                pred_seg2 = (noisy_image - beta_prod_t[:, None, None, None, None] \
                            ** (0.5) * prediction) / alpha_prod_t[:, None, None, None, None] \
                            ** (0.5)
            elif self.scheduler.prediction_type == "v_prediction":
                pred_seg = (alpha_prod_t[:, None, None, None, None].expand(prediction.shape) ** 0.5) * \
                           noisy_image - (beta_prod_t[:, None, None, None, None].expand(prediction.shape) ** 0.5) * \
                           prediction
            if self.tf_crop is not None:
                pred_seg = self.tf_crop(pred_seg)
            pred_seg = autoencoder_model.decode_stage_2_outputs(pred_seg / self.scale_factor)
        else:
            return prediction, None

        return prediction, pred_seg

    def sample(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        device: _device,
        scheduler: Optional[Callable[..., torch.Tensor]] = None,
        save_intermediates: Optional[bool] = False,
        intermediate_steps: Optional[int] = 100,
        conditioning: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = True,
        focal_loss: Optional[float] = 1.0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            device: CUDA or CPU device
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
            focal_loss: float weight given to focal loss on
        """

        outputs = super(LatentDiffusionInferer, self).sample(
            input_noise=input_noise,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            intermediate_steps=intermediate_steps,
            conditioning=conditioning,
            verbose=verbose,
        )

        if save_intermediates:
            latent, latent_intermediates = outputs
            latent_intermediates = [l.detach().cpu() for l in latent_intermediates]
        else:
            latent = outputs
        latent = latent.detach().cpu()
        autoencoder_model = autoencoder_model.cpu()
        with torch.no_grad():
            if self.tf_crop is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_crop(latent.numpy())
                latent = latent.type(latent_type)

            image = autoencoder_model.decode_stage_2_outputs(latent / self.scale_factor)
            latent = latent.detach().cpu()
        if save_intermediates:
            intermediates = []
            for latent_intermediate in latent_intermediates:
                with torch.no_grad():
                    if self.tf_crop is not None:
                        latent_intermediate_ = latent_intermediate
                        latent_intermediate_type = deepcopy(latent_intermediate_.type())
                        latent_intermediate_ = self.tf_crop(latent_intermediate_.numpy())
                        latent_intermediate_ = latent_intermediate_.type(latent_intermediate_type) #.to(device)
                    else:
                        latent_intermediate_ = latent_intermediate

                    intermediates.append(
                        autoencoder_model.decode_stage_2_outputs(latent_intermediate_
                                                                 / self.scale_factor) #.detach().cpu()
                    )

                    latent_intermediate_ = latent_intermediate_ #.detach().cpu()

            autoencoder_model = autoencoder_model.to(device)
            return image, intermediates

        else:
            autoencoder_model = autoencoder_model.to(device)
            return image

    def sample_wguid(
        self,
        input_noise: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        device: _device,
        scheduler: Optional[Callable[..., torch.Tensor]] = None,
        save_intermediates: Optional[bool] = False,
        intermediate_steps: Optional[int] = 100,
        conditioning: Optional[torch.Tensor] = None,
        verbose: Optional[bool] = True,
        guidance_scale: int = 7.0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Args:
            input_noise: random noise, of the same shape as the desired latent representation.
            autoencoder_model: first stage model.
            diffusion_model: model to sample from.
            device: CUDA or CPU device
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler.
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """

        unconditioning = torch.ones_like(conditioning) * (-1.0)
        conditioning = torch.cat([unconditioning, conditioning], dim = 0)

        image = input_noise
        if verbose and has_tqdm:
            progress_bar = tqdm(scheduler.timesteps)
        else:
            progress_bar = iter(scheduler.timesteps)
        intermediates = []
        for t in progress_bar:
            # GUID
            with torch.no_grad():
                input_im = torch.cat([image] * 2)
                model_output = diffusion_model(input_im, timesteps=torch.Tensor((t,)).to(device),
                                     context=conditioning)
                noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            image, _ = scheduler.step(noise_pred, t, image)
            if save_intermediates and t % intermediate_steps == 0:
                intermediates.append(image)

        latent = image
        latent_intermediates = [l.detach().cpu() for l in intermediates]
        latent = latent.detach().cpu()

        autoencoder_model = autoencoder_model.cpu()
        with torch.no_grad():
            if self.tf_crop is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_crop(latent.numpy())
                latent = latent.type(latent_type)

            image = autoencoder_model.decode_stage_2_outputs(latent / self.scale_factor)
            latent = latent.detach().cpu()
        if save_intermediates:
            intermediates = []

            for latent_intermediate in latent_intermediates:
                with torch.no_grad():
                    if self.tf_crop is not None:
                        latent_intermediate_ = latent_intermediate
                        latent_intermediate_type = deepcopy(latent_intermediate_.type())
                        latent_intermediate_ = self.tf_crop(latent_intermediate_.numpy())
                        latent_intermediate_ = latent_intermediate_.type(latent_intermediate_type) #.to(device)
                    else:
                        latent_intermediate_ = latent_intermediate

                    intermediates.append(
                        autoencoder_model.decode_stage_2_outputs(latent_intermediate_
                                                                 / self.scale_factor) #.detach().cpu()
                    )

                    latent_intermediate_ = latent_intermediate_ #.detach().cpu()

            autoencoder_model = autoencoder_model.to(device)
            return image, intermediates

        else:
            autoencoder_model = autoencoder_model.to(device)
            return image

    @torch.no_grad()
    def get_likelihood(
        self,
        inputs: torch.Tensor,
        autoencoder_model: Callable[..., torch.Tensor],
        diffusion_model: Callable[..., torch.Tensor],
        scheduler: Optional[Callable[..., torch.Tensor]] = None,
        save_intermediates: Optional[bool] = False,
        conditioning: Optional[torch.Tensor] = None,
        original_input_range: Optional[Tuple] = (0, 255),
        scaled_input_range: Optional[Tuple] = (0, 1),
        verbose: Optional[bool] = True,
        resample_latent_likelihoods: Optional[bool] = False,
        resample_interpolation_mode: Optional[str] = "bilinear",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Computes the likelihoods of the latent representations of the input.

        Args:
            inputs: input images, NxCxHxW[xD]
            autoencoder_model: first stage model.
            diffusion_model: model to compute likelihood from
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: save the intermediate spatial KL maps
            conditioning: Conditioning for network input.
            original_input_range: the [min,max] intensity range of the input data before any scaling was applied.
            scaled_input_range: the [min,max] intensity range of the input data after scaling.
            verbose: if true, prints the progression bar of the sampling process.
            resample_latent_likelihoods: if true, resamples the intermediate likelihood maps to have the same spatial
                dimension as the input images.
            resample_interpolation_mode: if use resample_latent_likelihoods, select interpolation 'nearest' or 'bilinear'
        """

        latents = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor
        if self.tf_pad is not None:
            latents = self.tf_pad(latents)

        outputs = super().get_likelihood(
            inputs=latents,
            diffusion_model=diffusion_model,
            scheduler=scheduler,
            save_intermediates=save_intermediates,
            conditioning=conditioning,
            verbose=verbose,
        )
        if save_intermediates and resample_latent_likelihoods:
            intermediates = outputs[1]
            from torchvision.transforms import Resize

            interpolation_modes = {"nearest": 0, "bilinear": 2}
            if resample_interpolation_mode not in interpolation_modes.keys():
                raise ValueError(
                    f"resample_interpolation mode should be either nearest or bilinear, not {resample_interpolation_mode}"
                )
            resizer = Resize(size=inputs.shape[2:], interpolation=interpolation_modes[resample_interpolation_mode])
            intermediates = [resizer(x) for x in intermediates]
            outputs = (outputs[0], intermediates)
        return outputs

    @torch.no_grad()
    def decode(self,
               latents: torch.Tensor,
               autoencoder_model: Callable[..., torch.Tensor],
               device: _device):

        """
        For predicted noise on samples (from __call__) decodes predictions.
        :param autoencoder_model:
        :param latents:
        :param device: CUDA or CPU device
        :return:
        """

        with torch.no_grad():
            if self.tf_crop is not None:
                latents = latents.detach().cpu()
                latent_type = deepcopy(latents.type())
                latents = self.tf_crop(latents.numpy())
                latents = latents.type(latent_type).to(device)

            image = autoencoder_model.decode_stage_2_outputs(latents / self.scale_factor)

        return image

    @torch.no_grad()
    def noise_and_denoise(self, inputs: torch.Tensor,
                          unet: Callable[..., torch.Tensor],
                          autoencoder_model: Callable[..., torch.Tensor],
                          timesteps: torch.Tensor,
                          noise: torch.Tensor,
                          conditioning: Union[torch.Tensor, None],
                          device: _device):

        with torch.no_grad():
            latent = autoencoder_model.encode_stage_2_inputs(inputs) * self.scale_factor
            if self.tf_pad is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_pad(latent.numpy())
                latent = latent.type(latent_type).to(device)

        noised_latent = self.scheduler.add_noise(latent, noise, timesteps)
        denoised_latents = []
        for t_ind, t in enumerate(timesteps):
            denoised_latent = noised_latent[t_ind, ...].unsqueeze(0)
            for t_rev in reversed(range(0, t)):
                noise_output = unet(denoised_latent,
                                    timesteps=torch.Tensor((t_rev, )).to(noise.device),
                                    context = conditioning[t_ind, ...].unsqueeze(0))
                denoised_latent, _ = self.scheduler.step(noise_output, t_rev, denoised_latent)
            denoised_latents.append(denoised_latent)

        denoised_latents = torch.cat(denoised_latents, 0)
        if self.tf_crop is not None:
            denoised_latents = denoised_latents.detach().cpu()
            denoised_latent_type = deepcopy(denoised_latents.type())
            denoised_latents = self.tf_crop(denoised_latents.numpy())
            denoised_latents = denoised_latents.type(denoised_latent_type).to(device)


        return autoencoder_model.decode_stage_2_outputs(denoised_latents / self.scale_factor), latent, denoised_latents

    def get_velocity(self, vae, images, noise, timesteps, device):

        with torch.no_grad():
            if self.wrapped_vae:
                latent = vae(images)
                print(len(latent))
                for l_ in latent:
                    print(l_.shape)
                latent = latent * self.scale_factor
            else:
                latent = vae.encode_stage_2_inputs(images) * self.scale_factor
            if self.tf_pad is not None:
                latent = latent.detach().cpu()
                latent_type = deepcopy(latent.type())
                latent = self.tf_pad(latent.numpy())
                latent = latent.type(latent_type).to(device)

        target = self.scheduler.get_velocity(latent, noise, timesteps)
        return target


    def sample_multiple_steps(self,
                              input_noise: torch.Tensor,
                              autoencoder_model: Callable[..., torch.Tensor],
                              diffusion_model: Callable[..., torch.Tensor],
                              scheduler: Optional[Callable[..., torch.Tensor]] = None,
                              save_intermediates: Optional[bool] = False,
                              how_many_intermediates: Optional[int] = 5,
                              conditioning: Optional[torch.Tensor] = None,
                              guidance_scale: int = 7.0):


        """
        Args:
            input_noise: random noise, of the same shape as the desired sample.
            diffusion_model: model to sample from.
            scheduler: diffusion scheduler. If none provided will use the class attribute scheduler
            save_intermediates: whether to return intermediates along the sampling change
            intermediate_steps: if save_intermediates is True, saves every n steps
            conditioning: Conditioning for network input.
            verbose: if true, prints the progression bar of the sampling process.
        """

        image = input_noise
        if has_tqdm:
            progress_bar = tqdm(self.scheduler.timesteps)
        else:
            progress_bar = iter(self.scheduler.timesteps)

        intermediates = {}
        straight_from_decoder = {}
        to_plot = self.scheduler.timesteps[0::int(len(self.scheduler.timesteps)/how_many_intermediates)]
        for t in progress_bar:
            with torch.no_grad():
                if guidance_scale > 0:
                    input_im = torch.cat([image] * 2)
                    input_conditioning = torch.cat([torch.zeros_like(conditioning), conditioning], 0)
                else:
                    input_im = image
                    input_conditioning = conditioning

                model_output = diffusion_model(
                    input_im, timesteps=torch.Tensor((t,)).to(input_im.device), context=input_conditioning,
                )

                if guidance_scale > 0:
                    noise_pred_uncond, noise_pred_cond = model_output.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = model_output


                if t in to_plot:
                    # We predict the image DIRECTLY from this step.
                    alpha_prod_t = torch.gather(self.scheduler.alphas_cumprod, 0, torch.tensor([t]*3)).to(image.device)
                    beta_prod_t = 1 - alpha_prod_t
                    if self.scheduler.prediction_type == "epsilon":
                        pred_seg = (image - beta_prod_t[:, None, None, None, None].expand(noise_pred.shape) \
                                    ** (0.5) * noise_pred) / alpha_prod_t[:, None, None, None, None].expand(noise_pred.shape) \
                                    ** (0.5)
                    elif self.scheduler.prediction_type == "v_prediction":
                        pred_seg = (alpha_prod_t[:, None, None, None, None].expand(noise_pred.shape) ** 0.5) * \
                                   image - (beta_prod_t[:, None, None, None, None].expand(noise_pred.shape) ** 0.5) * \
                                   noise_pred
                    if self.tf_crop is not None:
                        pred_seg = self.tf_crop(pred_seg)

                    pred_seg = autoencoder_model.decode_stage_2_outputs(pred_seg / self.scale_factor).detach().cpu()
                    straight_from_decoder[t] = pred_seg

                image, _ = scheduler.step(noise_pred, t, image)

                # Save intermediates as well.
                if save_intermediates and t in to_plot:
                    if self.tf_crop is not None:
                        int_ = self.tf_crop(image)
                    else:
                        int_ = image
                    intermediates[t] = autoencoder_model.decode_stage_2_outputs(int_ / self.scale_factor).detach().cpu()

        return intermediates, straight_from_decoder, image