import sys
sys.path.append("/ws_virginia/allcode/brainSPADE3D/")
sys.path.append("/ws_virginia/allcode/brainSPADE3D/GenerativeModels/")
sys.path.append("/ws_virginia/allcode/brainSPADE3D/GenerativeModels/generative/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE3D/brainSPADE3D/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE3D/brainSPADE3D/GenerativeModels/")
sys.path.append("/nfs/home/vfernandez/models/brainSPADE3D/brainSPADE3D/GenerativeModels/generative/")
import os
import torch
from monai.utils import set_determinism
from tqdm import tqdm
from moreutils import saveNiiGrid
from conditioned_ldm_3D.sizeable_inferer import SizeableInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler
import argparse
import numpy as np
import shutil
import mlflow.pytorch
import nibabel as nib
from pathlib import Path
from tensorboardX import SummaryWriter
from omegaconf import OmegaConf
from conditioned_ldm_3D.labelgen_utils import log_3d_img, log_mlflow, TimeStepManager, set_lr, getLearningRate, pad_latent, get_lr
from conditioned_ldm_3D.labelgen_data_utils import get_training_loaders
import torch.optim as optim
import monai
from torch.cuda.amp import GradScaler, autocast
import subprocess
from monai.config import print_config
from pynvml.smi import nvidia_smi
from conditioned_ldm_3D.wrappers import Stage1Wrapper
# Filter Warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import torch.nn.functional as F

def print_gpu_memory_report():
    if torch.cuda.is_available():
        nvsmi = nvidia_smi.getInstance()
        data = nvsmi.DeviceQuery("memory.used, memory.total, utilization.gpu")["gpu"]
        print("Memory report")
        for i, data_by_rank in enumerate(data):
            mem_report = data_by_rank["fb_memory_usage"]
            print(f"gpu:{i} mem(%) {int(mem_report['used'] * 100.0 / mem_report['total'])}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2, help="Random seed to use.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size."),
    parser.add_argument("--config_file", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--data_dicts", type=str, help = "Path to the TSV containing paths to images")
    parser.add_argument("--data_dicts_val", type = str, help = "Path to the TSV containing paths to validation images")
    parser.add_argument("--checkpoints_dir", type=str, help = "Where do you want to save the models and logs.")
    parser.add_argument("--project_url", type=str, default="/project",
                        help="Path to the project where things will be saved.")
    parser.add_argument("--num_epochs", type = int, default = 100, help = "Number of epochs")
    parser.add_argument("--num_workers", type=int, default=3, help="Number of threads")
    parser.add_argument("--validation_epochs", type = int, default=10, help="After how many epochs we validate")
    parser.add_argument("--save_every", type = str, default = 5, help = "Every how many epochs we save")
    parser.add_argument("--augmentation", type=int, default=0, help = "Use augmentation.")
    parser.add_argument("--conditionings", type = str, default="",  help = "List containing the "
                                                                                        "conditionings")
    parser.add_argument("--vae_url", type=str, required=True, help = "Path to the VAE or VQVAE model folder within"
                                                                     "the mlflow saving folder.")
    parser.add_argument("--max_size", type=int, default=None, help = "Max dataset size for training")
    parser.add_argument("--no_scheduler", type=int, default=0, help = "If 1, deactivates LR scheduler")
    parser.add_argument("--dp_vae", type=int, default=1, help= "If 0, VAE is kept in CPU (less memory)"
                                                                        "but the training will be slower. ")
    parser.add_argument("--use_guidance", type=int, default=0, help = "Use guidance for conditioning.")
    parser.add_argument("--use_focal_loss", type=float, default=1.0, help="Use focal loss on LDM. ")
    args = parser.parse_args()
    return args

def main(args):
    print_config()
    # Quartiles
    LESION_QUARTILES = {'wmh': (0.0097, 0.0923),
                        'tumour': (0.0224, 0.1833),
                        'edema': (0.1408, 0.3595),
                        'gdtumour': (0.0712, 0.4280)}

    set_determinism(42)

    # Conditionings
    conditionings = args.conditionings.split("-")

    # Set outputs directory (mlflow)
    output_dir = Path(f"{args.project_url}/outputs/runs/")
    output_dir.mkdir(exist_ok=True, parents=True)

    run_dir = Path(args.checkpoints_dir) #
    # Load old model
    if run_dir.exists() and ((run_dir / "checkpoint.pth").exists() or (run_dir / "checkpoint_raw.pth").exists()):
        resume = True
    else:
        resume = False
        run_dir.mkdir(exist_ok=True)

    if not os.path.isdir(os.path.join(args.checkpoints_dir, 'train_imgs')):
        os.makedirs(os.path.join(args.checkpoints_dir, 'train_imgs'))
    if not os.path.isdir(os.path.join(args.checkpoints_dir, 'val_imgs')):
        os.makedirs(os.path.join(args.checkpoints_dir, 'val_imgs'))

    # Print arguments.
    print(f"Run directory: {str(run_dir)}")
    print(f"Arguments: {str(args)}")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    print("Loading configuration...")
    config = OmegaConf.load(args.config_file)

    print("Getting data...")

    # Check if there are conditioning elements
    if len(conditionings) == 0 and config['ldm']['params']['ddpm']['with_conditioning']:
        config['ldm']['params']['ddpm']['with_conditioning'] = False
        print("Length of conditionings was zero: changing flag 'with_conditioning' from the config file to False.")
    else:
        config['ldm']['params']['ddpm']['with_conditioning'] = True

    # Load dataset
    train_loader, val_loader = get_training_loaders(
        batch_size=args.batch_size,
        training_ids=args.data_dicts,
        spatial_size=config['ldm']['resolution'],
        validation_ids=args.data_dicts_val,
        augmentation=bool(args.augmentation),
        num_workers=args.num_workers,
        conditionings=conditionings,
        cache_dir=args.checkpoints_dir,
        max_size=args.max_size,
        even_brats = False,
        for_ldm=True,
    )

    # Load VAE
    print(f"Loading VQ-VAE/VAE from {args.vae_url}")
    vae = mlflow.pytorch.load_model(args.vae_url)
    vae.eval()

    print("Creating models...")
    unet = DiffusionModelUNet(
        **config['ldm']['params']['unet_config']['params']
    )

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule=config['ldm']['params']['ddpm']['beta_schedule'],
        beta_start=config['ldm']['params']['ddpm']['beta_start'],
        beta_end=config['ldm']['params']['ddpm']['beta_end'],
        prediction_type=config['ldm']['params']['ddpm']['prediction_type'],
        clip_sample=False,
    )

    time_step_manager = TimeStepManager(
        scheduler=scheduler,
        num_train_timesteps=scheduler.num_train_timesteps,
        use_loss_buffer=True,
        loss_buffer_interval=config['ldm']['params']['buffer_interval'],
        use_loss_buffer_interval_from=500)

    # Check whether latent space needs padding: if the VAE latent space cannot be downsampled as many times needed
    # for the LDM, we pad it before passing it to it, and crop it before passing it to the VAE.
    z_shape_vae = [vae.latent_channels] + [s // (2 ** (len(vae.decoder.num_channels) -1)) for s in config['ldm']['resolution']]
    need_adjust, z_shape_ldm = pad_latent(z_shape_vae,  len(config['ldm']['params']['unet_config']['params']
                                                            ['num_channels']))
    z_shape_ldm = [args.batch_size] + z_shape_ldm

    inferer = SizeableInferer(scheduler=scheduler, scale_factor=config['ldm']['params']['scale_factor'],
                              latent_shape_vae=z_shape_vae, latent_shape_ldm=z_shape_ldm[1:],
                              wrapped_vae=torch.cuda.device_count() > 1 and args.dp_vae)

    # Loss
    w = np.ones(10)
    w[6] = 1
    fl = monai.losses.FocalLoss(include_background=True, gamma = 3, weight=w, reduction="none")

    # Define optimizer
    optimizer = optim.Adam(unet.parameters(), lr=config["ldm"]["warmup_lr"])
    if args.no_scheduler == 1:
        scheduler_optimizer = None
    else:
        scheduler_optimizer = getLearningRate(optimizer,
                                    warmup_lr=config["ldm"]["warmup_lr"],
                                    base_lr=config["ldm"]["base_lr"],
                                    n_epochs_warmup=config["ldm"]["epochs_warmup"],
                                    n_epochs_shift=config["ldm"]["epochs_shift"],
                                    n_epochs=args.num_epochs,
                                    lr_scheduler_type=config["ldm"]["type_lr"],
                                    end_lr=config["ldm"]["end_lr"],
                              )

    # Get Checkpoint
    best_loss = float("inf")
    start_epoch = 0

    # Data Parallel
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    device = torch.device("cuda")
    if torch.cuda.device_count() > 1:
        unet = torch.nn.DataParallel(unet)
        if args.dp_vae:
            vae_w = Stage1Wrapper(model=vae)
            vae_w.eval()
            vae_w = torch.nn.DataParallel(vae_w)
        else:
            vae_w = vae.eval()
    else:
        vae_w = vae.eval()
    vae_w = vae_w.to(device)

    # Send to device
    unet = unet.to(device)

    if resume:
        print(f"Using checkpoint!")
        if torch.cuda.device_count() <= 1:
            checkpoint = torch.load(str(run_dir / "checkpoint_raw.pth"))
        else:
            checkpoint = torch.load(str(run_dir / "checkpoint.pth"))
        unet.load_state_dict(checkpoint['unet'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if args.no_scheduler == 0:
            scheduler_optimizer.load_state_dict(checkpoint['scheduler_optimizer'])
        else:
            optimizer = set_lr(optimizer, config["ldm"]["base_lr"])
        time_step_manager = checkpoint['time_step_manager']
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
    else:
        if args.no_scheduler != 0:
            optimizer = set_lr(optimizer, config["ldm"]["base_lr"])
        print(f"No checkpoint found.")

    # Summary writers
    writer_train = SummaryWriter(log_dir=str(run_dir / "train"))
    writer_val = SummaryWriter(log_dir=str(run_dir / "val"))

    # Training
    dm = monai.metrics.DiceMetric(include_background=True)
    epoch_loss_list = []
    epoch_ds_list = []
    focal_loss_list = []
    l1_loss_list = []
    scaler = GradScaler()
    stop_flag = False
    for epoch in range(start_epoch, args.num_epochs):
        print("epoch %d/%d" %(epoch, args.num_epochs))
        if stop_flag:
            print("ending training 'cause of NaN loss")
            break
        unet.train().to(device)
        epoch_loss = 0
        l1_loss = 0
        focal_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=135)
        progress_bar.set_description(f"Epoch {epoch}")
        plot_item = 1000000# np.random.choice(len(train_loader))
        for step, batch in progress_bar:
            images = batch["label"].to(device)
            if args.use_guidance >= 1:
                for b_el in range(images.shape[0]):
                    if np.random.uniform() > 0.8:
                        for cond in conditionings:
                            batch[cond][b_el] = -1.0

            if True in torch.isnan(images) or True in torch.isinf(images) or images.max()>1000:
                print("NaN or inf found in images")
                print(batch['label_meta_dict']['filename_or_obj'])
                stop_flag = True
                break
            cond_list = []
            cond_names = []
            for c in conditionings:
                cond_list.append(batch[c])
                cond_names.append(c)
            cond = torch.tensor(torch.stack(cond_list, -1)).unsqueeze(1).to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=False):
                timesteps = time_step_manager.selectTimeSteps(epoch=epoch, batch_size=images.shape[0],
                                                              device=images.device).long()
                noise = torch.randn([images.shape[0]] + z_shape_ldm[1:]).to(device)
                if config['ldm']['params']['ddpm']['prediction_type'] == 'v_prediction':
                    target = inferer.get_velocity(vae_w, images, noise, timesteps, device)
                else:
                    target = noise

                noise_pred, pred_seg = inferer(
                    inputs=images, autoencoder_model=vae_w,
                    diffusion_model=unet, noise=target,
                    device = device,
                    timesteps=timesteps,
                    condition = cond,
                    focal_loss_weight=args.use_focal_loss
                )

                if True in torch.isnan(noise_pred) or True in torch.isinf(noise_pred):
                    print("NaN found on predicting noise")
                    stop_flag = True
                    break

                #loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="none")
                #loss = F.mse_loss(noise_pred.float(), target.float())
                # loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none") + \
                #        0.5 * F.l1_loss(noise_pred.float(), target.float(), reduction="none")
                l1_loss_ = F.l1_loss(noise_pred.float(), target.float(), reduction="none")
                l1_loss_ = l1_loss_.view(l1_loss_.shape[0], -1).mean(dim=-1)

                if args.use_focal_loss > 0:
                    pred_seg = torch.softmax(pred_seg, 1)
                    # calculate the dice loss on these predictions
                    focal_loss_ = args.use_focal_loss * fl(input=pred_seg, target=images)
                    focal_loss_ = focal_loss_.view(focal_loss_.shape[0], -1).mean(dim=-1)
                    loss = focal_loss_ + l1_loss_
                else:
                    loss = l1_loss_
                    focal_loss_ = torch.Tensor([0.0])

                if True in torch.isnan(loss):
                    print("NaN loss straight on ")
                    stop_flag = True
                    break

                time_step_manager.manageLossBuffer(loss, timesteps)
                loss = loss.mean()
                l1_loss_ = l1_loss_.mean()
                focal_loss_ = focal_loss_.mean()
                if True in torch.isnan(loss):
                    print("NaN loss after TSM")
                    stop_flag = True
                    break
                scaler.scale(loss).backward()
                #torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if True in torch.isnan(loss):
                    print("NaN loss after scaler update")
                    stop_flag = True
                    break
                epoch_loss += loss.item()
                l1_loss += l1_loss_.item()
                focal_loss += focal_loss_.item()
                progress_bar.set_postfix(
                    {
                        "loss": epoch_loss / (step + 1),
                        "l1": l1_loss / (step + 1),
                        "fl": focal_loss / (step + 1),
                        "lr": get_lr(optimizer)
                    }
                )
                if step == plot_item and epoch % args.validation_epochs == 0:
                    with torch.no_grad():
                        denoised_images, latents, denoised_latents = inferer.noise_and_denoise(
                            images, unet, vae, timesteps, noise, cond, device,
                        )
                        denoised_latents = denoised_latents.detach().cpu().type(torch.float32)
                        latents = latents.detach().cpu().type(torch.float32)
                        denoised_images = denoised_images.detach().cpu().type(torch.float32)
                        denoised_images = torch.softmax(denoised_images, 1)
                        latents = latents.detach().cpu()[:, denoised_latents.shape[1]//2, ...]
                        denoised_latents = denoised_latents.detach().cpu()[:, denoised_latents.shape[1]//2, ...]
                        images = images.detach().cpu()
                        dice_score = dm(denoised_images, images).numpy()
                        dice_score = dice_score[~np.isnan(dice_score)]
                        epoch_ds_list.append(dice_score.mean())
                        grid = saveNiiGrid(torch.cat([images, denoised_images], 0),
                                           grid_shape=[images.shape[0], 2])
                        grid_ni = nib.Nifti1Image(grid.numpy(), np.eye(4))
                        time_steps = "-".join([str(t.detach().cpu().numpy()) for t in list(timesteps)])
                        nib.save(grid_ni, os.path.join(args.checkpoints_dir, 'train_imgs',
                                                       'train_img_%d_%s.nii.gz' %(epoch, time_steps)))
                        grid = saveNiiGrid(torch.cat([denoised_latents, latents], 0),
                                           grid_shape=[latents.shape[0], 2])
                        grid_ni = nib.Nifti1Image(grid.numpy(), np.eye(4))
                        time_steps = "-".join([str(t.detach().cpu().numpy()) for t in list(timesteps)])
                        nib.save(grid_ni, os.path.join(args.checkpoints_dir, 'train_imgs',
                                                       'train_latent_img_%d_%s.nii.gz' %(epoch, time_steps)))
                del noise_pred, noise

        epoch_loss_list.append(epoch_loss / (step + 1))
        focal_loss_list.append(focal_loss / (step + 1))
        l1_loss_list.append(l1_loss / (step + 1))

        # Validation
        if epoch % args.validation_epochs == 0:
            with torch.no_grad():
                with autocast(enabled=False):
                    raw_model = unet.module if hasattr(unet, "module") else unet
                    cond_list = []
                    cond_names = []
                    for c in conditionings:
                        if np.random.uniform() > 0.0:
                            cond_list.append(np.random.uniform(LESION_QUARTILES[c][0],
                                                               LESION_QUARTILES[c][1],
                                                               args.batch_size))
                        else:
                            cond_list.append(np.asarray([0.0]*args.batch_size))
                        cond_names.append(c)
                    cond = torch.from_numpy(np.stack(cond_list, -1)).unsqueeze(1).type(torch.float).to(device)
                    raw_model.eval()
                    noise = torch.randn(z_shape_ldm).to(device)
                    if args.use_guidance > 0:
                        samples, ints = inferer.sample_wguid(input_noise=noise,
                                                       autoencoder_model=vae,
                                                       diffusion_model=raw_model,
                                                       scheduler=scheduler,
                                                       conditioning=cond,
                                                       save_intermediates=True,
                                                       intermediate_steps=200,
                                                       device=device,
                                                             guidance_scale=args.use_guidance)
                    else:
                        samples, ints = inferer.sample(input_noise=noise,
                                                       autoencoder_model=vae,
                                                       diffusion_model=raw_model,
                                                       scheduler=scheduler,
                                                       conditioning=cond,
                                                       save_intermediates=True,
                                                       intermediate_steps=200,
                                                       device = device)
                    samples = torch.softmax(samples, 1)
                    for ind, i in enumerate(ints):
                        ints[ind] = torch.softmax(i, 1)
                    log_3d_img(rec_imgs=samples,
                               cond_list=cond_names,
                               gt_imgs=None,
                               writer=writer_val,
                               step=epoch)

                    # Intermediates
                    samples = samples
                    if samples.shape[0] > 3:
                        samples = samples[:4, ...]
                        ints = [ints_[:4, ...] for ints_ in ints]

                    out_grid_tensor = []
                    for b in range(samples.shape[0]):
                        for int_ in ints:
                            out_grid_tensor.append(int_[b,...])
                        out_grid_tensor.append(samples[b, ...])
                    out_grid_tensor = torch.stack(out_grid_tensor, 0)
                    grid = saveNiiGrid(out_grid_tensor, grid_shape=[samples.shape[0], len(ints)+1]).numpy().astype('uint8')
                    grid = nib.Nifti1Image(grid, affine = np.eye(4))
                    nib.save(grid, os.path.join(args.checkpoints_dir, 'val_imgs', 'val_%d.nii.gz' %epoch))

        # Check loss
        if len(epoch_loss_list) > 0:
            if epoch_loss_list[-1] <= best_loss:
                print(f"New best val loss {epoch_loss_list[-1]}")
                best_loss = epoch_loss_list[-1]
                raw_model = unet.module if hasattr(unet, "module") else unet
                torch.save(raw_model.state_dict(), os.path.join(args.checkpoints_dir , "best_model.pth"))

        # Save model
        checkpoint = {
            "epoch": epoch + 1,
            "unet": unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "time_step_manager": time_step_manager
        }

        checkpoint_raw = {
            "epoch": epoch + 1,
            "unet": unet.module.state_dict() if hasattr(unet, "module") else unet.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_loss": best_loss,
            "time_step_manager": time_step_manager
        }

        if args.no_scheduler == 1:
            checkpoint["scheduler_optimizer"] = None
            checkpoint_raw["scheduler_optimizer"]=None
        else:
            checkpoint["scheduler_optimizer"] = scheduler_optimizer.state_dict()
            checkpoint_raw["scheduler_optimizer"] = scheduler_optimizer.state_dict()


        torch.save(checkpoint, os.path.join(args.checkpoints_dir, "checkpoint.pth"))
        torch.save(checkpoint_raw, os.path.join(args.checkpoints_dir, "checkpoint_raw.pth"))

        if epoch%10 == 2:
            torch.save(checkpoint, os.path.join(args.checkpoints_dir, "checkpoint_%d.pth" %epoch))
        # Scheduler
        if not args.no_scheduler == 1:
            scheduler_optimizer.step()
        # Writer
        if len(epoch_loss_list) > 0:
            writer_train.add_scalar("loss", epoch_loss_list[-1], global_step = epoch)
        if len(epoch_ds_list)>0:
            writer_train.add_scalar("dice", epoch_ds_list[-1], global_step = epoch)
        if len(focal_loss_list)>0:
            writer_train.add_scalar("focal", focal_loss_list[-1], global_step=epoch)

        # Save nt
        torch.save(checkpoint, os.path.join(args.checkpoints_dir, "checkpoint.pth"))
        # Print memory report
        print_gpu_memory_report()

    print(f"Training finished!")
    print(f"Saving final model...")
    raw_model = unet.module if hasattr(unet, "module") else unet
    torch.save(raw_model.state_dict(), os.path.join(args.checkpoints_dir,"final_model.pth"))
    print("Logging mlflow details...")
    log_mlflow(
        model=raw_model,
        config=config,
        args=args,
        experiment="ldm",
        run_dir=run_dir,
        val_loss=epoch_loss_list[-1],
    )

    if os.path.isdir(os.path.join(args.checkpoints_dir, 'cache')):
        shutil.rmtree(os.path.join(args.checkpoints_dir, 'cache'))

args = parse_args()
subprocess.Popen(args=['set_mlflow.sh'], shell=True)
main(args)