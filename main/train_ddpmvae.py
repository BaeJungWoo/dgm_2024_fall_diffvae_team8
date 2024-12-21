# import copy
# import logging
# import os

# import hydra
# import pytorch_lightning as pl
# from omegaconf import OmegaConf
# from pytorch_lightning.callbacks import ModelCheckpoint
# from pytorch_lightning.utilities.seed import seed_everything
# from torch.utils.data import DataLoader

# from models.callbacks import EMAWeightUpdate
# from models.diffusion import (
#     DDPM,
#     DDPMv2,
#     DDPMWrapper,
#     DDPMWrapper_new,
#     SuperResModel,
#     UNetModel,
#     UNet,
#     SuperResModelv2,
# )
# from models.vae import VAE
# from util import configure_device, get_dataset

# logger = logging.getLogger(__name__)


# def __parse_str(s):
#     split = s.split(",")
#     return [int(s) for s in split if s != "" and s is not None]


# @hydra.main(config_path="configs")
# def train(config):
#     # Get config and setup
#     config = config.dataset
#     logger.info(OmegaConf.to_yaml(config))

#     # Set seed
#     seed_everything(config.ddpm.training.seed, workers=True)

#     # Dataset
#     root = config.ddpm.data.root
#     d_type = config.ddpm.data.name
#     image_size = config.ddpm.data.image_size
#     dataset = get_dataset(
#         d_type, root, image_size, norm=config.ddpm.data.norm, flip=config.ddpm.data.hflip
#     )
#     N = len(dataset)
#     batch_size = config.ddpm.training.batch_size
#     batch_size = min(N, batch_size)

#     # Model
#     lr = config.ddpm.training.lr
#     attn_resolutions = __parse_str(config.ddpm.model.attn_resolutions)
#     dim_mults = __parse_str(config.ddpm.model.dim_mults)
#     ddpm_type = config.ddpm.training.type

#     # Use the superres model for conditional training
#     decoder_cls = UNet if ddpm_type == "uncond" else SuperResModelv2
#     print(f'Using model type: {decoder_cls} with ddpm_type={ddpm_type}')
#     decoder = decoder_cls(
#         in_channels=config.ddpm.data.n_channels,
#         resolution=image_size,
#         model_channels=config.ddpm.model.dim,
#         out_channels=3,
#         num_res_blocks=config.ddpm.model.n_residual,
#         attention_resolutions=attn_resolutions,
#         dropout=config.ddpm.model.dropout,
#         channel_mult=dim_mults,
#         num_heads=config.ddpm.model.n_heads,
#         z_dim=config.ddpm.training.z_dim,
#         use_scale_shift_norm=config.ddpm.training.z_cond,
#         use_z=config.ddpm.training.z_cond,
#     )

#     # EMA parameters are non-trainable
#     ema_decoder = copy.deepcopy(decoder)
#     for p in ema_decoder.parameters():
#         p.requires_grad = False

#     ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
#     online_ddpm = ddpm_cls(
#         decoder,
#         beta_1=config.ddpm.model.beta1,
#         beta_2=config.ddpm.model.beta2,
#         T=config.ddpm.model.n_timesteps,
#     )
#     target_ddpm = ddpm_cls(
#         ema_decoder,
#         beta_1=config.ddpm.model.beta1,
#         beta_2=config.ddpm.model.beta2,
#         T=config.ddpm.model.n_timesteps,
#     )
#     vae = VAE(
#         input_res=image_size,
#         enc_block_str=config.vae.model.enc_block_config,
#         dec_block_str=config.vae.model.dec_block_config,
#         enc_channel_str=config.vae.model.enc_channel_config,
#         dec_channel_str=config.vae.model.dec_channel_config,
#         lr=config.vae.training.lr,
#         alpha=config.vae.training.alpha,
#     )

#     assert isinstance(online_ddpm, ddpm_cls)
#     assert isinstance(target_ddpm, ddpm_cls)
#     logger.info(f"Using DDPM with type: {ddpm_cls} and data norm: {config.ddpm.data.norm}")

#     ddpm_wrapper = DDPMWrapper_new(
#         online_ddpm,
#         target_ddpm,
#         vae,
#         lr=lr,
#         cfd_rate=config.ddpm.training.cfd_rate,
#         n_anneal_steps=config.ddpm.training.n_anneal_steps,
#         loss=config.ddpm.training.loss,
#         conditional=False if ddpm_type == "uncond" else True,
#         grad_clip_val=config.ddpm.training.grad_clip,
#         z_cond=config.ddpm.training.z_cond,
#     )


# ###########################
#     # Trainer
#     train_kwargs = {}
#     restore_path = config.ddpm.training.restore_path
#     if restore_path != "":
#         # Restore checkpoint
#         train_kwargs["resume_from_checkpoint"] = restore_path

#     # Setup callbacks
#     results_dir = config.ddpm.training.results_dir
#     chkpt_callback = ModelCheckpoint(
#         dirpath=os.path.join(results_dir, "checkpoints"),
#         filename=f"ddpmrepro-{config.ddpm.training.chkpt_prefix}"
#         + "-{epoch:02d}-{loss:.4f}",
#         every_n_epochs=config.ddpm.training.chkpt_interval,
#         save_on_train_epoch_end=True,
#     )

#     #VAE saved
#     results_dir = config.vae.training.results_dir
#     chkpt_callback_vae = ModelCheckpoint(
#         dirpath=os.path.join(results_dir, "checkpoints"),
#         filename=f"vae-{config.vae.training.chkpt_prefix}"
#         + "-{epoch:02d}-{train_loss:.4f}",
#         every_n_epochs=config.vae.training.chkpt_interval,
#         save_on_train_epoch_end=True,
#     )

#     train_kwargs["default_root_dir"] = results_dir
#     train_kwargs["max_epochs"] = config.ddpm.training.epochs
#     train_kwargs["log_every_n_steps"] = config.ddpm.training.log_step
#     train_kwargs["callbacks"] = [chkpt_callback, chkpt_callback_vae]

#     if config.ddpm.training.use_ema:
#         ema_callback = EMAWeightUpdate(tau=config.ddpm.training.ema_decay)
#         train_kwargs["callbacks"].append(ema_callback)

#     device = config.ddpm.training.device
#     loader_kws = {}
#     if device.startswith("gpu"):
#         _, devs = configure_device(device)
#         train_kwargs["gpus"] = devs

#         # Disable find_unused_parameters when using DDP training for performance reasons
#         from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin

#         train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
#         loader_kws["persistent_workers"] = True
#     elif device == "tpu":
#         train_kwargs["tpu_cores"] = 8

#     # Half precision training
#     if config.ddpm.training.fp16:
#         train_kwargs["precision"] = 16

#     # Loader
#     loader = DataLoader(
#         dataset,
#         batch_size,
#         num_workers=config.ddpm.training.workers,
#         pin_memory=True,
#         shuffle=True,
#         drop_last=True,
#         **loader_kws,
#     )

#     # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
#     # train_kwargs["gradient_clip_val"] = config.training.grad_clip

#     logger.info(f"Running Trainer with kwargs: {train_kwargs}")
#     trainer = pl.Trainer(**train_kwargs)
#     trainer.fit(ddpm_wrapper, train_dataloader=loader)


# if __name__ == "__main__":
#     train()





import copy
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
import torch
from models.callbacks import EMAWeightUpdate
from models.diffusion import (
    DDPM,
    DDPMv2,
    DDPMWrapper,
    DDPMWrapper_new2,
    DDPMWrapper_new3,
    SuperResModel,
    UNetModel,
    UNet,
    SuperResModelv2,
)
from models.vae import VAE
from util import configure_device, get_dataset

logger = logging.getLogger(__name__)

class VAECheckpointCallback(pl.Callback):
    def __init__(self, vae, save_dir, filename="vae_checkpoint_epoch{epoch:02d}.ckpt"):
        self.vae = vae
        self.save_dir = save_dir
        self.filename = filename

    def on_epoch_end(self, trainer, pl_module):
        # Save VAE state_dict at each epoch end
        if (trainer.current_epoch+1)%50==0 or trainer.current_epoch == 0:
            filepath = os.path.join(self.save_dir, self.filename.format(epoch=trainer.current_epoch))
            torch.save(self.vae.state_dict(), filepath)
            print(f"VAE checkpoint saved at {filepath}")

def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="configs")
def train(config):
    # Get config and setup
    config_vae = config.dataset.vae
    config = config.dataset.ddpm
    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    root = config.data.root
    d_type = config.data.name
    image_size = config.data.image_size
    dataset = get_dataset(
        d_type, root, image_size, norm=config.data.norm, flip=config.data.hflip
    )
    N = len(dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    # Model
    lr = config.training.lr
    attn_resolutions = __parse_str(config.model.attn_resolutions)
    dim_mults = __parse_str(config.model.dim_mults)
    ddpm_type = config.training.type

    # Use the superres model for conditional training
    decoder_cls = UNet if ddpm_type == "uncond" else SuperResModelv2
    print(f'Using model type: {decoder_cls} with ddpm_type={ddpm_type}')
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        resolution=image_size,
        model_channels=config.model.dim,
        out_channels=3,
        num_res_blocks=config.model.n_residual,
        attention_resolutions=attn_resolutions,
        dropout=config.model.dropout,
        channel_mult=dim_mults,
        num_heads=config.model.n_heads,
        z_dim=config.training.z_dim,
        use_scale_shift_norm=config.training.z_cond,
        use_z=config.training.z_cond,
    )

    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    ddpm_cls = DDPMv2 if ddpm_type == "form2" else DDPM
    online_ddpm = ddpm_cls(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = ddpm_cls(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )


    vae = VAE(
        input_res=image_size,
        enc_block_str=config_vae.model.enc_block_config,
        dec_block_str=config_vae.model.dec_block_config,
        enc_channel_str=config_vae.model.enc_channel_config,
        dec_channel_str=config_vae.model.dec_channel_config,
        lr=config_vae.training.lr,
        alpha=config_vae.training.alpha,
    )

    ddpm_wrapper = DDPMWrapper_new3(
        online_ddpm,
        target_ddpm,
        vae,
        lr=config.training.lr,
        vae_lr=config_vae.training.lr,
        cfd_rate=config.training.cfd_rate,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=False if ddpm_type == "uncond" else True,
        grad_clip_val=config.training.grad_clip,
        z_cond=config.training.z_cond,
    )

    # Trainer
    train_kwargs = {}
    restore_path = config.training.restore_path
    if restore_path != "":
        # Restore checkpoint
        train_kwargs["resume_from_checkpoint"] = restore_path

    # Setup callbacks
    results_dir = config.training.results_dir
    chkpt_callback_ddpm = ModelCheckpoint(
        dirpath=config.training.results_dir,
        filename="ddpm-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
    )

    vae_chkpt_callback = VAECheckpointCallback(
        vae=vae,
        save_dir=config_vae.training.results_dir,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback_ddpm, vae_chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    device = config.training.device
    loader_kws = {}
    if device.startswith("gpu"):
        _, devs = configure_device(device)
        train_kwargs["gpus"] = devs

        # Disable find_unused_parameters when using DDP training for performance reasons
        from pytorch_lightning.plugins import DDPPlugin, DDPSpawnPlugin

        train_kwargs["plugins"] = DDPPlugin(find_unused_parameters=False)
        loader_kws["persistent_workers"] = True
    elif device == "tpu":
        train_kwargs["tpu_cores"] = 8
    
    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    loader = DataLoader(
        dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )

    # Gradient Clipping by global norm (0 value indicates no clipping) (as in Ho et al.)
    # train_kwargs["gradient_clip_val"] = config.training.grad_clip

    logger.info(f"Running Trainer with kwargs: {train_kwargs}")
    trainer = pl.Trainer(**train_kwargs)
    trainer.fit(ddpm_wrapper, train_dataloader=loader)


if __name__ == "__main__":
    train()
