import pytorch_lightning as pl
import torch
import torch.nn as nn
from models.diffusion.spaced_diff import SpacedDiffusion
from models.diffusion.spaced_diff_form2 import SpacedDiffusionForm2
from models.diffusion.ddpm_form2 import DDPMv2
from util import space_timesteps

import math
class DDPMWrapper(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )

    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        cond = None
        z = None
        if self.conditional:
            x = batch
            with torch.no_grad():
                mu, logvar = self.vae.encode(x * 0.5 + 0.5)
                z = self.vae.reparameterize(mu, logvar)
                cond = self.vae.decode(z)
                cond = 2 * cond - 1

            # Set the conditioning signal based on clf-free guidance rate
            if torch.rand(1)[0] < self.cfd_rate:
                cond = torch.zeros_like(x)
                z = torch.zeros_like(z)
        else:
            x = batch

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )

        # Compute loss
        loss = self.criterion(eps, eps_pred)

        # Clip gradients and Optimize
        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()

        # Scheduler step
        lr_sched.step()
        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            recons = 2 * recons - 1

            mu, logvar = self.vae.encode(recons)
            uncertainty = torch.mean(torch.sigmoid(torch.exp(logvar)-2),dim=1,keepdim=True)
            recons = recons*(1-uncertainty) +  torch.randn_like(recons) * uncertainty
            
            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)

            # mu, logvar = self.vae.encode(x_t)
            # # logvar = torch.clamp(logvar, min=-50.0, max=50.0)

            # # Compute latent variance
            # latent_variance = torch.exp(logvar)

            # # Reconstruct image using VAE
            # recons = 2 * self.vae.decode(z) - 1

            # # Adaptive refinement weight based on variance
            # uncertainty = latent_variance.mean(dim=1, keepdim=True)  # Mean variance per sample
            # uncertainty_weight = torch.sigmoid(5 * (uncertainty - 0.5))  # Map to (0, 1)

            # Combine reconstruction and uncertainty into condition
            # cond = torch.cat((recons, latent_variance), dim=1)

            # Adjust temperature scaling based on uncertainty
            # x_t = x_t * (1 - uncertainty_weight) + self.temp * uncertainty_weight

            # # Formulation-2 initial latent adjustment
            # if isinstance(self.online_network, DDPMv2):
            #     x_t = recons + uncertainty_weight * torch.randn_like(recons)

        else:
            print("fadfasfasfasfasdf")
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }

###

class DDPMWrapper_new(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps

        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None

    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )
    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def training_step(self, batch, batch_idx):
        # Optimizers
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        vae_optim = self.vae.configure_optimizers()
        # vae_lr_sched = self.vae.lr_schedulers()



        cond = None
        z = None

        x = batch
        mu, logvar = self.vae.encode(x * 0.5 + 0.5)
        z = self.vae.reparameterize(mu, logvar)
        cond = self.vae.decode(z)


        vae_weight = 1.2 - ((self.current_epoch+1) / self.max_epochs)  # VAE 가중치
        ddpm_weight = ((self.current_epoch+1) / self.max_epochs) + 0.2

        if self.current_epoch==0:
            print(self.current_epoch, self.max_epochs)
            print(vae_weight, ddpm_weight)

        mse_loss = nn.MSELoss(reduction="mean") # mean
        recons_loss = mse_loss(cond, x)
        kl_loss = self.compute_kl(mu, logvar)
        self.log("Recons Loss", recons_loss, prog_bar=True)
        self.log("Kl Loss", kl_loss*1e-4, prog_bar=True) 
        vae_loss = recons_loss + kl_loss * 1e-4

        cond = 2 * cond - 1

        # Sample timepoints
        t = torch.randint(
            0, self.online_network.T, size=(x.size(0),), device=self.device
        )

        # Sample noise
        eps = torch.randn_like(x)

        # Predict noise
        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )

        # Compute loss
        ddpm_loss = self.criterion(eps, eps_pred)
        self.log("ddpm loss", ddpm_loss, prog_bar=True)
        loss = vae_loss + ddpm_loss

        # Clip gradients and Optimize
        optim.zero_grad()
        vae_optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()
        vae_optim.step()



        # Scheduler step
        lr_sched.step()
        # vae_lr_sched.step()

        self.log("loss", loss, prog_bar=True)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.online_network.decoder.parameters(), lr=self.lr
        )

        # Define the LR scheduler (As in Ho et al.)
        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }



class DDPMWrapper_new2(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        vae_lr=1e-4,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.vae_lr = vae_lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps


        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None
    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )
    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def training_step(self, batch, batch_idx):
        """
        Combined loss calculation for VAE and DDPM.
        """
        x = batch
        mu, logvar = self.vae.encode(x * 0.5 + 0.5)
        logvar = torch.clamp(logvar, min=-50.0, max=50.0)

        z = self.vae.reparameterize(mu, logvar)
        x_recon = 2 * self.vae.decode(z) - 1


        #Original
        ddpm_weight, vae_weight = 1, 1

        #Linear
        # vae_weight = 1.8 - ((self.current_epoch+1) / 500)
        # ddpm_weight = ((self.current_epoch+1) / 500) + 0.2

        # #Exponential
        # k = math.log(1.8 / 0.2) / 500  # Decay rate
        # vae_weight = 1.8 * math.exp(-k * (self.current_epoch+1))
        # ddpm_weight = 2 - vae_weight  

        # #Sigmoid
        # k=0.1
        # midpoint = 500 / 2
        # sigmoid_value = 1 / (1 + math.exp(-k * (self.current_epoch - midpoint)))
        # vae_weight = 1.8 - sigmoid_value 
        # ddpm_weight = 2 - vae_weight


        recon_loss = self.criterion(x_recon, x)
        kl_loss = (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())) * 1e-4
        vae_loss = recon_loss + kl_loss

        cond = x_recon if self.conditional else None

        t = torch.randint(0, self.online_network.T, size=(x.size(0),), device=self.device)
        eps = torch.randn_like(x)

        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )
        ddpm_loss = self.criterion(eps, eps_pred)

        loss = vae_loss*vae_weight + ddpm_loss*ddpm_weight

        # Optimize
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()
        lr_sched.step()

        # Logging
        self.log("vae_recon_loss", recon_loss, prog_bar=True)
        self.log("vae_kl_loss", kl_loss, prog_bar=True)
        self.log("ddpm_loss", ddpm_loss, prog_bar=True)
        self.log("total_loss", loss, prog_bar=True)

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        """
        Returns a single optimizer for joint VAE and DDPM training.
        """
        optimizer = torch.optim.Adam(
            [
                {"params": self.vae.parameters(), "lr": self.vae_lr},
                {"params": self.online_network.decoder.parameters(), "lr": self.lr},
            ]
        )

        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }
    


#Stage3
class DDPMWrapper_new3(pl.LightningModule):
    def __init__(
        self,
        online_network,
        target_network,
        vae,
        lr=2e-5,
        vae_lr=1e-4,
        cfd_rate=0.0,
        n_anneal_steps=0,
        loss="l1",
        grad_clip_val=1.0,
        sample_from="target",
        resample_strategy="spaced",
        skip_strategy="uniform",
        sample_method="ddpm",
        conditional=True,
        eval_mode="sample",
        pred_steps=None,
        pred_checkpoints=[],
        temp=1.0,
        guidance_weight=0.0,
        z_cond=False,
        ddpm_latents=None,
    ):
        super().__init__()
        assert loss in ["l1", "l2"]
        assert eval_mode in ["sample", "recons"]
        assert resample_strategy in ["truncated", "spaced"]
        assert sample_method in ["ddpm", "ddim"]
        assert skip_strategy in ["uniform", "quad"]

        self.z_cond = z_cond
        self.online_network = online_network
        self.target_network = target_network
        self.vae = vae
        self.cfd_rate = cfd_rate

        # Training arguments
        self.criterion = nn.MSELoss(reduction="mean") if loss == "l2" else nn.L1Loss()
        self.lr = lr
        self.vae_lr = vae_lr
        self.grad_clip_val = grad_clip_val
        self.n_anneal_steps = n_anneal_steps


        # Evaluation arguments
        self.sample_from = sample_from
        self.conditional = conditional
        self.sample_method = sample_method
        self.resample_strategy = resample_strategy
        self.skip_strategy = skip_strategy
        self.eval_mode = eval_mode
        self.pred_steps = self.online_network.T if pred_steps is None else pred_steps
        self.pred_checkpoints = pred_checkpoints
        self.temp = temp
        self.guidance_weight = guidance_weight
        self.ddpm_latents = ddpm_latents

        # Disable automatic optimization
        self.automatic_optimization = False

        # Spaced Diffusion (for spaced re-sampling)
        self.spaced_diffusion = None
    def forward(
        self,
        x,
        cond=None,
        z=None,
        n_steps=None,
        ddpm_latents=None,
        checkpoints=[],
    ):
        sample_nw = (
            self.target_network if self.sample_from == "target" else self.online_network
        )
        spaced_nw = (
            SpacedDiffusionForm2
            if isinstance(self.online_network, DDPMv2)
            else SpacedDiffusion
        )
        # For spaced resampling
        if self.resample_strategy == "spaced":
            num_steps = n_steps if n_steps is not None else self.online_network.T
            indices = space_timesteps(sample_nw.T, num_steps, type=self.skip_strategy)
            if self.spaced_diffusion is None:
                self.spaced_diffusion = spaced_nw(sample_nw, indices).to(x.device)

            if self.sample_method == "ddim":
                return self.spaced_diffusion.ddim_sample(
                    x,
                    cond=cond,
                    z_vae=z,
                    guidance_weight=self.guidance_weight,
                    checkpoints=checkpoints,
                )
            return self.spaced_diffusion(
                x,
                cond=cond,
                z_vae=z,
                guidance_weight=self.guidance_weight,
                checkpoints=checkpoints,
                ddpm_latents=ddpm_latents,
            )

        # For truncated resampling
        if self.sample_method == "ddim":
            raise ValueError("DDIM is only supported for spaced sampling")
        return sample_nw.sample(
            x,
            cond=cond,
            z_vae=z,
            n_steps=n_steps,
            guidance_weight=self.guidance_weight,
            checkpoints=checkpoints,
            ddpm_latents=ddpm_latents,
        )
    def compute_kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    def training_step(self, batch, batch_idx):
        """
        Combined loss calculation for VAE and DDPM.
        """
        x = batch
        mu, logvar = self.vae.encode(x * 0.5 + 0.5)
        logvar = torch.clamp(logvar, min=-100.0, max=100.0)

        z = self.vae.reparameterize(mu, logvar)
        x_recon = 2 * self.vae.decode(z) - 1


        #Original
        ddpm_weight, vae_weight = 1, 1

        #Linear
        # vae_weight = 1.8 - ((self.current_epoch+1) / 500)
        # ddpm_weight = ((self.current_epoch+1) / 500) + 0.2

        # #Exponential
        # k = math.log(1.8 / 0.2) / 500  # Decay rate
        # vae_weight = 1.8 * math.exp(-k * (self.current_epoch+1))
        # ddpm_weight = 2 - vae_weight  

        # #Sigmoid
        # k=0.1
        # midpoint = 500 / 2
        # sigmoid_value = 1 / (1 + math.exp(-k * (self.current_epoch - midpoint)))
        # vae_weight = 1.8 - sigmoid_value 
        # ddpm_weight = 2 - vae_weight


        # recon_loss = self.criterion(x_recon, x)
        mse_loss = nn.MSELoss(reduction="sum")
        recon_loss = mse_loss(x_recon, x)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) * 1e-4
        vae_loss = recon_loss + kl_loss


        
        uncertainty = torch.mean(torch.sigmoid(torch.exp(logvar)-2),dim=1,keepdim=True)
        # print(f"log_var.shape: {logvar.shape}")
        # print(f"x_recon.shape: {x_recon.shape}")
        # print(f"uncertainty: {uncertainty}")
        x_recon = x_recon*(1-uncertainty) +  torch.randn_like(x_recon) * uncertainty



        cond = x_recon if self.conditional else None

        t = torch.randint(0, self.online_network.T, size=(x.size(0),), device=self.device)
        eps = torch.randn_like(x)

        eps_pred = self.online_network(
            x, eps, t, low_res=cond, z=z.squeeze() if self.z_cond else None
        )
        ddpm_loss = self.criterion(eps, eps_pred)

        loss = vae_loss*vae_weight + ddpm_loss*ddpm_weight

        # Optimize
        optim = self.optimizers()
        lr_sched = self.lr_schedulers()

        optim.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            self.online_network.decoder.parameters(), self.grad_clip_val
        )
        optim.step()
        lr_sched.step()

        # Logging
        self.log("vae_recon_loss", recon_loss, prog_bar=True)
        self.log("vae_kl_loss", kl_loss, prog_bar=True)
        self.log("ddpm_loss", ddpm_loss, prog_bar=True)
        self.log("total_loss", loss, prog_bar=True)

        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        if not self.conditional:
            if self.guidance_weight != 0.0:
                raise ValueError(
                    "Guidance weight cannot be non-zero when using unconditional DDPM"
                )
            x_t = batch
            return self(
                x_t,
                cond=None,
                z=None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=None,
            )

        if self.eval_mode == "sample":
            x_t, z = batch
            recons = self.vae(z)
            recons = 2 * recons - 1

            # Initial temperature scaling
            x_t = x_t * self.temp

            # Formulation-2 initial latent
            if isinstance(self.online_network, DDPMv2):
                x_t = recons + self.temp * torch.randn_like(recons)
        else:
            img = batch
            recons = self.vae.forward_recons(img * 0.5 + 0.5)
            recons = 2 * recons - 1

            # DDPM encoder
            x_t = self.online_network.compute_noisy_input(
                img,
                torch.randn_like(img),
                torch.tensor(
                    [self.online_network.T - 1] * img.size(0), device=img.device
                ),
            )

            if isinstance(self.online_network, DDPMv2):
                x_t += recons

        return (
            self(
                x_t,
                cond=recons,
                z=z.squeeze() if self.z_cond else None,
                n_steps=self.pred_steps,
                checkpoints=self.pred_checkpoints,
                ddpm_latents=self.ddpm_latents,
            ),
            recons,
        )

    def configure_optimizers(self):
        """
        Returns a single optimizer for joint VAE and DDPM training.
        """
        optimizer = torch.optim.Adam(
            [
                {"params": self.vae.parameters(), "lr": self.vae_lr},
                {"params": self.online_network.decoder.parameters(), "lr": self.lr},
            ]
        )

        if self.n_anneal_steps == 0:
            lr_lambda = lambda step: 1.0
        else:
            lr_lambda = lambda step: min(step / self.n_anneal_steps, 1.0)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "strict": False,
            },
        }