# CIFAR-10 (Form-2)
python main/train_ddpmvae.py +dataset=cifar10/train \
                     dataset.vae.data.root='/home/jungwoo/datasets/' \
                     dataset.vae.data.name='cifar10' \
                     dataset.vae.training.log_step=10 \
                     dataset.vae.training.epochs=500 \
                     dataset.vae.training.device=\'gpu:1,2,3\' \
                     dataset.vae.training.results_dir=\'/home/jungwoo/HW/DiffuseVAE/results/stage3/\' \
                     dataset.ddpm.data.root=\'/home/jungwoo/datasets/\' \
                     dataset.ddpm.data.name='cifar10' \
                     dataset.ddpm.data.norm=True \
                     dataset.ddpm.data.hflip=True \
                     dataset.vae.training.workers=2 \
                     dataset.vae.training.chkpt_prefix=\'cifar10_alpha=1.0\' \
                     dataset.vae.training.alpha=1.0 \
                     dataset.ddpm.model.dim=128 \
                     dataset.ddpm.model.dropout=0.3 \
                     dataset.ddpm.model.attn_resolutions=\'16,\' \
                     dataset.ddpm.model.n_residual=2 \
                     dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                     dataset.ddpm.model.n_heads=8 \
                     dataset.ddpm.training.type='form2' \
                     dataset.ddpm.training.cfd_rate=0.0 \
                     dataset.ddpm.training.epochs=500 \
                     dataset.ddpm.training.z_cond=False \
                     dataset.ddpm.training.batch_size=64 \
                     dataset.ddpm.training.device=\'gpu:1,2,3\' \
                     dataset.ddpm.training.results_dir=\'/home/jungwoo/HW/DiffuseVAE/results/stage3/\' \
                     dataset.ddpm.training.workers=1 \
                     dataset.ddpm.training.chkpt_prefix=\'cifar10_reprotry1_form2\'

