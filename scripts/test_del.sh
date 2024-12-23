python main/eval/ddpm/sample_cond.py +dataset=cifar10/test \
                        dataset.ddpm.data.norm=True \
                        dataset.ddpm.model.dim=128 \
                        dataset.ddpm.model.dropout=0.3 \
                        dataset.ddpm.model.attn_resolutions=\'16,\' \
                        dataset.ddpm.model.n_residual=2 \
                        dataset.ddpm.model.dim_mults=\'1,2,2,2\' \
                        dataset.ddpm.model.n_heads=8 \
                        dataset.ddpm.evaluation.guidance_weight=0.0 \
                        dataset.ddpm.evaluation.seed=0 \
                        dataset.ddpm.evaluation.sample_prefix='gpu_0' \
                        dataset.ddpm.evaluation.device=\'gpu:0\' \
                        dataset.ddpm.evaluation.save_mode='image' \
                        dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_cifar10_reprotry1_form2/checkpoints/ddpmrepro-cifar10_reprotry1_form2-epoch=2001-loss=0.0238.ckpt\' \
                        dataset.ddpm.evaluation.type='form2' \
                        dataset.ddpm.evaluation.resample_strategy='truncated' \
                        dataset.ddpm.evaluation.skip_strategy='quad' \
                        dataset.ddpm.evaluation.sample_method='ddpm' \
                        dataset.ddpm.evaluation.sample_from='target' \
                        dataset.ddpm.evaluation.temp=1.0 \
                        dataset.ddpm.evaluation.batch_size=64 \
                        dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/ddpm_cifar10_form2_repro_try1/2000_expde_50k/\' \
                        dataset.ddpm.evaluation.z_cond=False \
                        dataset.ddpm.evaluation.n_samples=12500 \
                        dataset.ddpm.evaluation.n_steps=1000 \
                        dataset.ddpm.evaluation.save_vae=True \
                        dataset.ddpm.evaluation.workers=1 \
                        dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/checkpoints/cifar10/vae-cifar10-epoch=500-train_loss=0.00.ckpt\' \
                        dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/cifar10_latents/gmm_z/gmm_50.joblib\'