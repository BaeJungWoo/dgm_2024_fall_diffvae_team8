# python main/eval/ddpm/interpolate_vae.py +dataset=celebamaskhq128/test \
#                             dataset.ddpm.data.norm=True \
#                             dataset.ddpm.model.attn_resolutions=\'16,\' \
#                             dataset.ddpm.model.dropout=0.1 \
#                             dataset.ddpm.model.n_residual=2 \
#                             dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
#                             dataset.ddpm.model.n_heads=8 \
#                             dataset.ddpm.evaluation.guidance_weight=0.0 \
#                             dataset.ddpm.evaluation.seed=2021 \
#                             dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cmhq/ddpmv2-cmhq128_rework_form1_18thJune_sota_nheads=8_dropout=0.1-epoch=999-loss=0.0066.ckpt\' \
#                             dataset.ddpm.evaluation.type='form1' \
#                             dataset.ddpm.evaluation.resample_strategy='truncated' \
#                             dataset.ddpm.evaluation.skip_strategy='uniform' \
#                             dataset.ddpm.evaluation.sample_method='ddpm' \
#                             dataset.ddpm.evaluation.sample_from='target' \
#                             dataset.ddpm.evaluation.temp=1.0 \
#                             dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/diffusevae_rework_experiments/linear_interpolate_z/form1_latents_shared/\' \
#                             dataset.ddpm.evaluation.z_cond=False \
#                             dataset.ddpm.evaluation.n_steps=1000 \
#                             dataset.ddpm.evaluation.save_vae=True \
#                             dataset.ddpm.evaluation.workers=1 \
#                             dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cmhq/vae-cmhq128_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
#                             dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/cmhq128_latents/gmm_z/gmm_100.joblib\' \
#                             dataset.ddpm.data.ddpm_latent_path=\'/data1/kushagrap20/ddpm_latents.pt\' \

python main/eval/ddpm/interpolate_ddpm.py +dataset=celebamaskhq128/test \
                            dataset.ddpm.data.norm=True \
                            dataset.ddpm.model.attn_resolutions=\'16,\' \
                            dataset.ddpm.model.dropout=0.1 \
                            dataset.ddpm.model.n_residual=2 \
                            dataset.ddpm.model.dim_mults=\'1,2,2,3,4\' \
                            dataset.ddpm.model.n_heads=8 \
                            dataset.ddpm.evaluation.guidance_weight=0.0 \
                            dataset.ddpm.evaluation.seed=1 \
                            dataset.ddpm.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cmhq/ddpmv2-cmhq128_rework_form1_18thJune_sota_nheads=8_dropout=0.1-epoch=999-loss=0.0066.ckpt\' \
                            dataset.ddpm.evaluation.type='form1' \
                            dataset.ddpm.evaluation.resample_strategy='truncated' \
                            dataset.ddpm.evaluation.skip_strategy='quad' \
                            dataset.ddpm.evaluation.sample_method='ddpm' \
                            dataset.ddpm.evaluation.sample_from='target' \
                            dataset.ddpm.evaluation.temp=1.0 \
                            dataset.ddpm.evaluation.save_path=\'/data1/kushagrap20/diffusevae_rework_experiments/linear_interpolate_ddpm/form1_fixed/\' \
                            dataset.ddpm.evaluation.z_cond=False \
                            dataset.ddpm.evaluation.n_steps=1000 \
                            dataset.ddpm.evaluation.save_vae=True \
                            dataset.ddpm.evaluation.workers=1 \
                            dataset.vae.evaluation.chkpt_path=\'/data1/kushagrap20/diffusevae_rework/cmhq/vae-cmhq128_alpha=1.0-epoch=499-train_loss=0.0000.ckpt\' \
                            dataset.vae.evaluation.expde_model_path=\'/data1/kushagrap20/cmhq128_latents/gmm_z/gmm_100.joblib\' \
