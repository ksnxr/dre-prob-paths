# [ICML 2025] Density Ratio Estimation with Conditional Probability Paths

This repository implements a method to estimate a density-ratio. It is largely based on https://github.com/ermongroup/dre-infinity/. We have cleaned up the code a bit. GPT was used to support coding.

## Users

First create an environment for this project by running

```bash
# Create and activate an environment
conda create -n dre-cfm python=3.10.15
conda activate dre-cfm
pip install -r my_cpu_requirements.txt  # which is for CPU or my_requirements.txt for CUDA
```

## Experiments

For toy experiments and the MI estimation experiments, when one data setting is first used, there should be one run run beforehand with "--config.training.n_iters=-1", generating the validation set. Afterwards, all jobs using the same data setting can be used as usual.  Note that we use `wandb` for keeping track of train/test statistics. You will have to configure the API Key and set up wandb in the `main.py` file.

We give example commands for running different methods below. We note that the hyperparameters as used below may not be those that resulted in the best performances.

# Gaussians

#### TSM

```
python3 main.py --toy --project=gaussians --config configs/gaussians/time/mlp.py --mode=train \
--doc=gaussians_2_mlp_time_path_var_unit_256_0.002_256 --workdir=../results/gaussians_inits/gaussians_2_mlp_time_path_var_unit/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=path_var \
--config.data.dim=2 --config.training.unit_factor=True
```

#### CTSM

```
python3 main.py --toy --project=gaussians --config configs/gaussians/time/c_mlp.py --mode=train \
--doc=gaussians_2_c_mlp_time_obj_var_unit_256_0.002_256 --workdir=../results/gaussians_inits/gaussians_2_c_mlp_time_obj_var_unit/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=obj_var \
--config.data.dim=2 --config.training.unit_factor=True
```

#### CTSM-v

```
python3 main.py --toy --project=gaussians --config configs/gaussians/time/c_full_mlp.py --mode=train \
--doc=gaussians_2_c_full_mlp_time_obj_var_unit_256_0.002_256 --workdir=../results/gaussians_inits/gaussians_2_c_full_mlp_time_obj_var_unit/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=obj_var \
--config.data.dim=2 --config.training.unit_factor=True
```

# GMMs

#### TSM

```
python3 main.py --toy --project=gmms --config configs/gmms/time/mlp.py --mode=train \
--doc=gmms_20_mlp_time_path_var_raw_two_sb_two_sb_var=1.0_1.0_20000_1000_256_0.002_256 --workdir=../results/gmms_inits/gmms_20_mlp_time_path_var_raw_two_sb_two_sb_var=1.0_1.0_20000_1000/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=path_var \
--config.data.dim=20 --config.training.unit_factor=False \
--config.training.use_two_sb=True --config.training.two_sb_var=1.0 \
--config.data.k=1.0 --config.training.n_iters=20000 --config.training.eval_freq=1000
```

#### CTSM

```
python3 main.py --toy --project=gmms --config configs/gmms/time/c_mlp.py --mode=train \
--doc=gmms_20_c_mlp_time_obj_var_unit_two_sb_two_sb_var=1.0_1.0_20000_1000_256_0.002_256 --workdir=../results/gmms_inits/gmms_20_c_mlp_time_obj_var_unit_two_sb_two_sb_var=1.0_1.0_20000_1000/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=obj_var \
--config.data.dim=20 --config.training.unit_factor=True \
--config.training.use_two_sb=True --config.training.two_sb_var=1.0 \
--config.data.k=1.0 --config.training.n_iters=20000 --config.training.eval_freq=1000
```

#### CTSM-v

```
python3 main.py --toy --project=gmms --config configs/gmms/time/c_full_mlp.py --mode=train \
--doc=gmms_20_c_full_mlp_time_obj_var_unit_two_sb_two_sb_var=1.0_1.0_20000_1000_256_0.002_256 --workdir=../results/gmms_inits/gmms_20_c_full_mlp_time_obj_var_unit_two_sb_two_sb_var=1.0_1.0_20000_1000/256_0.002_256/ \
--config.seed=1 --config.training.batch_size=256 \
--config.optim.lr=0.002 --config.model.z_dim=256 --config.training.reweight=obj_var \
--config.data.dim=20 --config.training.unit_factor=True \
--config.training.use_two_sb=True --config.training.two_sb_var=1.0 \
--config.data.k=1.0 --config.training.n_iters=20000 --config.training.eval_freq=1000
```

# MIs

#### TSM

```
python3 main.py --toy \
--config configs/gmm_mutual_info/time/param.py --mode=train \
--doc=mi_40_time_1_0.001 --config.model.type=time \
--config.training.joint=False --config.data.dim=40 \
--config.seed=1 --config.training.n_iters=20001 \
--workdir=../results/mi/mi_40_time_1_0.001/ \
--config.training.batch_size=512 --config.training.eval_freq=2000 \
--project=mi --config.optim.lr=0.001
```

#### CTSM

```
python3 main.py --toy \
--config configs/gmm_mutual_info/time/c_param.py --mode=train \
--doc=c_mi_40_time_1_0.001 --config.model.type=time \
--config.training.joint=False --config.data.dim=40 \
--config.seed=1 --config.training.n_iters=20001 \
--workdir=../results/mi/c_mi_40_time_1_0.001/ \
--config.training.batch_size=512 --config.training.eval_freq=2000 \
--project=mi --config.optim.lr=0.001
```

#### CTSM-v

```
python3 main.py --toy \
--config configs/gmm_mutual_info/time/c_full_param.py --mode=train \
--doc=c_full_mi_40_time_1_0.001 --config.model.type=time \
--config.training.joint=False --config.data.dim=40 \
--config.seed=1 --config.training.n_iters=20001 \
--workdir=../results/mi/c_full_mi_40_time_1_0.001/ \
--config.training.batch_size=512 --config.training.eval_freq=2000 \
--project=mi --config.optim.lr=0.001
```

# EBMs

## Gaussian flows

### Train

#### TSM

```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=train --doc=gaussian_1e-3_40 \
--workdir=../results/ebm/gaussian_1e-3_40 --project=ebm \
--config.training.iw=True --config.training.interpolate=True --config.training.batch_size=500 \
--config.training.buffer_size=100 --config.optim.lr=1e-3 --config.training.n_iters=400001
```

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_z_gaussian_time_interpolate_epsilons.py \
--mode=train --doc=c_mnist_gaussian_x_none_2e-3_resample \
--workdir=../results/ebm/c_mnist_gaussian_x_none_2e-3_resample --project=ebm \
--config.training.iw=False --config.training.interpolate=True --config.training.batch_size=500 \
--config.training.buffer_size=100 --config.optim.lr=2e-3 --config.training.n_iters=400001 \
--config.model.embedding_type=fourier --config.training.use_zt=False \
--config.seed=1 --config.training.resample_t=True
```

### Evaluate Approx. BPD

#### TSM

```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=eval --doc=mnist_gaussian_1e-3_direct \
--workdir=../results/ebm/gaussian_1e-3_40 --project=ebm_eval \
--config.eval.begin_ckpt=193 --config.eval.end_ckpt=193 \
--config.eval.enable_bpd=True --config.eval.ais=False \
--eval_folder=eval_direct --config.seed=1
```

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_z_gaussian_time_interpolate_epsilons.py \
--mode=eval --doc=c_mnist_gaussian_x_none_2e-3_direct \
--workdir=../results/ebm/c_mnist_gaussian_x_none_2e-3_resample --project=ebm_eval \
--config.eval.begin_ckpt=134 --config.eval.end_ckpt=134 \
--config.eval.enable_bpd=True --config.eval.ais=False \
--eval_folder=eval_direct --config.model.embedding_type="fourier" \
--config.training.use_zt=False --config.seed=1
```

### AIS

#### TSM

```
python3 main.py --flow \
--config configs/mnist/z_gaussian_time_interpolate.py \
--mode=eval --doc=mnist_gaussian_1e-3_ais_7.5e-3 \
--workdir=../results/ebm/gaussian_1e-3_40 --project=ebm_eval \
--config.eval.begin_ckpt=193 --config.eval.end_ckpt=193 \
--config.eval.ais=True --config.eval.n_ais_samples=100 \
--config.eval.ais_batch_size=50 --config.eval.n_ais_steps=1000 \
--config.eval.n_continue=100 --config.eval.initial_step_size=7.5e-3 \
--eval_folder=eval_ais_7.5e-3 --config.eval.ais_method=ais \
--config.eval.ais_rtol=1e-3 --config.eval.ais_atol=1e-6 --config.seed=1
```

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_z_gaussian_time_interpolate_epsilons.py \
--mode=eval --doc=c_mnist_gaussian_x_none_2e-3_ais_7.5e-3 \
--workdir=../results/ebm/c_mnist_gaussian_x_none_2e-3_resample --project=ebm_eval \
--config.eval.begin_ckpt=134 --config.eval.end_ckpt=134 \
--config.eval.ais=True --config.eval.n_ais_samples=100 \
--config.eval.ais_batch_size=50 --config.eval.n_ais_steps=1000 \
--config.eval.n_continue=100 --config.eval.initial_step_size=7.5e-3 \
--eval_folder=eval_ais_7.5e-3 --config.eval.ais_method=ais \
--config.model.embedding_type="fourier" --config.eval.ais_rtol=1e-3 \
--config.eval.ais_atol=1e-6 --config.training.use_zt=False \
--config.seed=1
```

## Pixel space

### Train

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_none_time_interpolate_epsilons.py \
--mode=train --doc=m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample \
--workdir=../results/ebm/m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample --project=ebm \
--config.training.iw=False --config.training.interpolate=True --config.training.batch_size=500 \
--config.training.buffer_size=100 --config.optim.lr=1e-3 --config.training.n_iters=250001 \
--config.model.name=c_ncsnpp_t --config.model.embedding_type=fourier --config.optim.grad_clip=1.0 \
--config.training.use_zt=False --config.training.resample_t=True \
--config.seed=1
```

### Evaluate Approx. BPD

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_none_time_interpolate_epsilons.py \
--mode=eval --doc=m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample_direct \
--workdir=../results/ebm/m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample --project=ebm_eval \
--config.eval.begin_ckpt=87 --config.eval.end_ckpt=87 \
--config.eval.enable_bpd=True --config.eval.ais=False \
--eval_folder=eval_direct --config.model.embedding_type=fourier \
--config.training.use_zt=False --config.seed=1 \
--config.model.name=c_ncsnpp_t
```

### AIS

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_none_time_interpolate_epsilons.py \
--mode=eval --doc=m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample_ais_ais_2.5e-2 \
--workdir=../results/ebm/m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample --project=ebm_eval \
--config.eval.begin_ckpt=87 --config.eval.end_ckpt=87 \
--config.eval.ais=True --config.eval.n_ais_samples=50 \
--config.eval.ais_batch_size=50 --config.eval.n_ais_steps=1000 \
--config.eval.n_continue=100 --config.eval.initial_step_size=2.5e-2 \
--eval_folder=eval_ais_2.5e-2 --config.eval.ais_method=ais \
--config.model.embedding_type=fourier --config.eval.ais_rtol=1e-3 \
--config.eval.ais_atol=1e-6 --config.training.use_zt=False \
--config.seed=1 --config.model.name=c_ncsnpp_t
```

### PF ODE

#### CTSM-v

```
python3 main.py --flow \
--config configs/mnist/c_none_time_interpolate_epsilons.py \
--mode=eval --doc=m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample_sampling \
--workdir=../results/ebm/m_c_mnist_c_ncsnpp_t_fourier_1e-3_resample --project=ebm_eval \
--config.eval.begin_ckpt=87 --config.eval.end_ckpt=87 \
--config.eval.enable_bpd=False --config.eval.ais=False \
--eval_folder=eval_sampling --config.seed=1 \
--config.model.embedding_type=fourier --config.training.use_zt=False \
--config.sampling.method=ode --config.eval.enable_sampling=True \
--config.model.name=c_ncsnpp_t --config.eval.batch_size=64
```


## Citation

If you find our repository useful, please consider citing it as follows
```
@inproceedings{
yu2025density,
title={Density Ratio Estimation with Conditional Probability Paths},
author={Hanlin Yu and Arto Klami and Aapo Hyvarinen and Anna Korba and Omar Chehab},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=Gn2izAiYzZ}
}
```
