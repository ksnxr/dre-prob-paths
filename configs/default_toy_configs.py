import ml_collections
import torch


def get_default_configs():
    config = ml_collections.ConfigDict()

    # training
    config.training = training = ml_collections.ConfigDict()
    config.training.batch_size = 128
    training.n_iters = 20000
    training.snapshot_freq = 1000
    training.eval_freq = 1000
    training.log_freq = 100
    training.ratio_freq = 1000
    ## store additional checkpoints for preemption in cloud computing environments
    training.snapshot_freq_for_preemption = 10000
    ## produce samples at each snapshot.
    training.snapshot_sampling = True
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.reweight = "path_var"

    training.conditional = False
    training.dsm = False
    training.prob_path = "OneVP"
    training.unit_factor = False

    training.full = False

    training.plot_scatter = False

    # losses
    training.joint = False
    training.algo = "ssm"

    # sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = False
    sampling.probability_flow = False
    sampling.snr = 0.16

    # evaluation
    config.eval = evaluate = ml_collections.ConfigDict()
    evaluate.begin_ckpt = 9
    evaluate.end_ckpt = 26
    evaluate.batch_size = 64
    evaluate.enable_sampling = False
    evaluate.num_samples = 50000
    evaluate.enable_loss = True
    evaluate.enable_bpd = False
    evaluate.bpd_dataset = "test"
    evaluate.rtol = 1e-6
    evaluate.atol = 1e-6

    # data
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "MNIST"
    data.image_size = 28
    data.random_flip = False
    data.centered = False
    data.uniform_dequantization = False
    data.num_channels = 1
    data.k = 1.0  # for GMMs

    # model
    config.model = model = ml_collections.ConfigDict()
    model.ema = False
    model.z_dim = 128

    # optimization
    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0.0
    optim.optimizer = "Adam"
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.grad_clip = -1.0
    optim.warmup = 0
    optim.amsgrad = False
    optim.scheduler = True

    config.seed = 42
    config.device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )

    return config
