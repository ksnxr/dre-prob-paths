import os
import numpy as np
import jax

# import tensorflow as tf
# import tensorflow_datasets as tfds
import torch.utils.data

# flow-specific code
import torchvision.transforms.functional as F
import torchvision as tv
import torchvision.transforms as tr
from torchvision.datasets import MNIST, FashionMNIST
import torchvision.transforms as transforms


def logit_transform(image, lambd=1e-6):
    image = lambd + (1 - 2 * lambd) * image
    return torch.log(image) - torch.log1p(-image)


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


# def crop_resize(image, resolution):
#     """Crop and resize an image to the given resolution."""
#     crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
#     h, w = tf.shape(image)[0], tf.shape(image)[1]
#     image = image[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
#     image = tf.image.resize(
#         image,
#         size=(resolution, resolution),
#         antialias=True,
#         method=tf.image.ResizeMethod.BICUBIC,
#     )
#     return tf.cast(image, tf.uint8)


# def resize_small(image, resolution):
#     """Shrink an image to the given resolution."""
#     h, w = image.shape[0], image.shape[1]
#     ratio = resolution / min(h, w)
#     h = tf.round(h * ratio, tf.int32)
#     w = tf.round(w * ratio, tf.int32)
#     return tf.image.resize(image, [h, w], antialias=True)


# def central_crop(image, size):
#     """Crop the center of an image to the given size."""
#     top = (image.shape[0] - size) // 2
#     left = (image.shape[1] - size) // 2
#     return tf.image.crop_to_bounding_box(image, top, left, size, size)


# def get_dataset(config, uniform_dequantization=False, evaluation=False):
#     """Create data loaders for training and evaluation.

#     Args:
#       config: A ml_collection.ConfigDict parsed from config files.
#       uniform_dequantization: If `True`, add uniform dequantization to images.
#       evaluation: If `True`, fix number of epochs to 1.

#     Returns:
#       train_ds, eval_ds, dataset_builder.
#     """
#     # Compute batch size for this worker.
#     batch_size = (
#         config.training.batch_size if not evaluation else config.eval.batch_size
#     )
#     if batch_size % jax.device_count() != 0:
#         raise ValueError(
#             f"Batch sizes ({batch_size} must be divided by"
#             f"the number of devices ({jax.device_count()})"
#         )

#     # Reduce this when image resolution is too large and data pointer is stored
#     shuffle_buffer_size = 10000
#     prefetch_size = tf.data.experimental.AUTOTUNE
#     num_epochs = None if not evaluation else 1

#     # Create dataset builders for each dataset.
#     if config.data.dataset == "CIFAR10":
#         dataset_builder = tfds.builder("cifar10")
#         train_split_name = "train"
#         eval_split_name = "test"

#         def resize_op(img):
#             img = tf.image.convert_image_dtype(img, tf.float32)
#             return tf.image.resize(
#                 img, [config.data.image_size, config.data.image_size], antialias=True
#             )

#     elif config.data.dataset == "MNIST":
#         dataset_builder = tfds.builder("mnist")
#         train_split_name = "train"
#         eval_split_name = "test"

#         def resize_op(img):
#             img = tf.image.convert_image_dtype(img, tf.float32)
#             return tf.image.resize(
#                 img, [config.data.image_size, config.data.image_size], antialias=True
#             )

#     elif config.data.dataset == "SVHN":
#         dataset_builder = tfds.builder("svhn_cropped")
#         train_split_name = "train"
#         eval_split_name = "test"

#         def resize_op(img):
#             img = tf.image.convert_image_dtype(img, tf.float32)
#             return tf.image.resize(
#                 img, [config.data.image_size, config.data.image_size], antialias=True
#             )

#     elif config.data.dataset == "CELEBA":
#         dataset_builder = tfds.builder("celeb_a")
#         train_split_name = "train"
#         eval_split_name = "validation"

#         def resize_op(img):
#             img = tf.image.convert_image_dtype(img, tf.float32)
#             img = central_crop(img, 140)
#             img = resize_small(img, config.data.image_size)
#             return img

#     elif config.data.dataset == "LSUN":
#         dataset_builder = tfds.builder(f"lsun/{config.data.category}")
#         train_split_name = "train"
#         eval_split_name = "validation"

#         if config.data.image_size == 128:

#             def resize_op(img):
#                 img = tf.image.convert_image_dtype(img, tf.float32)
#                 img = resize_small(img, config.data.image_size)
#                 img = central_crop(img, config.data.image_size)
#                 return img

#         else:

#             def resize_op(img):
#                 img = crop_resize(img, config.data.image_size)
#                 img = tf.image.convert_image_dtype(img, tf.float32)
#                 return img

#     elif config.data.dataset in ["FFHQ", "CelebAHQ"]:
#         dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
#         train_split_name = eval_split_name = "train"

#     else:
#         raise NotImplementedError(f"Dataset {config.data.dataset} not yet supported.")

#     # Customize preprocess functions for each dataset.
#     if config.data.dataset in ["FFHQ", "CelebAHQ"]:

#         def preprocess_fn(d):
#             sample = tf.io.parse_single_example(
#                 d,
#                 features={
#                     "shape": tf.io.FixedLenFeature([3], tf.int64),
#                     "data": tf.io.FixedLenFeature([], tf.string),
#                 },
#             )
#             data = tf.io.decode_raw(sample["data"], tf.uint8)
#             data = tf.reshape(data, sample["shape"])
#             data = tf.transpose(data, (1, 2, 0))
#             img = tf.image.convert_image_dtype(data, tf.float32)
#             if config.data.random_flip and not evaluation:
#                 img = tf.image.random_flip_left_right(img)
#             if uniform_dequantization:
#                 img = (
#                     tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
#                 ) / 256.0
#             return dict(image=img, label=None)

#     else:

#         def preprocess_fn(d):
#             """Basic preprocessing function scales data to [0, 1) and randomly flips."""
#             img = resize_op(d["image"])
#             if config.data.random_flip and not evaluation:
#                 img = tf.image.random_flip_left_right(img)
#             if uniform_dequantization:
#                 img = (
#                     tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.0
#                 ) / 256.0

#             return dict(image=img, label=d.get("label", None))

#     def create_dataset(dataset_builder, split):
#         dataset_options = tf.data.Options()
#         dataset_options.experimental_optimization.map_parallelization = True
#         dataset_options.experimental_threading.private_threadpool_size = 48
#         dataset_options.experimental_threading.max_intra_op_parallelism = 1
#         read_config = tfds.ReadConfig(options=dataset_options)
#         if isinstance(dataset_builder, tfds.core.DatasetBuilder):
#             dataset_builder.download_and_prepare()
#             ds = dataset_builder.as_dataset(
#                 split=split, shuffle_files=True, read_config=read_config
#             )
#         else:
#             ds = dataset_builder.with_options(dataset_options)
#         ds = ds.repeat(count=num_epochs)
#         ds = ds.shuffle(shuffle_buffer_size)
#         ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         ds = ds.batch(batch_size, drop_remainder=True)
#         return ds.prefetch(prefetch_size)

#     train_ds = create_dataset(dataset_builder, train_split_name)
#     eval_ds = create_dataset(dataset_builder, eval_split_name)
#     return train_ds, eval_ds, dataset_builder


def get_dataset_for_flow(config, uniform_dequantization=False):
    """
    hello
    :param config:
    :param uniform_dequantization:
    :param evaluation:
    :return:
    """
    data_dir = "./data"
    train_transform = test_transform = transforms.Compose(
        [transforms.Resize(config.data.image_size), transforms.ToTensor()]
    )
    if config.data.dataset == "MNIST":
        dataset = MNIST(
            os.path.join(data_dir, "datasets", "mnist"),
            train=True,
            download=True,
            transform=train_transform,
        )
    elif config.data.dataset == "FashionMNIST":
        dataset = FashionMNIST(
            os.path.join(data_dir, "datasets", "fashion_mnist"),
            train=True,
            download=True,
            transform=train_transform,
        )
    # subset to first 50K examples for train
    train_indices = np.arange(50000)
    train_ds = torch.utils.data.Subset(dataset, train_indices)
    eval_ds = torch.utils.data.Subset(dataset, np.arange(50000, 60000))

    # eval_ds = MNIST(os.path.join(data_dir, 'datasets', 'mnist_test'),
    #                 train=False, download=True,
    #                 transform=test_transform)

    # TODO: not set up for actual evaluation yet! this is just returning the validation set
    train_ds = torch.utils.data.DataLoader(
        train_ds,
        config.training.batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=False,
    )
    eval_ds = torch.utils.data.DataLoader(
        eval_ds, config.eval.batch_size, shuffle=False, num_workers=2, drop_last=False
    )

    return train_ds, eval_ds


def get_test_set_for_flow(config):
    data_dir = "./data"
    test_transform = transforms.Compose(
        [transforms.Resize(config.data.image_size), transforms.ToTensor()]
    )
    eval_ds = MNIST(
        os.path.join(data_dir, "datasets", "mnist_test"),
        train=False,
        download=True,
        transform=test_transform,
    )
    eval_ds = torch.utils.data.DataLoader(
        eval_ds, config.eval.batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return eval_ds


def get_ais_test_set_for_flow(config):
    data_dir = "./data"
    test_transform = transforms.Compose(
        [transforms.Resize(config.data.image_size), transforms.ToTensor()]
    )
    eval_ds = MNIST(
        os.path.join(data_dir, "datasets", "mnist_test"),
        train=False,
        download=True,
        transform=test_transform,
    )
    dataset_size = len(eval_ds)
    chosen_indexes = torch.randperm(dataset_size)[: config.eval.n_ais_samples]
    eval_ds = torch.utils.data.Subset(eval_ds, chosen_indexes)
    eval_ds = torch.utils.data.DataLoader(
        eval_ds,
        config.eval.ais_batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
    )
    return eval_ds


def get_raise_batch(config):
    data_dir = "./data"
    test_transform = transforms.Compose(
        [transforms.Resize(config.data.image_size), transforms.ToTensor()]
    )
    eval_ds = MNIST(
        os.path.join(data_dir, "datasets", "mnist_test"),
        train=False,
        download=True,
        transform=test_transform,
    )

    n_samples = config.eval.n_ais_samples

    dataset_size = len(eval_ds)
    random_indices = torch.randperm(dataset_size)[:n_samples]

    subset_dataset = torch.utils.data.Subset(eval_ds, random_indices)

    subset_loader = torch.utils.data.DataLoader(subset_dataset, batch_size=n_samples)

    for batch in subset_loader:
        break

    return batch
