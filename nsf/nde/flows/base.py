"""Basic definitions for the flows module."""

import torch
import nsf.nsf_utils as nsf_utils

from nsf.nde import distributions


class Flow(distributions.Distribution):
    """Base class for all flow objects."""

    def __init__(self, transform, distribution):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution

    def _log_prob(self, inputs, context):
        noise, logabsdet = self._transform(inputs, context=context)
        # TODO TODO TODO
        # noise, logabsdet = self._transform.inverse(inputs, context=None)
        log_prob = self._distribution.log_prob(noise, context=context)
        # TODO: HACK
        try:
            log_p = log_prob + logabsdet
        except:
            log_p = log_prob.detach().cpu() + logabsdet
        return log_p

    def _sample(self, num_samples, context=None):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nsf_utils.merge_leading_dims(noise, num_dims=2)
            context = nsf_utils.repeat_rows(context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nsf_utils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample(self, z, context=None, rescale=True):
        """
        Added this myself, makes sure samples are rescaled to [-1, 1] for score model
        :param z:
        :return:
        """
        samples, _ = self._transform.inverse(z, context=context)
        if rescale:
            # automatically rescale samples such that they lie within [-1, 1]
            samples /= 256.0
            samples = (samples * 2.0) - 1.0
        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=context
        )

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nsf_utils.merge_leading_dims(noise, num_dims=2)
            context = nsf_utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nsf_utils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = nsf_utils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(self, inputs, context=None, logdet=False):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        noise, sldj = self._transform(inputs, context=context)
        if logdet:
            return noise, sldj
        else:
            return noise


class FlowDataTransform(distributions.Distribution):
    """Base class for all flow objects.
    NOTE: the Base class has been modified such that the data transform module
    is a separate property that can be called by the model
    """

    def __init__(self, transform, distribution, train_transform, val_transform):
        """Constructor.

        Args:
            transform: A `Transform` object, it transforms data into noise.
            distribution: A `Distribution` object, the base distribution of the flow that
                generates the noise.
        """
        super().__init__()
        self._transform = transform
        self._distribution = distribution
        self._train_transform = train_transform
        self._val_transform = val_transform

    def _log_prob(self, inputs, context, transform=False, train=True):
        if transform:
            if train:
                inputs, x_logabsdet = self._train_transform(inputs)
            else:
                inputs, x_logabsdet = self._val_transform(inputs)
        noise, logabsdet = self._transform(inputs, context=context)
        log_prob = self._distribution.log_prob(noise, context=context)
        # TODO: HACK
        try:
            log_p = log_prob + logabsdet
        except:
            log_p = log_prob.detach().cpu() + logabsdet
        if transform:
            # account for data transform
            log_p = log_p + x_logabsdet
        return log_p

    def _sample(self, num_samples, context=None):
        noise = self._distribution.sample(num_samples, context=context)

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nsf_utils.merge_leading_dims(noise, num_dims=2)
            context = nsf_utils.repeat_rows(context, num_reps=num_samples)

        samples, _ = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nsf_utils.split_leading_dim(samples, shape=[-1, num_samples])

        return samples

    def sample(self, z, context=None, rescale=True, transform=False, train=True):
        """
        Added this myself, makes sure samples are rescaled to [-1, 1] for score model
        :param z:
        :return:
        """
        samples, _ = self._transform.inverse(z, context=context)
        if transform:
            if train:
                samples, _ = self._train_transform.inverse(samples)
            else:
                samples, _ = self._val_transform.inverse(samples)
            # samples = torch.clamp(samples, 0., 256.)
        if rescale:
            # automatically rescale samples such that they lie within [-1, 1]
            samples /= 256.0
            samples = (samples * 2.0) - 1.0
        return samples

    def sample_and_log_prob(self, num_samples, context=None):
        """Generates samples from the flow, together with their log probabilities.

        For flows, this is more efficient that calling `sample` and `log_prob` separately.
        """
        noise, log_prob = self._distribution.sample_and_log_prob(
            num_samples, context=context
        )

        if context is not None:
            # Merge the context dimension with sample dimension in order to apply the transform.
            noise = nsf_utils.merge_leading_dims(noise, num_dims=2)
            context = nsf_utils.repeat_rows(context, num_reps=num_samples)

        samples, logabsdet = self._transform.inverse(noise, context=context)

        if context is not None:
            # Split the context dimension from sample dimension.
            samples = nsf_utils.split_leading_dim(samples, shape=[-1, num_samples])
            logabsdet = nsf_utils.split_leading_dim(logabsdet, shape=[-1, num_samples])

        return samples, log_prob - logabsdet

    def transform_to_noise(
        self, inputs, context=None, transform=False, train=True, logdet=False
    ):
        """Transforms given data into noise. Useful for goodness-of-fit checking.

        Args:
            inputs: A `Tensor` of shape [batch_size, ...], the data to be transformed.
            context: A `Tensor` of shape [batch_size, ...] or None, optional context associated
                with the data.

        Returns:
            A `Tensor` of shape [batch_size, ...], the noise.
        """
        if transform:
            if train:
                inputs, x_logabsdet = self._train_transform(inputs)
            else:
                inputs, x_logabsdet = self._val_transform(inputs)
        # default setting
        noise, logabsdet = self._transform(inputs, context=context)

        # just return transformed
        if not logdet:
            return noise
        else:
            if transform:
                logabsdet = logabsdet + x_logabsdet
            return noise, logabsdet
