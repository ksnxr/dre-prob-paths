"""Implementations of orthogonal transforms."""

import torch

from torch import nn

import nsf.nsf_utils as nsf_utils

from nsf.nde import transforms


class HouseholderSequence(transforms.Transform):
    """A sequence of Householder transforms.

    This class can be used as a way of parameterizing an orthogonal matrix.
    """

    def __init__(self, features, num_transforms):
        """Constructor.

        Args:
            features: int, dimensionality of the input.
            num_transforms: int, number of Householder transforms to use.

        Raises:
            TypeError: if arguments are not the right type.
        """
        if not nsf_utils.is_positive_int(features):
            raise TypeError("Number of features must be a positive integer.")
        if not nsf_utils.is_positive_int(num_transforms):
            raise TypeError("Number of transforms must be a positive integer.")

        super().__init__()
        self.features = features
        self.num_transforms = num_transforms
        # TODO: are randn good initial values?
        self.q_vectors = nn.Parameter(torch.randn(num_transforms, features))

    @staticmethod
    def _apply_transforms(inputs, q_vectors):
        """Apply the sequence of transforms parameterized by given q_vectors to inputs.

        Costs O(KDN), where:
        - K is number of transforms
        - D is dimensionality of inputs
        - N is number of inputs

        Args:
            inputs: Tensor of shape [N, D]
            q_vectors: Tensor of shape [K, D]

        Returns:
            A tuple of:
            - A Tensor of shape [N, D], the outputs.
            - A Tensor of shape [N], the log absolute determinants of the total transform.
        """
        squared_norms = torch.sum(q_vectors**2, dim=-1)
        outputs = inputs
        for q_vector, squared_norm in zip(q_vectors, squared_norms):
            temp = outputs @ q_vector  # Inner product.
            temp = torch.ger(temp, (2.0 / squared_norm) * q_vector)  # Outer product.
            outputs = outputs - temp
        batch_size = inputs.shape[0]
        logabsdet = torch.zeros(batch_size)
        return outputs, logabsdet

    def forward(self, inputs, context=None):
        return self._apply_transforms(inputs, self.q_vectors)

    def inverse(self, inputs, context=None):
        # Each householder transform is its own inverse, so the total inverse is given by
        # simply performing each transform in the reverse order.
        reverse_idx = torch.arange(self.num_transforms - 1, -1, -1)
        return self._apply_transforms(inputs, self.q_vectors[reverse_idx])

    def matrix(self):
        """Returns the orthogonal matrix that is equivalent to the total transform.

        Costs O(KD^2), where:
        - K is number of transforms
        - D is dimensionality of inputs

        Returns:
            A Tensor of shape [D, D].
        """
        identity = torch.eye(self.features, self.features)
        outputs, _ = self.inverse(identity)
        return outputs
