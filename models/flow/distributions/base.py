from typing import Optional, Tuple

import torch


class Distribution(torch.nn.Module):

    """
    Base class for all distribution objects.

    """

    def forward(
        self,
        *args,
    ) -> None:
        raise RuntimeError("forward() does not apply to distributions.")

    def log_prob(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        """
        Calculate log probability under the distribution.

        Args:
            inputs (torch.Tensor): input tensor
            context (torch.Tensor, optional): conditioning variables (one per input)

        Returns:
            torch.Tensor: log-prob of inputs, or conditional log-prob if context is given

        """

        if context is not None:
            assert inputs.size(0) == context.size(0)
        return self._log_prob(inputs, context)

    def sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:

        """
        Generates samples from the distribution.
        Samples can be generated in batches.

        Args:
            num_samples (int): number of samples to generate
            context (torch.Tensor, optional): conditioning variables
            batch_size (int, optional): if None (default), defaults to num_samples

        Returns:
            torch.Tensor: generated samples
                context == None - (num_samples, ...)
                context != None - (context_size, num_samples, ...)

        """
        assert num_samples > 0

        if batch_size is None:
            return self._sample(num_samples, context)

        else:
            assert batch_size > 0
            num_batches = num_samples // batch_size
            num_leftover = num_samples % batch_size
            samples = [self._sample(batch_size, context) for _ in range(num_batches)]
            if num_leftover > 0:
                samples.append(self._sample(num_leftover, context))
            return torch.cat(samples, dim=0)

    def sample_and_log_prob(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Generates samples from the distribution together with their log probability.

        Args:
            num_samples (int): number of samples to generate
            context (torch.Tensor, optional): conditioning variables

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: generated samples and their log-probs
                context == None - (num_samples, ...)
                context != None - (context_size, num_samples, ...)

        """
        samples = self.sample(num_samples, context=context)

        # if context is not None:
        #     # Merge the context dimension with sample dimension in order to call log_prob.
        #     samples = torchutils.merge_leading_dims(samples, num_dims=2)
        #     context = torchutils.repeat_rows(context, num_reps=num_samples)
        #     assert samples.shape[0] == context.shape[0]

        log_prob = self.log_prob(samples, context=context)

        # if context is not None:
        #     # Split the context dimension from sample dimension.
        #     samples = torchutils.split_leading_dim(samples, shape=[-1, num_samples])
        #     log_prob = torchutils.split_leading_dim(log_prob, shape=[-1, num_samples])

        return samples, log_prob

    def mean(
        self,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        return self._mean(context)

    def _log_prob(
        self,
        inputs: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()

    def _sample(
        self,
        num_samples: int,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()

    def _mean(
        self,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        raise NotImplementedError()
