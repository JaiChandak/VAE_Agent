from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as D

from oatomobile.torch import types
from oatomobile.torch import transforms

from oatomobile.torch.networks.perception import MobileNetV2
from oatomobile.torch.networks.sequence import AutoregressiveFlow
from oatomobile.torch.networks.mlp import MLP

class VAEModel(nn.Module):
    """A `PyTorch` implementation of a variational autoencoder model."""

    def __init__(
        self,
        output_shape: types.Shape = (4, 2),
    ) -> None:
        
        super(VAEModel, self).__init__()
        self._output_shape = output_shape

        # CNN Encoder
        self._encoder = MobileNetV2(num_classes=128, in_channels=2)

        # MLP Merger
        self._merger = MLP(
            input_size=128 + 3 + 1 + 1,
            output_sizes=[64, 64, 64],
            activation_fn=nn.ReLU,
            dropout_rate=None,
            activate_final=True,
        )

        # MLP Estimators
        self._mean_layer = MLP(
            input_size=64,
            output_sizes=[64, 64, 32],
            activation_fn=nn.ReLU,
            dropout_rate=None,
            activate_final=False,
            )
        
        self._log_var_layer = MLP(
            input_size=64,
            output_sizes=[64, 64, 32],
            activation_fn=nn.ReLU,
            dropout_rate=None,
            activate_final=False,
            )
        
        # GRU Decoder
        self._decoder = AutoregressiveFlow(
            output_shape=self._output_shape,
            hidden_size=32,
        )
    
    def visual_features(self, **context: torch.Tensor) -> torch.Tensor:
        # Parses context variables.
        if not "visual_features" in context:
            raise ValueError("Missing `visual_features` keyword argument.")
        batch_size = context["visual_features"].shape[0]
        if not "velocity" in context:
            raise ValueError("Missing `velocity` keyword argument.")
        if not "is_at_traffic_light" in context:
            raise ValueError("Missing `is_at_traffic_light` keyword argument.")
        if not "traffic_light_state" in context:
            raise ValueError("Missing `traffic_light_state` keyword argument.")
        visual_features = context.get("visual_features")
        velocity = context.get("velocity")
        is_at_traffic_light = context.get("is_at_traffic_light")
        traffic_light_state = context.get("traffic_light_state")

        # Encodes the visual input.
        visual_features = self._encoder(visual_features)

        # Merges visual input logits and vector inputs.
        visual_features = torch.cat(
            tensors=[
                visual_features,
                velocity,
                is_at_traffic_light,
                traffic_light_state,
            ],
            dim=-1,
        )

        visual_features = self._merger(visual_features)

        return visual_features, batch_size
    
    def latent_sampler(self, visual_features):
        # Extract mean and var vectors
        mean_vector, log_var_vector = self._mean_layer(visual_features), self._log_var_layer(visual_features)
        log_var_vector_exp = torch.exp(log_var_vector)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epsilon = torch.randn_like(log_var_vector).to(device)
        
        z = mean_vector + log_var_vector_exp*epsilon
        return z, mean_vector, log_var_vector
    
    def _goal_likelihood(self, y: torch.Tensor, goal: torch.Tensor,
                       **hyperparams) -> torch.Tensor:
        """Returns the goal-likelihood of a plan `y`, given `goal`.

        Args:
        y: A plan under evaluation, with shape `[B, T, 2]`.
        goal: The goal locations, with shape `[B, K, 2]`.
        hyperparams: (keyword arguments) The goal-likelihood hyperparameters.

        Returns:
        The log-likelihodd of the plan `y` under the `goal` distribution.
        """
        # Parses tensor dimensions.
        B, K, _ = goal.shape

        # Fetches goal-likelihood hyperparameters.
        epsilon = hyperparams.get("epsilon", 1.0)

        # Initializes the goal distribution.
        goal_distribution = D.MixtureSameFamily(
            mixture_distribution=D.Categorical(
                probs=torch.ones((B, K)).to(goal.device)),
            component_distribution=D.Independent(
                D.Normal(loc=goal, scale=torch.ones_like(goal) * epsilon),
                reinterpreted_batch_ndims=1,
            ))

        return torch.mean(goal_distribution.log_prob(y[:, -1, :]), dim=0)

    def forward(
        self,
        num_steps: int,
        batch_size,
        z: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
        lr: float = 1e-1,
        epsilon: float = 1.0,
        **context: torch.Tensor
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:

        # Sets initial sample to base distribution's mean.
        # Initialise a gradient-based optimiser.
        x = self._decoder._base_dist.sample().clone().detach().repeat(
            batch_size, 1).view(
                batch_size,
                *self._output_shape,
            )
        x.requires_grad = True

        optimizer = optim.Adam(params=[x], lr=lr)

        # Stores the best values.
        x_best = x.clone()
        loss_best = torch.ones(()).to(x.device) * 1000.0

        for _ in range(num_steps):
            # Resets optimizer's gradients.
            optimizer.zero_grad()
            # Operate on `y`-space.
            y, _ = self._decoder._forward(x=x, z=z)
            # Calculates imitation prior.
            _, log_prob, logabsdet = self._decoder._inverse(y=y, z=z)
            imitation_prior = torch.mean(log_prob - logabsdet)
            # Calculates goal likelihodd.
            goal_likelihood = 0.0
            if goal is not None:
                goal_likelihood = self._goal_likelihood(y=y, goal=goal, epsilon=epsilon)
            loss = -(imitation_prior + goal_likelihood)
            # Backward pass.
            loss.backward(retain_graph=True)
            # Performs a gradient descent step.
            optimizer.step()
            # Book-keeping
            if loss < loss_best:
                x_best = x.clone()
                loss_best = loss.clone()

        y, _ = self._decoder._forward(x=x_best, z=z)

        return y, loss

    def to(self, *args, **kwargs):
        """Handles non-parameter tensors when moved to a new device."""
        self = super().to(*args, **kwargs)
        self._decoder = self._decoder.to(*args, **kwargs)
        return self

    def transform(
        self,
        sample: Mapping[str, types.Array],
    ) -> Mapping[str, torch.Tensor]:
        """Prepares variables for the interface of the model.
        Args:
        sample: (keyword arguments) The raw sample variables.

        Returns:
        The processed sample.
        """

        # Preprocesses the target variables.
        if "player_future" in sample:
            sample["player_future"] = transforms.downsample_target(
                player_future=sample["player_future"],
                num_timesteps_to_keep=self._output_shape[-2],
            )

        # Renames `lidar` to `visual_features`.
        if "lidar" in sample:
            sample["visual_features"] = sample.pop("lidar")

        # Preprocesses the visual features.
        if "visual_features" in sample:
            sample["visual_features"] = transforms.transpose_visual_features(
                transforms.downsample_visual_features(
                    visual_features=sample["visual_features"],
                    output_shape=(100, 100),
                ))

        return sample
    
    def mean(self,
             **context: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Returns a local mode from the posterior.
        Args:
        context: (keyword arguments) The conditioning
            variables used for the conditional flow.

        Returns:
        A mode from the posterior, with shape `[D, 2]`.
        """
        visual_features, batch_size = self.visual_features(**context)

        # Sets initial sample to base distribution's mean.
        x = self._decoder._base_dist.mean.clone().detach().repeat(
            batch_size, 1).view(
                batch_size,
                *self._output_shape,
            )

        z, _, __ = self.latent_sampler(visual_features)

        y, _ = self._decoder._forward(x=x, z=z)

        return y
    
    def sample(
        self,
        **context: torch.Tensor) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        """Returns a local mode from the posterior.
        Args:
        context: (keyword arguments) The conditioning
        variables used for the conditional flow.

        Returns:
        A mode from the posterior, with shape `[D, 2]`.
        """
        visual_features, batch_size = self.visual_features(**context)

        # Sets initial sample to base distribution's mean.
        x = self._decoder._base_dist.mean.clone().detach().repeat(
            batch_size, 1).view(
                batch_size,
                *self._output_shape,
            )

        z, _, __ = self.latent_sampler(visual_features)

        y, _ = self._decoder._forward(x=x, z=z)

        return y
