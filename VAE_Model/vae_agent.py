from typing import Mapping

import numpy as np
import scipy.interpolate
import torch
import torch.optim as optim

import oatomobile
from oatomobile.baselines.base import SetPointAgent
from vae_model import VAEModel

class VAEAgent(SetPointAgent):
    """
    The VAE implemented agent.
    """

    def __init__(self, environment: oatomobile.Env, *, model: VAEModel,
               **kwargs) -> None:
        """
        Args:
        environment: The navigation environment to spawn the agent.
        model: The vae model.
        """
        super(VAEAgent, self).__init__(environment=environment, **kwargs)
        # Determines device, accelerator.
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self._device = torch_directml.device()
        self._model = model.to(self._device)

    def __call__(self, observation: Mapping[str, np.ndarray],
               **kwargs) -> np.ndarray:
        """Returns the imitative prior."""

        # Prepares observation for the neural-network.
        observation["overhead_features"] = observation["bird_view_camera_cityscapes"]
        for attr in observation:
            if not isinstance(observation[attr], np.ndarray):
                observation[attr] = np.atleast_1d(observation[attr])
            observation[attr] = observation[attr][None, ...].astype(np.float32)

        # Makes `goal` 2D.        
        observation["goal"] = observation["goal"][..., :2]
        # Convert image to CHW.
        observation["lidar"] = np.transpose(observation["lidar"], (0, 3, 1, 2))
        # Processes observations for the `VAEModel`.
        observation = {
            key: torch.from_numpy(tensor).to(self._device)
            for (key, tensor) in observation.items()
        }
        observation = self._model.transform(observation)

        # Get plan from the model

        ######### AVGERAGED TRAJECTORY #############
        # visual_features, batch_size = self._model.visual_features(**observation)

        # num_samples = 100
        # zs = list()
        # for _ in range(num_samples):
        #     z, _, __ = self._model.latent_sampler(visual_features)
        #     zs.append(z)
        # z_mean = torch.mean(torch.stack(zs), dim=0)

        # num_steps = 15
        # posterior_threshold = -20.0
        # trajectories = list()
        # for _ in range(num_steps):
        #     trajectory, posterior = self._model(num_steps=kwargs.get("num_steps", 20),
        #                                         batch_size = batch_size,
        #                                         z = z_mean,
        #                                         epsilon=kwargs.get("epsilon", 1.0),
        #                                         lr=kwargs.get("lr", 2e-3),
        #                                         **observation)
        #     if posterior < posterior_threshold:
        #         trajectories.append(trajectory.detach().cpu().numpy()[0])

        # plan = np.mean(trajectories, axis=0)

        ######### MAP ALGORITHM #############
        visual_features, batch_size = self._model.visual_features(**observation)

        num_samples = 10
        zs = list()
        for _ in range(num_samples):
            z, _, __ = self._model.latent_sampler(visual_features)
            zs.append(z)
        z_mean = torch.mean(torch.stack(zs), dim=0)

        num_steps = 10
        posterior_best = 100.0
        for _ in range(num_steps):
            trajectory, posterior = self._model(num_steps=kwargs.get("num_steps", 15),
                                                batch_size = batch_size,
                                                z = z_mean,
                                                # epsilon=kwargs.get("epsilon", 1.0),
                                                epsilon=kwargs.get("epsilon", 0.95),
                                                lr=kwargs.get("lr", 15e-4),
                                                **observation)
            
            if posterior < posterior_best:
                posterior_best = posterior
                plan = trajectory.detach().cpu().numpy()[0]  # [T, 2]

        # Interpolates plan.
        player_future_length = 40
        increments = player_future_length // plan.shape[0]  
        time_index = list(range(0, player_future_length, increments))  # [T]
        plan_interp = scipy.interpolate.interp1d(x=time_index, y=plan, axis=0)
        xy = plan_interp(np.arange(0, time_index[-1]))

        # Appends z dimension.
        z = np.zeros(shape=(xy.shape[0], 1))
        return np.c_[xy, z]
