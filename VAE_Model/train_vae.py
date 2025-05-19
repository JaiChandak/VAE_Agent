import os
from typing import Mapping

import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from absl import app
from absl import flags
from absl import logging

from vae_model import VAEModel
from oatomobile.datasets.carla import CARLADataset
from oatomobile.torch import types
from oatomobile.torch.loggers import TensorBoardLogger
from oatomobile.torch.savers import Checkpointer

logging.set_verbosity(logging.DEBUG)
FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="dataset_dir",
    default=None,
    help="The full path to the processed dataset.",
)
flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="The full path to the output directory (for logs, ckpts).",
)
flags.DEFINE_integer(
    name="batch_size",
    default=256,
    help="The batch size used for training the neural network.",
)
flags.DEFINE_integer(
    name="num_epochs",
    default=501,
    help="The number of training epochs for the neural network.",
)
flags.DEFINE_integer(
    name="save_model_frequency",
    default=3,
    help="The number epochs between saves of the model.",
)
flags.DEFINE_float(
    name="learning_rate",
    default=1e-3,
    help="The ADAM learning rate.",
)
flags.DEFINE_integer(
    name="num_timesteps_to_keep",
    default=4,
    help="The numbers of time-steps to keep from the target, with downsampling.",
)
flags.DEFINE_float(
    name="weight_decay",
    default=0.0,
    help="The L2 penalty (regularization) coefficient.",
)
flags.DEFINE_bool(
    name="clip_gradients",
    default=False,
    help="If True it clips the gradients norm to 1.0.",
)

def main(argv):
  # Debugging purposes.
  logging.debug(argv)
  logging.debug(FLAGS)

  # Parses command line arguments.
  dataset_dir = FLAGS.dataset_dir
  output_dir = FLAGS.output_dir
  batch_size = FLAGS.batch_size
  num_epochs = FLAGS.num_epochs
  learning_rate = FLAGS.learning_rate
  save_model_frequency = FLAGS.save_model_frequency
  num_timesteps_to_keep = FLAGS.num_timesteps_to_keep
  weight_decay = FLAGS.weight_decay
  clip_gradients = FLAGS.clip_gradients
  noise_level = 1e-2

  # Determines device, accelerator.
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Creates the necessary output directory.
  os.makedirs(output_dir, exist_ok=True)
  log_dir = os.path.join(output_dir, "logs")
  os.makedirs(log_dir, exist_ok=True)
  ckpt_dir = os.path.join(output_dir, "ckpts")
  os.makedirs(ckpt_dir, exist_ok=True)

  # Initializes the model and its optimizer.
  output_shape = [num_timesteps_to_keep, 2]
  model = VAEModel(output_shape=output_shape).to(device)
  
  optimizer = optim.Adam(
      model.parameters(),
      lr=learning_rate,
      weight_decay=weight_decay,
  )
  writer = TensorBoardLogger(log_dir=log_dir)
  checkpointer = Checkpointer(model=model, ckpt_dir=ckpt_dir)

  def transform(batch: Mapping[str, types.Array]) -> Mapping[str, torch.Tensor]:
    """Preprocesses a batch for the model.

    Args:
      batch: (keyword arguments) The raw batch variables.

    Returns:
      The processed batch.
    """
    # Sends tensors to `device`.
    batch = {key: tensor.to(device) for (key, tensor) in batch.items()}
    # Preprocesses batch for the model.
    batch = model.transform(batch)
    return batch

  # Setups the dataset and the dataloader.
  modalities = (
      "lidar",
      "is_at_traffic_light",
      "traffic_light_state",
      "player_future",
      "velocity",
  )
  dataset_train = CARLADataset.as_torch(
      dataset_dir=os.path.join(dataset_dir, "train"),
      modalities=modalities,
  )
  dataloader_train = torch.utils.data.DataLoader(
      dataset_train,
      batch_size=batch_size,
      shuffle=True,
      num_workers=6,
  )
  dataset_val = CARLADataset.as_torch(
      dataset_dir=os.path.join(dataset_dir, "val"),
      modalities=modalities,
  )
  dataloader_val = torch.utils.data.DataLoader(
      dataset_val,
      batch_size=batch_size * 5,
      shuffle=True,
      num_workers=6,
  )

  def train_step(
      model: VAEModel,
      optimizer: optim.Optimizer,
      batch: Mapping[str, torch.Tensor],
      beta: float,
      clip: bool = False,
  ) -> torch.Tensor:
    """Performs a single gradient-descent optimisation step."""
    # Resets optimizer's gradients.
    optimizer.zero_grad()

    # Perturb target.
    y = torch.normal(
        mean=batch["player_future"][..., :2],
        std=torch.ones_like(batch["player_future"][..., :2]) * noise_level,
    )

    # Forward pass from the model.
    visual_features, batch_size = model.visual_features(
        velocity=batch["velocity"],
        visual_features=batch["visual_features"],
        is_at_traffic_light=batch["is_at_traffic_light"],
        traffic_light_state=batch["traffic_light_state"],
    )

    z, mean_vector, log_var_vector = model.latent_sampler(visual_features)
    _, log_prob, logabsdet = model._decoder._inverse(y=y, z=z)

    # Calculates loss (NLL)
    NLL_loss = -torch.mean(log_prob - logabsdet, dim=0)

    # Calculates loss (KL)
    KL_loss = - 0.5 * torch.sum(1+ log_var_vector - mean_vector.pow(2) - log_var_vector.exp())

    loss = NLL_loss + (beta * KL_loss)

    # Backward pass.
    loss.backward()

    # Clips gradients norm.
    if clip:
      torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)

    # Performs a gradient descent step.
    optimizer.step()

    return loss

  def train_epoch(
      model: VAEModel,
      optimizer: optim.Optimizer,
      dataloader: torch.utils.data.DataLoader,
      beta: float,
  ) -> torch.Tensor:
    """Performs an epoch of gradient descent optimization on `dataloader`."""
    model.train()
    loss = 0.0
    with tqdm.tqdm(dataloader) as pbar:
      for batch in pbar:
        # Prepares the batch.
        batch = transform(batch)
        
        # Performs a gradien-descent step.
        loss += train_step(model, optimizer, batch, beta, clip=clip_gradients)
    return loss / len(dataloader)

  def evaluate_step(
      model: VAEModel,
      batch: Mapping[str, torch.Tensor],
      beta: float,
  ) -> torch.Tensor:
    """Evaluates `model` on a `batch`."""
    # Forward pass from the model.

    visual_features, batch_size = model.visual_features(
        velocity=batch["velocity"],
        visual_features=batch["visual_features"],
        is_at_traffic_light=batch["is_at_traffic_light"],
        traffic_light_state=batch["traffic_light_state"],
    )

    z, mean_vector, log_var_vector = model.latent_sampler(visual_features)

    _, log_prob, logabsdet = model._decoder._inverse(
        y=batch["player_future"][..., :2],
        z=z,
    )

    # Calculates loss (NLL)
    NLL_loss = -torch.mean(log_prob - logabsdet, dim=0)

    # Calculates loss (KL)
    KL_loss = - 0.5 * torch.sum(1+ log_var_vector - mean_vector.pow(2) - log_var_vector.exp())

    loss = NLL_loss + (beta * KL_loss)

    return loss

  def evaluate_epoch(
      model: VAEModel,
      dataloader: torch.utils.data.DataLoader,
      beta: float,
  ) -> torch.Tensor:
    """Performs an evaluation of the `model` on the `dataloader."""
    model.eval()
    loss = 0.0
    with tqdm.tqdm(dataloader) as pbar:
      for batch in pbar:
        # Prepares the batch.
        batch = transform(batch)

        # Accumulates loss in dataset.
        with torch.no_grad():
          loss += evaluate_step(model, batch, beta)
    return loss / len(dataloader)

  with tqdm.tqdm(range(1,num_epochs)) as pbar_epoch:
    beta = 0
    beta_increment = 0.001
    for epoch in pbar_epoch:
      print(f"Epoch: {epoch}")
      if epoch % 10 == 0:
        beta += beta_increment
      print(f"Beta: {beta}")
      # Trains model on whole training dataset.
      loss_train = train_epoch(model, optimizer, dataloader_train, beta=beta)

      # Evaluates model on whole validation dataset.
      loss_val = evaluate_epoch(model, dataloader_val, beta=beta)

      # Checkpoints model weights.
      if epoch % save_model_frequency == 0:
        checkpointer.save(epoch)
      
      pbar_epoch.set_description(
          "TL: {:.2f} | VL: {:.2f}".format(
              loss_train.detach().cpu().numpy().item(),
              loss_val.detach().cpu().numpy().item()
          ))

if __name__ == "__main__":
  flags.mark_flag_as_required("dataset_dir")
  flags.mark_flag_as_required("output_dir")
  app.run(main)
