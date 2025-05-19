from defaults import MODELS_PATH, DATA_PATH, EVALS_PATH, device

import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tqdm
import imageio
import glob
import torch
import shutil

from oatomobile.datasets.carla import CARLADataset

from vae_model import VAEModel

from typing import List, Mapping
import pandas as pd
import matplotlib.pyplot as plt


def getVAEModel(path: str = None, device: str = "cpu") -> VAEModel:
    """load a DIM model from a checkpoint.

    Args:
        path (str, optional): The path of the checkpoint. If None, no checkpoint is loaded. Defaults to None.
        device (str, optional): the device on which the model should be stored. cuda or cpu. Defaults to "cpu".
        mobilenet_num_classes (int, optional): The output dimension of the mobilenet. Defaults to 128.

    Returns:
        ImitativeModel: the loaded DIM
    """
    model = VAEModel()
    if path is not None:
        x = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(x, strict=False)
    model.eval().to(device)
    return model


def transform(model: VAEModel, 
              batch: Mapping[str, torch.Tensor],
              device: str) -> Mapping[str, torch.Tensor]:
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


def ADE(pred: torch.Tensor, ground_truth: torch.Tensor):
    return (pred[:, :, :2] - ground_truth[:, :, :2]).square().sum(axis=-1).sqrt().mean(axis=-1)


def FDE(pred: torch.Tensor, ground_truth: torch.Tensor):
    return (pred[:, -1, :2] - ground_truth[:, -1, :2]).square().sum(axis=-1).sqrt()


def minADE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_ADE = [ADE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_ADE).min(axis=0)[0]


def minFDE(samples: List[torch.Tensor], ground_truth: torch.Tensor):
    samples_FDE = [FDE(sample, ground_truth) for sample in samples]
    return torch.stack(samples_FDE).min(axis=0)[0]


def evaluate(ckpt_path: str, 
             data_path: str, 
             output_path: str,  
             num_batches: int = np.inf, 
             k=5):
    """Calculate evaluation measures of a DIM model over a test data set and return the different measures.

    Args:
        ckpt_path (str): the path of the model checkpoint
        data_path (str): thepath of the test data used for the evaluation
        output_path (str): where the csv will be saved
        mobilenet_num_classes (int, optional): the output dimension of the mobilnet of the DIM. Defaults to 128.
        num_batches (int, optional): The number of batches of the test data over which the measures are averaged. Defaults to np.inf.
        k (int): the number of samples for minADE_k and minFdE_k

    Return:
        Dict[str, float]: the results for the different measures
    """
    model = getVAEModel(ckpt_path, device)
    modalities = (
        "lidar",
        "is_at_traffic_light",
        "traffic_light_state",
        "player_future",
        "velocity",
    )
    batch_size = 64
    dataloader = CARLADataset.as_torch(data_path, modalities)

    dataset = CARLADataset.as_torch(
        dataset_dir=data_path,
        modalities=modalities,
    )
    dataloader = torch.utils.data.DataLoader(  # errors with num_workers > 1
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    ADE_ms = []
    ADE_10s = []  # the number of steps
    ADE_100s = []
    minADE_ks = []
    FDE_ms = []
    FDE_10s = []
    FDE_100s = []
    minFDE_ks = []
    torch.cuda.empty_cache()
    with tqdm.tqdm(dataloader) as pbar:
        for i, batch in enumerate(pbar):  # gives errors with num_workers > 1

            batch = transform(model, batch, device)
            ground_truth = batch["player_future"]
            lengths = (ground_truth[:, 3, :]-ground_truth[:,
                       0, :]).square().sum(axis=1).sqrt()
            # [lengths>torch.zeros(lengths.size(),device="cuda")+0.1]
            ls = lengths
            print(ls.mean())
            y_samples = [model.sample(**batch).detach() for _ in range(k)]
            minADE_ks.append(minADE(y_samples, ground_truth))
            minFDE_ks.append(minFDE(y_samples, ground_truth))
            del y_samples
            y_mean = model.mean(**batch).detach()
            ADE_ms.append(ADE(y_mean, ground_truth))
            FDE_ms.append(FDE(y_mean, ground_truth))
            del y_mean
            
            # i skip the last batch since it could be of different size, could also just use something else than stack...
            if i >= num_batches-1:
                break
    measure_lists = ADE_ms, ADE_10s, ADE_100s, minADE_ks, FDE_ms, FDE_10s, FDE_100s, minFDE_ks
    measures = [torch.cat(measure, 0).mean().item() if len(
        measure) > 0 else np.nan for measure in measure_lists]
    ADE_m, ADE_10, ADE_100, minADE_k, FDE_m, FDE_10, FDE_100, minFDE_k = measures
    names = "ADE_m", "ADE_10", "ADE_100", "minADE_{}".format(
        k), "FDE_m", "FDE_10", "FDE_100", "minFDE_{}".format(k)
    vals = [val for val in measures]
    d = dict(zip(names, vals))
    return d


def evaluate_all_checkpoints(ckpts_path: str, 
                             data_path: str, 
                             output_path: str, 
                             mobilenet_num_classes=128, 
                             num_batches=np.inf, 
                             k=5):
    """Evaluate several measures on a model for all its saved checkpoints and store the result in a csv file.

    Args:
        ckpts_path (str): the path of the folder containing the checkpoints
        data_path (str): the path of the test data set
        output_path (str): the path of the output csv file
        mobilenet_num_classes (int, optional): The number of dimensions of the output of Mobilenet. Defaults to 128.
        num_batches ([type], optional): The number of batches of the . Defaults to np.inf.
        k (int, optional): the number of samples for minADE_k and minFDE_k. Defaults to 5.
    """
    dfs = {}
    for x in tqdm.tqdm(sorted(os.listdir(ckpts_path), key=lambda x: int(x.replace("model-", "").replace(".pt", "")))):
        if x.startswith("model-") and x.endswith(".pt"):
            epoch = int(x.replace("model-", "").replace(".pt", ""))
            ckpt_path = os.path.join(ckpts_path, x)
            df = evaluate(ckpt_path, data_path, output_path, num_batches, k)
            dfs[epoch] = df
    df = pd.DataFrame(dfs).transpose()

    plot(df, k, output_path)
    df.to_csv(output_path)


def plot(df: pd.DataFrame, k: int, output_path: str):
    """plot the measures over the epochnumber and save as pdf

    Args:
        df (pd.DataFrame): the dataframe conaining the evaluation measures
        k (int): the number of samples for minADE_k
        output_path (str): the plot ouput_path
    """
    plt.clf()
    plt.plot(df.index, df["FDE_m"], label="FDE")
    plt.plot(df.index, df["ADE_m"], label="ADE")
    plt.plot(df.index, df["minFDE_{}".format(k)], label="minFDE_{}".format(k))
    plt.plot(df.index, df["minADE_{}".format(k)], label="minADE_{}".format(k))
    plt.xlabel("epoch")
    plt.ylabel("meter")
    plt.ylim(0)
    plt.legend()
    plt.savefig(output_path+".pdf", bbox_inches='tight')


def plot_wrapper(csv_path: str, k: int):
    """load csv and save plot as pdf

    Args:
        csv_path (str): the path of the csv
        k (int): the number of samples for minADE_k
    """
    df = pd.read_csv(csv_path)
    plot(df, k, csv_path)


if __name__ == "__main__":
    k = 7
    ckpt_path = os.path.join(MODELS_PATH)
    data_path = os.path.join(DATA_PATH, "processed", "val")
    output_path = os.path.join(EVALS_PATH, "evalution.csv")
    evaluate_all_checkpoints(ckpt_path, data_path, output_path, mobilenet_num_classes=128, k=k)
    plot_wrapper(output_path, k=k)
