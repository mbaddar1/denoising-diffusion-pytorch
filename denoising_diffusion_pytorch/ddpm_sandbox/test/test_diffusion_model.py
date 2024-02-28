"""
This script is for testing diffusion models checkpoints, generated from this training script
denoising-diffusion-pytorch/denoising_diffusion_pytorch/ddpm_sandbox/ddpm_trainer.py
and which are saved here
denoising-diffusion-pytorch/denoising_diffusion_pytorch/models/checkpoints
"""
import json
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
from denoising_diffusion_pytorch import Unet2D, GaussianDiffusion
import logging
from PIL import Image
from os import listdir
from os.path import isfile, join

from denoising_diffusion_pytorch.denoising_diffusion_pytorch import HeadTail

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def tensor_to_images(sample_tensor: torch.Tensor, dataset_name: str):
    if dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
        # Assume images of the shape is B X C X H X W where N is the number of images
        assert len(sample_tensor.shape) == 4, "Images tensor must have dims B X C X H X W"
        # assert that C = 1, i.e., one channel and the image is BW
        assert sample_tensor.shape[1] == 1, "Supporting BW images only with one channel, i.e. dim C = 1"
        sample_tensor = torch.clamp(input=sample_tensor, min=0, max=1) * 255.0
        sample_tensor = sample_tensor.squeeze()
        logger.info(
            f"After clamping and reverse-normalizing , min and max = {torch.min(sample_tensor)}, {torch.max(sample_tensor)}")
        images_tensors_list = list(sample_tensor)
        prefix = "generated_img"
        for i, image_tensor in enumerate(images_tensors_list):
            img = Image.fromarray(image_tensor.detach().cpu().numpy()).convert("L")
            out_filename = f"./generated_images/{dataset_name}_{prefix}_{i}.png"
            img.save(out_filename)
            logger.info(f"Successfully written output file to {out_filename}")
    elif dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
        sample_tensor_np = sample_tensor.cpu().detach().numpy()
        plt.clf()
        plt.scatter(sample_tensor_np[:,0], sample_tensor_np[:,1])
        out_filename = f"./generated_images/{dataset_name}.png"
        plt.savefig(out_filename)
        logger.info(f"Successfully written output file to {out_filename}")
    else:
        raise ValueError(f"Unknown dataset_name : {dataset_name}")


if __name__ == '__main__':
    time_steps = 50
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 32
    # num_train_step = 20_000
    model_name = "ddpm_nn"
    dataset_name = "mnist8"
    # dataset_name = "circles"
    model_checkpoint_ext = ".pt"
    checkpoint_metadata_ext = ".json"
    diffusion_model_objective="pred_noise"
    model_checkpoints_path = f"../models/checkpoints/{model_name}_{dataset_name}"
    final_model_checkpoint_name = "checkpoint_model_5000.pt"
    final_model_path = os.path.join(model_checkpoints_path, final_model_checkpoint_name)
    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage

    logger.info(f"Creating Noise Model.")
    # UNet
    if dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
        noise_model = Unet2D(
            dim=64,
            channels=num_channels,
            dim_mults=(1, 2, 4, 8),
            flash_attn=True
        ).to(device)
        diffusion_model = GaussianDiffusion(
            dataset_name="mnist6",
            noise_model=noise_model,
            image_size=image_size,
            timesteps=time_steps,  # number of steps
            auto_normalize=False,
            objective=diffusion_model_objective
        ).to(device)
        sample_batch_size = 4
    elif dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
        noise_model = HeadTail(hidden_dim=128, input_dim=2, time_steps=time_steps).to(device)
        diffusion_model = GaussianDiffusion(noise_model=noise_model, dataset_name=dataset_name,
                                            timesteps=time_steps, objective=diffusion_model_objective).to(device)
        sample_batch_size = 1024
    else:
        raise ValueError(f"Unknown dataset_name : {dataset_name}")
    # Double-checking if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_unet_model_on_cuda = next(noise_model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_unet_model_on_cuda}')

    is_diffusion_model_on_cuda = next(diffusion_model.parameters()).is_cuda
    logger.info(f'Is diffusion model on cuda? : {is_diffusion_model_on_cuda}')
    #
    logger.info(f"Loading model checkpoints metadata from {model_checkpoints_path}")
    # https://stackoverflow.com/a/3207973
    all_files = [f for f in listdir(model_checkpoints_path) if
                 isfile(join(model_checkpoints_path, f)) and f.endswith(checkpoint_metadata_ext)]
    checkpoint_niters = sorted([int(f.split(".")[0].split("_")[2]) for f in all_files])
    max_niters = max(checkpoint_niters)
    logger.info(f"Finished")
    one_metadata_filename = all_files[0]
    ema_losses = []
    sinkhorn_distances = []
    sinkhorn_std = []
    plot_start_index = 1
    if dataset_name == "mnist8":
        baseline_sinkhorn_value = 40
    elif dataset_name == "circles":
        baseline_sinkhorn_value = 0.06
    elif dataset_name == "mnist6":
        baseline_sinkhorn_value = 35 # just adhoc val. need to set it correctly
    else:
        raise ValueError(f"Still have no baseline sinkhorn distance for dataset_name = {dataset_name}")
    for iter_num in checkpoint_niters:
        tmp_list = one_metadata_filename.split(".")[0].split("_")[:2]
        tmp_list.append(str(iter_num))
        filename = "_".join(tmp_list) + ".json"

        with open(os.path.join(model_checkpoints_path, filename)) as f:
            meta_data = json.load(f)
            sinkhorn_distances.append(float(meta_data["sinkhorn_dist_avg"]))
            ema_losses.append(float(meta_data["ema_loss"]))
            sinkhorn_std.append(float(meta_data["sinkhorn_dist_std"]))
    fig, ax = plt.subplots()
    x = checkpoint_niters[plot_start_index:]
    y = [baseline_sinkhorn_value] * len(x)
    ax.plot(x, y, linestyle="dotted", label="baseline sinkhorn distance")

    ci = np.array([1.96 * s / np.sqrt(5) for s in sinkhorn_std[plot_start_index:]])
    y = sinkhorn_distances[plot_start_index:]
    ax.plot(x, y)

    ax.fill_between(x, (y - ci), (y + ci), color='b', alpha=.1, linestyle="solid",
                    label="sinkhorn distances over iteration")
    # plt.plot(x, y, linestyle="solid", label="sinkhorn distances over iteration")
    plt.title("Sinkhorn distance vs Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Sinkhorn distance average")
    plt.legend(loc="upper left")
    plt.savefig(f"iter_sinkhorn.png")

    y = ema_losses[plot_start_index:]
    plt.clf()
    plt.xlabel("Iterations")
    plt.ylabel("EMA for L2 loss")
    plt.plot(x, y, label="loss vs iterations")
    plt.legend(loc="upper left")

    plt.savefig("iter_loss.png")
    logger.info(f"loading model weights from file {final_model_path}")
    diffusion_model.load_state_dict(torch.load(final_model_path))
    logger.info(f"Successfully loaded model weights")
    #
    logger.info("Sampling images")
    sampled_images = diffusion_model.sample(batch_size=sample_batch_size)
    # FIXME No need now for this quantile analysis , remove it later
    # quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    # logger.info(
    #     f"quantiles of levels {quantiles} = {torch.quantile(input=sampled_images, q=torch.tensor(quantiles).to(device))}")
    # logger.info(f"Average of the sampled images {torch.mean(sampled_images)}")
    tensor_to_images(sample_tensor=sampled_images, dataset_name=dataset_name)
    logger.info("Testing script finished")
