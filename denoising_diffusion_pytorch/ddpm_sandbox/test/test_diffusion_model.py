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
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import logging
from PIL import Image
from os import listdir
from os.path import isfile, join

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def tensor_to_images(images_tensor: torch.Tensor):
    # Assume images of the shape is B X C X H X W where N is the number of images
    assert len(images_tensor.shape) == 4, "Images tensor must have dims B X C X H X W"
    # assert that C = 1, i.e. one channel and the image is BW
    assert images_tensor.shape[1] == 1, "Supporting BW images only with one channel, i.e. dim C = 1"
    images_tensor = torch.clamp(input=images_tensor, min=0, max=1) * 255.0
    images_tensor = images_tensor.squeeze()
    logger.info(
        f"After clamping and reverse-normalizing , min and max = {torch.min(images_tensor)}, {torch.max(images_tensor)}")
    images_tensors_list = list(images_tensor)
    prefix = "generated_img"
    for i, image_tensor in enumerate(images_tensors_list):
        img = Image.fromarray(image_tensor.detach().cpu().numpy()).convert("L")
        img.save(f"{prefix}_{i}.png")


if __name__ == '__main__':
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 64
    # num_train_step = 20_000
    model_checkpoint_ext = ".pt"
    checkpoint_metadata_ext = ".json"

    model_checkpoints_path = "../models/checkpoints/ddpm_mnist8"
    final_model_checkpoint_name = "checkpoint_model_10000.pt"
    final_model_path = os.path.join(model_checkpoints_path,final_model_checkpoint_name)
    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage

    logger.info(f"Creating UNet and Diffusion Models ")
    # UNet
    unet_model = Unet(
        dim=64,
        channels=num_channels,
        dim_mults=(1, 2, 4, 8),
        flash_attn=True
    ).to(device)

    # Double-checking if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_unet_model_on_cuda = next(unet_model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_unet_model_on_cuda}')

    diffusion = GaussianDiffusion(
        model=unet_model,
        image_size=image_size,
        timesteps=time_steps,  # number of steps
        auto_normalize=False
    ).to(device)

    is_diffusion_model_on_cuda = next(diffusion.parameters()).is_cuda
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
    baseline_sinkhorn_value = 40
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
    diffusion.load_state_dict(torch.load(final_model_path))
    logger.info(f"Successfully loaded model weights")
    #
    logger.info("Sampling images")
    sampled_images = diffusion.sample(batch_size=4)
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    logger.info(
        f"quantiles of levels {quantiles} = {torch.quantile(input=sampled_images, q=torch.tensor(quantiles).to(device))}")
    logger.info(f"Average of the sampled images {torch.mean(sampled_images)}")
    tensor_to_images(images_tensor=sampled_images)
    logger.info("Testing script finished")
