"""
This script is an attempt to write my own trainer for DDPM based on the classes and methods in
denoising-diffusion-pytorch/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py.
The trainer will be also based on the Trainer class in denoising_diffusion_pytorch.denoising_diffusion_pytorch.Trainer

Running instructions :
=========================
* Before each run
----
1. Check all params
2. Clear checkpoints path, based on the param. of model and dataset
3. Make sure the log is saved (via IDE or other way)

* After each run
-----
1. Document the results in the relevant issue
2. Add quick piece of documentation in the PhD thesis with link to the GitHub issue. Write properly later
3. Save the log file as an attachment to the ticket
"""
import json
import os.path
from typing import Tuple

import numpy as np
from PIL import Image
from denoising_diffusion_pytorch import GaussianDiffusion, Unet
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms as T
from torch.optim import Optimizer, Adam
from tqdm import tqdm
from datetime import datetime

from stat_dist.layers import SinkhornDistance

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# similar to denoising_diffusion_pytorch.denoising_diffusion_pytorch.Dataset
class Dataset2(Dataset):
    EXTENSIONS = ['jpg', 'png']

    def __init__(self, folder: str, image_size: int, debug_flag: bool = False):
        super().__init__()
        self.debug_flag = debug_flag
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in Dataset2.EXTENSIONS for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        # based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L838
        self.transform = T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        # Line for learning purposes. Raw values are from 0 to 255.
        #   See this post https://discuss.pytorch.org/t/pil-image-and-its-normalisation/78812/4?u=mbaddar
        raw_img_data = np.array(img)  # FIXME for debugging only , remove later
        if self.debug_flag:
            old_level = logger.level
            logger.setLevel(logging.DEBUG)
            logger.debug(f"raw img data =\n {raw_img_data}")
            logger.setLevel(old_level)
        transformed_image = self.transform(img)
        return transformed_image

        # TODO Some coding notes (nothing todo, just for highlighting) :
        # --------------------
        # img = read_image(str(path))
        # transformed_image = torch.squeeze(img)  # Just squeezing dims with value = 1 , no actual transformation
        # Cannot do squeezing as code is designed to each image with channel, height and width : see this error
        # File "/home/mbaddar/Documents/mbaddar/phd/genmodel/denoising-diffusion-pytorch/denoising_diffusion_pytorch/
        #   denoising_diffusion_pytorch.py", line 854, in forward
        #     b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        # ValueError: not enough values to unpack (expected 6, got 5)
        # As long as we have validated loaded data against passed num_channel and image size , it should be fine


class Trainer2:
    def __init__(self, diffusion_model: GaussianDiffusion,
                 batch_size: int,
                 train_num_steps: int,
                 device: torch.device,
                 dataset: Dataset,
                 optimizer: Optimizer,
                 progress_bar_update_freq: int,
                 checkpoint_freq: int,
                 debug_flag: bool,
                 checkpoints_path: str):
        self.diffusion_model = diffusion_model
        self.debug_flag = debug_flag
        self.device = device
        self.train_num_steps = train_num_steps
        self.optimizer = optimizer
        self.step = 0
        self.checkpoint_freq = checkpoint_freq
        self.progress_bar_update_freq = progress_bar_update_freq
        self.checkpoints_path = checkpoints_path
        self.batch_size = batch_size
        #
        logger.info(f"Creating data loader for the dataset")
        # FIXME separate train and test datasets
        self.train_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        if self.debug_flag:
            self.__validate_dataset()  # FIXME remove later

    def sinkhorn_eval(self, generated_sample: torch.Tensor, num_iter: int) -> Tuple[float, float]:
        dist_list = []
        assert len(generated_sample.shape) == 4
        for _ in range(num_iter):
            ref_sample = next(iter(self.test_dataloader)).to(self.device)
            assert len(generated_sample.shape) == len(ref_sample.shape)
            for j in range(len(generated_sample.shape)):
                assert generated_sample.shape[j] == ref_sample.shape[j]
            # shape assume to be B X C X H X W
            B, C, H, W = (
                generated_sample.shape[0], generated_sample.shape[1], generated_sample.shape[2],
                generated_sample.shape[3])
            assert C == 1, "Assume BW images"
            generated_sample_flat = generated_sample.squeeze().reshape(B, H * W)
            ref_sample_flat = ref_sample.squeeze().reshape(B, H * W)
            sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=self.device)
            dist, P, C = sinkhorn(generated_sample_flat, ref_sample_flat)
            dist_list.append(dist.item())
        return np.nanmean(dist_list), np.std(dist_list)

    def train(self):
        # Use manual control for tqdm and progress bar update
        # See https://github.com/tqdm/tqdm#usage
        # and   https://stackoverflow.com/a/45808255
        self.step = 0
        # Exponential smoothing for loss reporting
        # Called Exponential Smoothing or Exponential Moving Average
        ema_loss = 0.0  # Just for initialization
        ema_loss_alpha = 0.9
        start_timestamp = datetime.now()
        with tqdm(initial=self.step, total=self.train_num_steps) as progress_bar:
            while self.step <= self.train_num_steps:  # note the boundary condition
                self.optimizer.zero_grad()
                data = next(iter(self.train_dataloader)).to(device)
                loss = self.diffusion_model(data)
                if self.step == 0:
                    ema_loss = loss.item()
                else:
                    ema_loss = ema_loss_alpha * loss.item() + (1 - ema_loss_alpha) * ema_loss
                loss.backward()
                self.optimizer.step()
                # TODO add
                #   1. loss curve tracking
                #   2. sampling code
                #   3. quality measure for samples , along with loss (FID  , WD , etc..)
                if self.step % self.progress_bar_update_freq == 0:
                    progress_bar.set_description(f'loss: {ema_loss:.4f}')
                if self.step % self.checkpoint_freq == 0:
                    # sinkhorn eval
                    logger.info(f"Generating a sample for sinkhorn evaluation.")
                    generated_sample = self.diffusion_model.sample(batch_size=self.batch_size)
                    sh_dist_avg, sh_dist_std = self.sinkhorn_eval(generated_sample=generated_sample, num_iter=3)
                    checkpoint_metadata = json.dumps({"ema_loss": ema_loss, "sinkhorn_dist_avg": sh_dist_avg,
                                                      "sh_dist_std": sh_dist_std})
                    checkpoint_metadata_filename = f"checkpoint_metadata_{self.step}.json"
                    with open(os.path.join(self.checkpoints_path, checkpoint_metadata_filename), "w") as f:
                        f.write(checkpoint_metadata)
                        f.close()
                    # check point save
                    checkpoint_file_name = f"checkpoint_model_{self.step}.pt"
                    logger.info(f"Saving check point : {checkpoint_file_name} to {self.checkpoints_path}")
                    torch.save(self.diffusion_model.state_dict(),
                               os.path.join(self.checkpoints_path, checkpoint_file_name))
                    # TODO save sinkhorn dist, loss into another checkpoint metadata file
                self.step += 1
                progress_bar.update(1)
        end_datetime = datetime.now()
        elapsed_time = (end_datetime - start_timestamp).seconds
        logger.info(f"Training time = {elapsed_time} seconds")

    # private method
    # https://www.geeksforgeeks.org/private-methods-in-python/
    def __validate_dataset(self):
        """
        Test the coherence of loaded data to diffusion model data-related parameters
        """

        num_test_iters = 10
        quantiles = [0., 0.25, 0.5, 0.75, 1.0]
        for i in range(num_test_iters):
            data = next(iter(self.train_dataloader))
            old_level = logger.level
            logger.setLevel(logging.DEBUG)  # would it work ?
            logger.debug(f"Data batch # {i + 1} loaded with dimensions : {data.shape}")
            logger.debug(
                f"For quantiles at levels {quantiles} = {torch.quantile(input=data, q=torch.tensor(quantiles))}")
            logger.setLevel(old_level)

            # data should have dimension : batch, channels , height , width
            # height should be = width
            assert self.diffusion_model.image_size == data.shape[2], \
                "loaded data height must be equal to diffusion model image size property"
            assert self.diffusion_model.image_size == data.shape[3], \
                "loaded data width must be equal to diffusion model image size property"


def set_sinkhorn_baseline(dataset: Dataset, batch_size: int, device: torch.device, num_iter=100) -> Tuple[float, float]:
    dl = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    dist_list = []
    for _ in tqdm(range(num_iter), desc="sinkhorn baseline calculations"):
        x1 = next(iter(dl)).to(device)
        x2 = next(iter(dl)).to(device)

        assert len(x1.shape) == len(x2.shape), "all batches must be of the same shape "  # might be a redundant check
        for j in range(len(x1.shape)):
            assert x1.shape[j] == x2.shape[j], f"shape{j} does not match between two samples"
        # Assume batch sample is of shape B X C X H X W
        assert len(x1.shape) == 4, "assume batch sample of shape B X C X H X W with len(shape)=4"
        B = x1.shape[0]
        C = x1.shape[1]
        H = x1.shape[2]
        W = x1.shape[3]
        # Currently assume BW images, i.e. C = 1
        assert C == 1, "Assume dimension C = 1 i.e. BW image"
        x1_flat = x1.squeeze().reshape(B, H * W)
        x2_flat = x2.squeeze().reshape(B, H * W)
        assert len(x1_flat.shape) == 2, "first batch should have two dimensions after flattening"
        assert len(x2_flat.shape) == 2, "second batch should have two dimensions after flattening"
        norm_ = torch.norm(x1_flat - x2_flat).item()
        assert norm_ > 0, "Two batches must be different but the norm of diff is zero"
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=device)
        dist, P, C = sinkhorn(x1_flat, x2_flat)
        dist_list.append(dist.item())
    dist_avg = np.nanmean(dist_list)
    dist_std = np.std(dist_list)
    logger.info(f"Baseline sinkhorn average distance = {dist_avg}, and std = {dist_std}")
    return dist_avg, dist_std


if __name__ == '__main__':
    # Params and constants
    time_steps = 1000
    device = torch.device('cuda')
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 64
    num_train_step = 10_000
    mnist_number = 8
    debug_flag = False
    pbar_update_freq = 100
    checkpoint_freq = 1000
    assert num_train_step % checkpoint_freq == 0
    checkpoints_path = "../models/checkpoints/ddpm_mnist_8"
    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage
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

    # Dataset
    logger.info("Setting up MNIST dataset")
    mnist_data_path = f"../mnist_image_samples/{mnist_number}"
    mnist_dataset = Dataset2(folder=mnist_data_path, image_size=image_size)
    # set sinkhorn baseline
    sh_baseline_dist_avg, sh_baseline_dist_std = (
        set_sinkhorn_baseline(dataset=mnist_dataset, batch_size=batch_size, device=device))
    # Trainer
    opt = Adam(params=diffusion.parameters(), lr=1e-4)
    trainer = Trainer2(diffusion_model=diffusion, batch_size=batch_size, dataset=mnist_dataset, debug_flag=True,
                       optimizer=opt, device=device, train_num_steps=num_train_step,
                       progress_bar_update_freq=pbar_update_freq, checkpoint_freq=checkpoint_freq,
                       checkpoints_path=checkpoints_path)
    trainer.train()
    logger.info(f"Saving the diffusion model...")
    model_path = f"../models/diffusion_mnist_{mnist_number}_n_train_steps_{num_train_step}.pkl"
    torch.save(diffusion.state_dict(), model_path)
    logger.info(f"Successfully model saved to {model_path}")
    logger.info(f"Training script finished")

    #######################
    """
        Note
        =====
        In the original code , I have tried to use cuda and had the following error
        "RuntimeError: CUDA error: out of memory"?
        I have tried
            1. using gc collect and torch empty cache
            2. os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
        But none of them worked
        What worked is to reduce the image size passed for data and mode from 128 to 64
            ...
            device = torch.device('cuda')
            image_size=64
            diffusion = GaussianDiffusion( model,
                image_size=image_size,
                timesteps=time_steps  # number of steps
                ).to(device)
            training_images = torch.rand(num_images, num_channels, image_size, image_size).to(device)  
            # images are normalized from 0 to 1
        This is the snippet that works.
        """
