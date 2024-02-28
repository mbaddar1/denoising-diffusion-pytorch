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
2. Add a quick piece of documentation in the PhD thesis with link to the GitHub issue. Write properly later
3. Save the log file as an attachment to the ticket
"""
import json
import os.path
from typing import Tuple

import matplotlib.pyplot as plt
from sklearn.datasets import make_circles, make_swiss_roll, make_blobs, make_moons
import numpy as np
from PIL import Image
from denoising_diffusion_pytorch import GaussianDiffusion, Unet2D
import torch
import logging
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms as T
from torch.optim import Optimizer, Adam
from tqdm import tqdm
from datetime import datetime
import torch
import shutil
from denoising_diffusion_pytorch.denoising_diffusion_pytorch import HeadTail, SimpleNN1, SimpleResNet
from stat_dist.layers import SinkhornDistance

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


# Global Vars


# similar to denoising_diffusion_pytorch.denoising_diffusion_pytorch.Dataset
class ImageDataset(Dataset):
    EXTENSIONS = ['jpg', 'png']

    # can be considered as a batch. Why 128 ? as it is the first larger int after the default of 100

    def __init__(self, image_size: int, data_dir: str, debug_flag: bool = False):
        """
        data_source: str : can be path to data or the name of some ready-made dataset, like sklearn datasets
        debug_flag : bool

        """
        super().__init__()
        self.debug_flag = debug_flag
        self.image_size = image_size
        self.data_dir = data_dir
        assert self.image_size is not None, f"image size cannot be null with mnist dataset"
        assert self.data_dir is not None, f"data dir cannot be None"
        self.paths = [p for ext in ImageDataset.EXTENSIONS for p in Path(f'{self.data_dir}').glob(f'**/*.{ext}')]
        # Transformation based on https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/
        #   denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L838
        self.transform = T.Compose([
            T.Resize(self.image_size),
            T.CenterCrop(self.image_size),
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


class DDPmTrainer:
    def __init__(self,
                 diffusion_model: GaussianDiffusion,
                 batch_size: int,
                 num_train_iterations: int,
                 device: torch.device,
                 optimizer: Optimizer,
                 progress_bar_update_freq: int,
                 checkpoint_freq: int,
                 debug_flag: bool,
                 checkpoints_path: str, **kwargs):
        """

        Notes:
            - The DiffusionModel object is attached to a dataset_name ,
            hence no need to pass dataset_name to the trainer class
            -
        """
        self.diffusion_model = diffusion_model
        self.debug_flag = debug_flag
        self.device = device
        self.num_train_iterations = num_train_iterations
        self.optimizer = optimizer
        self.step = 0
        self.checkpoint_freq = checkpoint_freq
        self.progress_bar_update_freq = progress_bar_update_freq
        self.checkpoints_path = checkpoints_path
        self.batch_size = batch_size
        if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
            assert "image_dataset" in kwargs.keys(), \
                ("If DiffusionModel is attached to image dataset, the train must have an image_dataset passed as "
                 "a parameter")
            self.image_dataset = kwargs.get("image_dataset")
            self.train_dataloader = DataLoader(dataset=self.image_dataset, batch_size=batch_size, shuffle=True)
            self.test_dataloader = DataLoader(dataset=self.image_dataset, batch_size=batch_size, shuffle=True)
        elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
            pass  # FIXME, this pass line is just for clarity but can be better
        else:
            raise ValueError(f"Diffusion model has invalid dataset_name : {self.diffusion_model.dataset_name}")

        #
        logger.info(f"Creating data loader for the dataset")
        # FIXME separate train and test datasets

        if self.debug_flag:
            self.__validate_dataset()  # FIXME remove later

    def sinkhorn_eval(self, generated_sample: torch.Tensor, num_iter: int) -> Tuple[float, float]:

        if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
            assert len(generated_sample.shape) == 4, f"For datasets : {GaussianDiffusion.MNIST_DATASET_NAMES}"
            distances_list = []
            for _ in range(num_iter):
                ref_sample = self.get_next_data_batch()
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
                distances_list.append(dist.item())
            return np.nanmean(distances_list), np.std(distances_list)
        elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
            assert len(generated_sample.shape) == 2, f"For datasets : {GaussianDiffusion.SKLEARN_DATASET_NAMES}"
            distances_list = []
            for _ in range(num_iter):
                ref_sample = self.get_next_data_batch()
                assert len(generated_sample.shape) == len(ref_sample.shape)
                for j in range(len(generated_sample.shape)):
                    assert generated_sample.shape[j] == ref_sample.shape[j]
                # shape assume to be B X D
                sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=self.device)
                distance, P, C = sinkhorn(generated_sample, ref_sample)
                distances_list.append(distance.item())
            return np.nanmean(distances_list), np.std(distances_list)

    def get_next_data_batch(self):
        data_batch = None
        if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
            data_batch = next(iter(self.train_dataloader)).to(self.device)
        elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
            if self.diffusion_model.dataset_name == "circles":
                data_np, _ = make_circles(n_samples=self.batch_size, shuffle=True, noise=0.05, factor=0.3)
            else:
                raise ValueError(f"Unsupported dataset_name {self.diffusion_model.dataset_name}")
            data_batch = torch.tensor(data=data_np, dtype=torch.float64).to(self.device)
        else:
            raise ValueError(f"dataset_name : {dataset_name} not supported")

        return data_batch

    def train(self):
        # Use manual control for tqdm and progress bar update
        # See https://github.com/tqdm/tqdm#usage
        # and https://stackoverflow.com/a/45808255
        self.step = 0
        # Exponential smoothing for loss reporting
        # Called Exponential Smoothing or Exponential Moving Average
        ema_loss = 0.0  # Just for initialization
        ema_loss_alpha = 0.9
        start_timestamp = datetime.now()
        with tqdm(initial=self.step, total=self.num_train_iterations) as progress_bar:
            while self.step <= self.num_train_iterations:  # note the boundary condition
                self.optimizer.zero_grad()
                data = self.get_next_data_batch()
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
                    # FIXME, there is redundancy in the metadata.
                    #  Make a main meta data file and another per-iter metadata one
                    checkpoint_metadata = json.dumps({"ema_loss": ema_loss,
                                                      "sinkhorn_dist_avg": sh_dist_avg,
                                                      "sinkhorn_dist_std": sh_dist_std,
                                                      "train_num_iters": self.num_train_iterations,
                                                      "batch_size": self.batch_size,
                                                      "data_shape": data.shape,
                                                      "train_step": self.step})
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

    def forward_process_viz(self):
        logger.info(f"Forward process Viz")
        # Use this article as ref
        # https://papers-100-lines.medium.com/diffusion-models-from-scratch-tutorial-in-100-lines-of-pytorch-code-5dac9f472f1c
        start_sample_filename = f"{self.diffusion_model.dataset_name}_start.png"  # t0
        midway_sample_filename = f"{self.diffusion_model.dataset_name}_middle.png"  # t_middle
        end_sample_filename = f"{self.diffusion_model.dataset_name}_end.png"  # t_N
        t_midway_int = int(self.diffusion_model.num_timesteps // 20)
        IMG_MAX_PIXEL_VALUE = 255.0
        # Midway point: close to start to see the original with some noise not actual midway
        with torch.no_grad():
            if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
                fw_process_dl = DataLoader(dataset=self.image_dataset, batch_size=1, shuffle=True)
                x_start = next(iter(fw_process_dl)).to(self.device)
                x_start_norm = self.diffusion_model.normalize(x_start)
                t_midway_tensor = torch.tensor(t_midway_int, device=self.device).reshape(1, )
                t_end_tensor = torch.tensor(self.diffusion_model.num_timesteps - 1).reshape(1, ).to(self.device)

                # Forward process sampling
                x_halfway = self.diffusion_model.q_sample(x_start=x_start_norm, t=t_midway_tensor)

                # Clamping and Re-Scaling
                x_end = self.diffusion_model.q_sample(x_start=x_start_norm, t=t_end_tensor)
                x_start_norm = (x_start_norm.clamp(0, 1) * IMG_MAX_PIXEL_VALUE).squeeze()
                x_halfway = (x_halfway.clamp(0, 1) * IMG_MAX_PIXEL_VALUE).type(torch.float).squeeze()
                x_end = (x_end.clamp(0, 1) * IMG_MAX_PIXEL_VALUE).type(torch.float).squeeze()
                # Create image objects

                img_start = Image.fromarray(x_start_norm.detach().cpu().numpy()).convert("L")
                img_halfway = Image.fromarray(x_halfway.detach().cpu().numpy()).convert("L")
                img_end = Image.fromarray(x_end.detach().cpu().numpy()).convert("L")

                # Save images
                img_start.save(start_sample_filename)
                img_halfway.save(midway_sample_filename)
                img_end.save(end_sample_filename)

            elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
                x_start = self.get_next_data_batch()
                t_halfway = torch.tensor(
                    np.repeat(t_midway_int, self.batch_size).reshape(-1, )).to(
                    self.device)
                t_end_tensor = torch.tensor(
                    np.repeat(self.diffusion_model.num_timesteps - 1, self.batch_size).reshape(-1, )).to(self.device)
                x_halfway = self.diffusion_model.q_sample(x_start=x_start, t=t_halfway)
                x_end = self.diffusion_model.q_sample(x_start=x_start, t=t_end_tensor)
                plt.scatter(x_start.cpu().detach().numpy()[:, 0], x_start.cpu().detach().numpy()[:, 1])
                plt.savefig(start_sample_filename)

                plt.clf()
                plt.scatter(x_halfway.cpu().detach().numpy()[:, 0], x_halfway.cpu().detach().numpy()[:, 1])
                plt.savefig(midway_sample_filename)

                plt.clf()
                plt.scatter(x_end.cpu().detach().numpy()[:, 0], x_end.cpu().detach().numpy()[:, 1])
                plt.savefig(end_sample_filename)
            else:
                raise ValueError(f"Unknown dataset_name : {self.diffusion_model.dataset_name}")
        logger.info(f"Forward process Viz finished")

    def set_sinkhorn_baseline(self, num_iter=10, **kwargs) -> Tuple[float, float]:
        distances_list = []
        sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=device)
        for _ in tqdm(range(num_iter), desc="sinkhorn baseline calculations"):
            x1 = self.get_next_data_batch()
            x2 = self.get_next_data_batch()
            assert len(x1.shape) == len(
                x2.shape), "all batches must be of the same shape "  # might be a redundant check
            for j in range(len(x1.shape)):
                assert x1.shape[j] == x2.shape[j], f"shape{j} does not match between two samples"

            if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
                assert len(x1.shape) == 4, "For image datasets  batch sample of shape B X C X H X W with len(shape)=4"
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
                sinkhorn_distance, P, C = sinkhorn(x1_flat, x2_flat)
                distances_list.append(sinkhorn_distance.item())
            elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
                """
                for sklearn datasets the batches are of size B X D where:
                    B is the batch size
                    D is the dimension of the dataset
                We will use the batching feature in the sinkhorn implementation here (with modification from original)
                denoising-diffusion-pytorch/stat_dist/layers.py:51
                and here is the github ref (original impl.) 
                https://github.com/dfdazac/wassdistance/blob/master/layers.py#L37
                """
                assert len(x1.shape) == 2, "For sklearn dataset , shape must be of len 2 : B X D"
                sinkhorn_distance, P, C = sinkhorn(x1, x2)  # apply batched sinkhorn calculations
                # FIXME, this assertion was needed when the sklearn dataset was batch, now it is not
                # TODO remove the commented code later
                # assert len(sinkhorn_distance.shape) == 1 and sinkhorn_distance.shape[0] == x1.shape[0], \
                #     (f"Batched distances must be of length equal to data-batch batch-size : "
                #      f"{sinkhorn_distance.shape[0]} != {x1.shape[0]}")
                distances_list.append(sinkhorn_distance.item())
            else:
                raise ValueError(f"dataset_name : {self.diffusion_model.dataset_name} is not supported")

        distances_avg = np.nanmean(distances_list)
        distances_std = np.std(distances_list)
        logger.info(f"Baseline sinkhorn average distance = {distances_avg}, and std = {distances_std}")
        return distances_avg, distances_std

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
                f"For quantiles at levels {quantiles} = "
                f"{torch.quantile(input=data, q=torch.tensor(quantiles, dtype=data.dtype))}")
            logger.setLevel(old_level)
            # data should have dimension: batch, channels, height, width
            # height should be = width
            dataset_name_attr = "dataset_name"

            if not hasattr(image_dataset, dataset_name_attr):
                raise ValueError(f"Dataset class instance must have property {dataset_name_attr}")
            if self.diffusion_model.dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
                assert self.diffusion_model.image_size == data.shape[2], \
                    "loaded data height must be equal to diffusion model image size property"
                assert self.diffusion_model.image_size == data.shape[3], \
                    "loaded data width must be equal to diffusion model image size property"

            elif self.diffusion_model.dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
                assert self.diffusion_model.image_size is None, (" For sklearn datasets , image_size property in "
                                                                 "diffusion model is useless and must be None")


if __name__ == '__main__':
    # Params
    # This code part is for mnist dataset name
    mnist_num = "8"  # Can be None if we are going to use all mnist numbers
    dataset_name = f"mnist{mnist_num}"
    dataset_dir = f"../mnist_image_samples/{mnist_num}"  # Needed only for actual, not synthetic datasets

    # This code part is for sklearn dataset
    # dataset_name = "circles"
    # Diffusion Model parameters
    diffusion_model_name = "ddpm_nn"
    diffusion_model_objective = "pred_noise"
    noise_model_name = "unet2d"
    # noise_model_name = "head_tail"  # noise_model_name must be consistent with dataset_name
    # dataset_name variables if dataset is mnist

    checkpoints_dir = f"../models/checkpoints"
    device = torch.device('cuda')

    # constants
    time_steps = 50  # 1000 for mnist datasets and 40 or 50 for Sklearn datasets
    image_size = 32
    num_images = 1
    num_channels = 1
    batch_size = 32
    num_train_iterations = 5_000
    debug_flag = False
    pbar_update_freq = 100
    checkpoint_freq = 100
    unet_dim = 64
    hidden_dim = 256

    # Some assertion for params
    assert num_train_iterations % checkpoint_freq == 0
    checkpoints_path = os.path.join(checkpoints_dir, f"{diffusion_model_name}_{dataset_name}")
    assert dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES + GaussianDiffusion.SKLEARN_DATASET_NAMES
    # models assertions
    assert diffusion_model_name in ["ddpm_nn"]
    if dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
        assert noise_model_name in ["unet2d"]
    elif dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
        assert noise_model_name in ["head_tail", "simple_nn1", "simple_resnet"]
    else:
        raise ValueError(f"Unknown dataset_name : {dataset_name}")

    # Delete an old checkpoint path, with contents, then create a new fresh one
    logger.info(f"Removing checkpoint dir : {checkpoints_path} if exists")
    shutil.rmtree(checkpoints_path, ignore_errors=True)
    logger.info(f"Creating fresh checkpoint dir : {checkpoints_path}")
    os.makedirs(checkpoints_path)

    # Test if cuda is available
    logger.info(f"Cuda checks")
    logger.info(f'Is cuda available ? : {torch.cuda.is_available()}')
    logger.info(f'Cuda device count = {torch.cuda.device_count()}')

    diffusion_model = None
    noise_model = None
    trainer = None
    if dataset_name in GaussianDiffusion.MNIST_DATASET_NAMES:
        noise_model = Unet2D(dim=unet_dim,
                             channels=num_channels,
                             dim_mults=(1, 2, 4, 8),
                             flash_attn=True)
        diffusion_model = GaussianDiffusion(
            noise_model=noise_model,
            dataset_name=dataset_name,
            image_size=image_size,
            timesteps=time_steps,  # number of steps
            auto_normalize=False,
            objective=diffusion_model_objective
        ).to(device)
        opt = Adam(params=diffusion_model.parameters(), lr=1e-4)
        image_dataset = ImageDataset(image_size=image_size, data_dir=dataset_dir)
        trainer = DDPmTrainer(diffusion_model=diffusion_model, image_dataset=image_dataset, batch_size=batch_size,
                              num_train_iterations=num_train_iterations, device=device, optimizer=opt,
                              progress_bar_update_freq=pbar_update_freq, checkpoint_freq=checkpoint_freq,
                              checkpoints_path=checkpoints_path, debug_flag=debug_flag)
        trainer.forward_process_viz()
        sh_baseline_dist_avg, sh_baseline_dist_std = (
            trainer.set_sinkhorn_baseline(dataset_name=dataset_name, batch_size=batch_size, device=device))

    elif dataset_name in GaussianDiffusion.SKLEARN_DATASET_NAMES:
        if noise_model_name == "head_tail":
            noise_model = HeadTail(hidden_dim=hidden_dim, input_dim=2, time_steps=time_steps).to(device)
        elif noise_model_name == "simple_nn1":
            noise_model = SimpleNN1(hidden_dim=hidden_dim, input_out_dim=2, diffusion_time_steps=time_steps)
        elif noise_model_name == "simple_resnet":
            noise_model = SimpleResNet(hidden_dim=hidden_dim, input_out_dim=2, diffusion_time_steps=time_steps)
        else:
            raise ValueError(f"Unsupported noise_model : {noise_model_name}")
        diffusion_model = GaussianDiffusion(noise_model=noise_model, dataset_name=dataset_name,
                                            timesteps=time_steps, objective=diffusion_model_objective).to(device)

        opt = Adam(params=diffusion_model.parameters(), lr=1e-4)
        trainer = DDPmTrainer(diffusion_model=diffusion_model, batch_size=batch_size,
                              num_train_iterations=num_train_iterations, device=device, optimizer=opt,
                              progress_bar_update_freq=pbar_update_freq,
                              checkpoint_freq=checkpoint_freq, checkpoints_path=checkpoints_path,
                              debug_flag=debug_flag)
        trainer.forward_process_viz()
        sh_baseline_dist_avg, sh_baseline_dist_std = (
            trainer.set_sinkhorn_baseline(dataset_name=dataset_name, batch_size=batch_size, device=device))
    else:
        raise ValueError(f"Unknown dataset_name : {dataset_name}")
    # Check if models are on cuda actually
    # https://github.com/mbaddar1/denoising-diffusion-pytorch?tab=readme-ov-file#usage
    # DDPM step model: can any regression model
    # Double-check if models are actually on cuda
    #   https://discuss.pytorch.org/t/how-to-check-if-model-is-on-cuda/180/2
    is_unet_model_on_cuda = next(noise_model.parameters()).is_cuda
    logger.info(f'If core model is on cuda ? : {is_unet_model_on_cuda}')
    is_diffusion_model_on_cuda = next(diffusion_model.parameters()).is_cuda
    logger.info(f'Is diffusion model on cuda? : {is_diffusion_model_on_cuda}')
    trainer.train()
    logger.info(f"Model training finished and the final model is the latest checkpoint ")
    logger.info(f"Saving the diffusion model...")

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
