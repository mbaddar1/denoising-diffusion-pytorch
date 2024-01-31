"""
This script is to test the experiment with sinkhorn pytorch implementation
https://github.com/fwilliams/scalable-pytorch-sinkhorn
with some sklearn datasets, inspired by this example
https://github.com/dfdazac/wassdistance
"""
import torch
from datetime import datetime

# FIXME , GeomLoss sinkhorn causes some error related to CUDA lib loading failure
#   see this github issue for details https://github.com/mbaddar1/genmodel/issues/19
# from geomloss import SamplesLoss
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.ddpm_sandbox.ddpm_trainer import CustomDataset
# from sinkhorn import sinkhorn
from layers import SinkhornDistance
from sklearn import datasets
import logging
from PIL import Image
from torchvision import transforms as T
import pandas as pd

# logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def sinkhorn_wrapper(X1: torch.Tensor, X2: torch.Tensor):
    logger.info(f"Starting sinkhorn calculations")
    start_timestamp = datetime.now()
    sinkhorn = SinkhornDistance(eps=0.1, max_iter=100, device=device)
    dist, P, C = sinkhorn(X1, X2)
    # FIXME , GeomLoss sinkhorn causes some error related to CUDA lib loading failure
    #   see this github issue for details https://github.com/mbaddar1/genmodel/issues/19
    # with torch.no_grad():
    #     loss = SamplesLoss("gaussian", blur=0.5)
    #     dist = loss(X1, X2).item()
    end_timestamp = datetime.now()

    # example

    logger.info(f"Sinkhorn calculations finished in {(end_timestamp - start_timestamp).seconds} seconds")
    logger.info(f"Sinkhorn distance = {dist.item()}")
    return dist.item()


if __name__ == '__main__':
    n_samples = 10_000
    batch_size = 64
    img_size = 32  # mnist sample single images
    device = torch.device("cuda")
    transform = T.Compose([T.Resize(img_size),
                           T.CenterCrop(img_size),
                           T.ToTensor()])

    mnist_0_dataset = CustomDataset(
        dataset_name="../../denoising-diffusion-pytorch/denoising_diffusion_pytorch/mnist_image_samples/0",
        image_size=img_size)
    mnist_8_dataset = CustomDataset(
        dataset_name="../../denoising-diffusion-pytorch/denoising_diffusion_pytorch/mnist_image_samples/8",
        image_size=img_size)
    mnist_0_dl = DataLoader(dataset=mnist_0_dataset, batch_size=batch_size, shuffle=True)
    mnist_8_dl = DataLoader(dataset=mnist_8_dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Testing sinkhorn with sklearn circles dataset")

    data_pairs = {"sklearn_circles_same_params":
                      {"X1": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.5, noise=0.05)[0], device=device),
                       "X2": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.5, noise=0.05)[0], device=device)}
        ,
                  "sklearn_circles_different_factors":
                      {"X1": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.9, noise=0.05)[0], device=device),
                       "X2": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.5, noise=0.05)[0], device=device)}
        ,
                  "sklearn_circles_different_noises":
                      {"X1": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.5, noise=0.9)[0], device=device),
                       "X2": torch.tensor(data=datasets.make_circles(n_samples=n_samples,
                                                                     factor=0.5, noise=0.05)[0], device=device)}
        ,
                  "mnist_two_zeros":
                      {"X1": next(iter(mnist_0_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size),
                       "X2": next(iter(mnist_0_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)}
        ,
                  "mnist_two_eights":
                      {"X1": next(iter(mnist_8_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size),
                       "X2": next(iter(mnist_8_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)}
        ,
                  "mnist_zero_eight":
                      {"X1": next(iter(mnist_0_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size),
                       "X2": next(iter(mnist_8_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)}
                  }
    results_dict_list = []
    for key, val in data_pairs.items():
        logger.info(f"calculating sinkhorn distance with dataset pairs : {key}")
        sh_dist = sinkhorn_wrapper(X1=val["X1"], X2=val["X2"])
        results_dict_list.append({"data_pair_name": key, "sh_distance": sh_dist})
        logger.info("***")
    results_df = pd.DataFrame.from_records(data=results_dict_list)
    logger.info(f"SH distances df = \n {results_df}")

    # # Test using mnist data
    # logger.info(f"Sinkhorn test with MNIST dataset (for zero and eight dataset only, for now...) ")
    # mnist_0_dataset = Dataset2(
    #     folder="../../denoising-diffusion-pytorch/denoising_diffusion_pytorch/mnist_image_samples/0",
    #     image_size=img_size)
    # mnist_8_dataset = Dataset2(
    #     folder="../../denoising-diffusion-pytorch/denoising_diffusion_pytorch/mnist_image_samples/8",
    #     image_size=img_size)
    # mnist_0_dl = DataLoader(dataset=mnist_0_dataset, batch_size=batch_size, shuffle=True)
    # mnist_8_dl = DataLoader(dataset=mnist_8_dataset, batch_size=batch_size, shuffle=True)
#
#     X01 = next(iter(mnist_0_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)
#     X02 = next(iter(mnist_0_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)
#     norm_ = torch.norm(X01 - X02)
#
#     d00 = sinkhorn_wrapper(X1=X01, X2=X02)
#     print(d00)
#
#     X81 = next(iter(mnist_8_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)
#     X82 = next(iter(mnist_8_dl)).to(device).squeeze().reshape(batch_size, img_size * img_size)
#     d88 = sinkhorn_wrapper(X1=X81, X2=X82)
#     print(d88)
#
#     d08 = sinkhorn_wrapper(X1=X01, X2=X81)
#     print(d08)
# # # Experiment with mnist datasets
#
#
#
#
# X1 = transform(Image.open(img01)).to(device)
# X2 = transform(Image.open(img02)).to(device)
# sinkhorn_wrapper(X1, X2)
# logger.info(f"Finished test sinkhorn")
