"""

"""
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.ddpm_sandbox.ddpm_trainer import CustomDataset

if __name__ == '__main__':
    img_size = 32
    dataset_name = "mnist0"
    data_dir = "../../mnist_image_samples/0"  #
    batch_size = 64
    ds = CustomDataset(dataset_name="circles", num_samples=100)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
    for i in range(1000):
        b = next(iter(dl))
        print("...")
