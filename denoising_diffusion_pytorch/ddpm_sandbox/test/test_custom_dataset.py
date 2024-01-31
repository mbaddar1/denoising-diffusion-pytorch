"""

"""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.ddpm_sandbox.ddpm_trainer import CustomDataset

if __name__ == '__main__':
    img_size = 32
    dataset_name = "mnist0"
    data_dir = "../../mnist_image_samples/0"
    batch_size = 32
    ds = CustomDataset(dataset_name="moons", num_samples=100)
    dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)

    for i in range(3):
        b = next(iter(dl))
        x = b.reshape(b.shape[0] * b.shape[1], b.shape[2]).detach().numpy()
        plt.clf()
        _, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1])
        plt.savefig(f"./data_plot_{i + 1}.png")
