"""

"""
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.ddpm_sandbox.ddpm_trainer import CustomDataset

if __name__ == '__main__':
    img_size = 32
    dataset_name = "mnist0"
    batch_size = 32
    mnist_data_set_names = {"mnist0": "../../mnist_image_samples/0", "mnist8": "../../mnist_image_samples/8"}
    for dataset_name, dataset_dir in mnist_data_set_names.items():
        ds = CustomDataset(dataset_name=dataset_name, image_size=img_size, data_dir=dataset_dir)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
        b = next(iter(dl))
        # FIXME , add some kind for validation to the generated batches

    sklearn_dataset_names = ["circles", "swissroll2d", "blobs", "moons"]
    for dataset_name in sklearn_dataset_names:
        ds = CustomDataset(dataset_name=dataset_name, num_samples=100)
        dl = DataLoader(dataset=ds, batch_size=batch_size, shuffle=True)
        b = next(iter(dl))
        x = b.reshape(b.shape[0] * b.shape[1], b.shape[2]).detach().numpy()
        plt.clf()
        _, ax = plt.subplots()
        ax.scatter(x[:, 0], x[:, 1])
        plt.savefig(f"./{dataset_name}.png")
