"""
Load the MNIST from torchvision and apply edge detection with CV2 to create
(image, edges of image) data.
"""

from typing import Generator

import cv2
import jax.numpy as jnp
import numpy as np
from jax import Array
from torchvision import datasets

from ark.cdg.cdg import CDG, CDGNode


class DataLoader:

    def __init__(
        self,
        dataset: datasets.MNIST,
        batch_size=32,
        shuffle=True,
        downsample: int = 1,
    ):
        images = dataset.data.numpy()

        # Downsample the images
        images = images[:, ::downsample, ::downsample]

        # Rescale the images to [-1, 1]
        edge_images = np.array([cv2.Canny(img, 100, 200) for img in images])
        images = images / 255 * 2 - 1
        edge_images = edge_images / 255 * 2 - 1

        self.images = images
        self.edge_images = edge_images
        self.batch_size = batch_size
        self.shuffle = shuffle

    def set_cnn_info(
        self,
        inp_nodes: list[list[CDGNode]],
        graph: CDG,
        cnn_ckt_cls,
    ):
        assert len(inp_nodes) == self.image_shape()[0]
        assert len(inp_nodes[0]) == self.image_shape()[1]
        self.inp_nodes = inp_nodes
        self.cnn_ckt_cls = cnn_ckt_cls
        self.graph = graph

    def load_edge_detected_data(self, edge_images):
        # Because cnn and cv2 edge detection behave differently
        # Need to load edge detected data separately
        self.edge_images = edge_images

    def __iter__(self) -> Generator[tuple[Array, Array, Array, Array], None, None]:
        images, edge_images = self.images, self.edge_images

        if self.shuffle:
            images, edge_images = self._shuffle(images, edge_images)
        batch_size = self.batch_size

        for i in range(0, len(images), batch_size):
            args_seed = jnp.array(np.random.randint(0, 2**32 - 1, size=batch_size))
            noise_seed = jnp.array(np.random.randint(0, 2**32 - 1, size=batch_size))

            imgs_i = jnp.array(images[i : i + batch_size])
            edge_imgs_i = jnp.array(edge_images[i : i + batch_size])

            # Pad the last batch
            if i + batch_size > len(images):
                imgs_i = jnp.pad(
                    imgs_i,
                    ((0, batch_size - imgs_i.shape[0]), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )
                edge_imgs_i = jnp.pad(
                    edge_imgs_i,
                    ((0, batch_size - edge_imgs_i.shape[0]), (0, 0), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            # Map the image to initial state of input nodes
            # (which will stay the same throughout the simulation)
            x = []
            for img in imgs_i:
                for row_id, row in enumerate(img):
                    for col_id, val in enumerate(row):
                        self.inp_nodes[row_id][col_id].set_init_val(val, n=0)
                x.append(self.cnn_ckt_cls.cdg_to_initial_states(self.graph))
            x = jnp.array(x).reshape(batch_size, -1)
            y = jnp.array(edge_imgs_i).reshape(batch_size, -1)
            yield x, args_seed, noise_seed, y

    def __len__(self):
        return len(self.images) // self.batch_size + 1

    def image_shape(self):
        return self.images[0].shape

    def _shuffle(self, img, edg):
        indices = np.random.permutation(len(self.images))
        return img[indices], edg[indices]


class TrainDataLoader(DataLoader):

    def __init__(self, batch_size, shuffle=True, downsample=1):
        dataset = datasets.MNIST(root="data", train=True, download=True)
        super().__init__(dataset, batch_size, shuffle, downsample)


class TestDataLoader(DataLoader):

    def __init__(self, batch_size, shuffle=True, downsample=1):
        dataset = datasets.MNIST(root="data", train=False, download=True)
        super().__init__(dataset, batch_size, shuffle, downsample)


if __name__ == "__main__":
    train_loader = TrainDataLoader(32)
    test_loader = TestDataLoader(32)

    print(len(train_loader))
    print(len(test_loader))
    print(train_loader.image_shape())
    print(test_loader.image_shape())

    next(iter(train_loader))

    for x, args_seed, noise_seed, y in train_loader:
        print(x.shape, y.shape)
        break
    for x, args_seed, noise_seed, y in test_loader:
        print(x.shape, y.shape)
        break
