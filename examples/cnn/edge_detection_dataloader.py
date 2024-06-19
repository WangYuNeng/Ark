"""
Load the MNIST from torchvision and apply edge detection with CV2 to create
(image, edges of image) data.
"""

from typing import Generator

import jax.numpy as jnp
import numpy as np
from jax import Array


try:
    from torchvision import datasets
except ImportError:
    print(
        "Please install torchvision to run this example to download the MNIST dataset."
    )
    raise ImportError

from ark.cdg.cdg import CDG, CDGNode


class DataLoader:

    def __init__(
        self,
        images: np.ndarray,
        batch_size=32,
        shuffle=True,
    ):

        self.images = images
        # Because cv2 edge detection behaves differently
        # Need to load edge detected data separately
        self.edge_images = np.zeros_like(images)
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
        if len(self.images) % self.batch_size == 0:
            return len(self.images) // self.batch_size
        return len(self.images) // self.batch_size + 1

    def image_shape(self):
        return self.images[0].shape

    def _shuffle(self, img, edg):
        indices = np.random.permutation(len(self.images))
        return img[indices], edg[indices]


class MNISTDataLoader(DataLoader):

    def __init__(
        self, dataset: datasets.MNIST, batch_size: int, shuffle=True, downsample=1
    ):
        images = dataset.data.numpy()

        # Downsample the images
        images = images[:, ::downsample, ::downsample]
        super().__init__(images, batch_size, shuffle)


class MNISTTrainDataLoader(MNISTDataLoader):

    def __init__(self, batch_size, shuffle=True, downsample=1):
        dataset = datasets.MNIST(root="data", train=True, download=True)
        super().__init__(dataset, batch_size, shuffle, downsample)


class MNISTTestDataLoader(MNISTDataLoader):

    def __init__(self, batch_size, shuffle=True, downsample=1):
        dataset = datasets.MNIST(root="data", train=False, download=True)
        super().__init__(dataset, batch_size, shuffle, downsample)


class SimpleShapeDataloader(DataLoader):

    circle8x8 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    circle8x8_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    inverted_circle8x8 = 1 - circle8x8
    inverted_circle8x8_edge = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    square8x8 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    square8x8_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    inverted_square8x8 = 1 - square8x8
    inverted_square8x8_edge = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    rectangle8x8 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    rectangle8x8_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    inverted_rectangle8x8 = 1 - rectangle8x8
    inverted_rectangle8x8_edge = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    diamond8x8 = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    diamond8x8_edge = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )
    inverted_diamond8x8 = 1 - diamond8x8
    inverted_diamond8x8_edge = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )

    def __init__(self, batch_size, shuffle=True):
        base_images = np.array(
            [
                self.circle8x8,
                self.square8x8,
                self.rectangle8x8,
                self.diamond8x8,
                self.inverted_circle8x8,
                self.inverted_square8x8,
                self.inverted_rectangle8x8,
                self.inverted_diamond8x8,
            ]
        )
        base_images = 2 * base_images - 1
        base_edge_images = np.array(
            [
                self.circle8x8_edge,
                self.square8x8_edge,
                self.rectangle8x8_edge,
                self.diamond8x8_edge,
                self.inverted_circle8x8_edge,
                self.inverted_square8x8_edge,
                self.inverted_rectangle8x8_edge,
                self.inverted_diamond8x8_edge,
            ]
        )
        base_edge_images = 2 * base_edge_images - 1

        n_images = len(base_images)
        # Sample batch_size number of images from the base images
        images = np.array([base_images[i % n_images] for i in range(batch_size)])
        edge_images = np.array(
            [base_edge_images[i % n_images] for i in range(batch_size)]
        )
        super().__init__(images, batch_size, shuffle)
        self.load_edge_detected_data(edge_images)


if __name__ == "__main__":
    train_loader = SimpleShapeDataloader(32)
    test_loader = SimpleShapeDataloader(32)

    print(len(train_loader))
    print(len(test_loader))
    print(train_loader.image_shape())
    print(test_loader.image_shape())
