"""
Load the MNIST from torchvision and apply edge detection with CV2 to create
(image, edges of image) data.
"""

import os
from pathlib import Path
from typing import Callable, Generator, Literal

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
from ark.optimization.base_module import BaseAnalogCkt, TimeInfo


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

    def gen_edge_detected_img(
        self,
        ideal_edge_detector: BaseAnalogCkt,
        time_info: TimeInfo,
        activation: Callable,
    ):
        """Simulate ideal edge detection with CNN on the images.

        Before use this function, the cnn_info should be set with the ideal edge detector.
        Afterward, should reset the info with the CNN for training.

        Args:
            ideal_edge_detector (BaseAnalogCkt): A CNN circuit with ideal components.
            time_info (TimeInfo): Simulation timing information.
            activation (Callable): activation function used in the CNN.
        """
        from tqdm import tqdm

        assert hasattr(self, "inp_nodes"), "Please set CNN info first."
        y = []
        print("Generating edge detected images...")
        for img in tqdm(self.images):
            for row_id, row in enumerate(img):
                for col_id, val in enumerate(row):
                    self.inp_nodes[row_id][col_id].set_init_val(val, n=0)
            x = jnp.array(self.cnn_ckt_cls.cdg_to_initial_states(self.graph)).flatten()
            y_raw = ideal_edge_detector(time_info, x, [], 0, 0)
            y_end_readout = activation(y_raw[-1, :]).reshape(self.image_shape())
            y.append(y_end_readout)
        self.load_edge_detected_data(jnp.array(y))

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


class RandomImgDataloader(DataLoader):

    def __init__(self, batch_size, image_shape=(3, 3), shuffle=True):
        base_images = np.array(
            [
                SimpleShapeDataloader.circle8x8,
                SimpleShapeDataloader.square8x8,
                SimpleShapeDataloader.rectangle8x8,
                SimpleShapeDataloader.diamond8x8,
                SimpleShapeDataloader.inverted_circle8x8,
                SimpleShapeDataloader.inverted_square8x8,
                SimpleShapeDataloader.inverted_rectangle8x8,
                SimpleShapeDataloader.inverted_diamond8x8,
            ]
        )
        images = np.array(
            [base_images[i % len(base_images)] for i in range(batch_size)]
        )

        # 1/3 stay the same
        # 1/3 apply one-sided white noise with 0.05 std
        #   If the original pixel is 0, add the noise
        #   If the original pixel is 1, subtract the noise
        # 1/3 is uniform noise
        for i in range(batch_size):
            choice = np.random.choice(3)
            if choice == 2:
                continue
            elif choice == 1:
                std = 0.05 * (2**choice)
                noise = np.abs(np.random.normal(0, std, image_shape))
                noise = np.where(images[i] == 1, -noise, noise)
                images[i] = np.clip(images[i] + noise, 0, 1)
            else:
                images[i] = np.random.randn(*image_shape)

        images = 2 * images - 1
        super().__init__(images, batch_size, shuffle)


class SilhouettesDataLoader(DataLoader):
    """Caltech 101 Silhouettes Dataset"""

    DATA_DIR = Path("data/silhouettes")

    def __init__(
        self,
        batch_size,
        dataset_type: Literal["train", "validation", "test"],
        img_size=16,
        shuffle=True,
    ):
        # Try if the dataset is downloaded at DATA_DIR
        # If not, download from source
        if not os.path.exists("data/silhouettes"):

            import requests
            from scipy.io import loadmat

            os.makedirs(self.DATA_DIR, exist_ok=True)
            url = f"https://people.cs.umass.edu/~marlin/data/caltech101_silhouettes_{img_size}_split1.mat"
            tmp_file = "caltech101_silhouettes.mat"
            with open(tmp_file, "wb") as file:
                response = requests.get(url)
                file.write(response.content)

            mat = loadmat(tmp_file)

            train, valid, test = (
                mat["train_data"].reshape(-1, img_size, img_size),
                mat["val_data"].reshape(-1, img_size, img_size),
                mat["test_data"].reshape(-1, img_size, img_size),
            )

            # Augment dataset with inverted images
            # and shift the dataset to [-1, 1] to fit CNN io
            train = 2 * np.concatenate([train, 1 - train]) - 1
            valid = 2 * np.concatenate([valid, 1 - valid]) - 1
            test = 2 * np.concatenate([test, 1 - test]) - 1

            np.savez(self.DATA_DIR / "train.npz", images=train)
            np.savez(self.DATA_DIR / "validation.npz", images=valid)
            np.savez(self.DATA_DIR / "test.npz", images=test)

        data: dict = np.load(self.DATA_DIR / f"{dataset_type}.npz")
        images = data["images"]
        self.dataset_type = dataset_type

        super().__init__(images, batch_size, shuffle)

    def gen_edge_detected_img(
        self,
        ideal_edge_detector: BaseAnalogCkt,
        time_info: TimeInfo,
        activation: Callable,
    ):
        # Try if the edge detected data is computed and stored at DATA_DIR
        # If not, compute and store
        edge_detected_img_path = (
            self.DATA_DIR / f"{self.dataset_type}_edge_detected.npz"
        )
        if not os.path.exists(edge_detected_img_path):
            super().gen_edge_detected_img(ideal_edge_detector, time_info, activation)
            np.savez(edge_detected_img_path, images=self.edge_images)
        else:
            data: dict = np.load(edge_detected_img_path)
            self.edge_images = data["images"]


if __name__ == "__main__":
    train_loader = SimpleShapeDataloader(32)
    test_loader = SimpleShapeDataloader(32)

    print(len(train_loader))
    print(len(test_loader))
    print(train_loader.image_shape())
    print(test_loader.image_shape())
