import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader
from object_counting_ds import ObjectCounting_DS

class ObjectCounting_DM(pl.LightningDataModule):
    """
    Self contained PyTorch Lightning DataModule for testing object 
    counting models with PyTorch Lightning.Uses the torch dataset 
    ObjectCounting_DS.

    Args:
        train_val_size (int): total size of the training and validation
            sets combined.
        train_val_split (Tuple[float, float]): should sum to 1.0. For example
            if train_val_size = 100 and train_val_split = (0.80, 0.20)
            then the training set will contain 80 imgs and the validation
            set will contain 20 imgs.
        test_size (int): the size of the test data set.
        batch_size (int): batch size to be input to dataloaders. Applies
            for training, val, and test datasets.
        dataloader_shuffle (Dict): whether or not to shuffle for each of
            the three dataloaders. Dict must contain the keys: "train",
            "val", "test".
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy rng.
        class_map (Dict[Dict]): the class map must contain keys (0,1,2,3)
            and contain names "background", "rectangle", "line", and "donut".
            "gs_range" specifies the upper and lower bound of the
            grayscale values (0, 255) used to color the shapes.
            "target_color" can be used by visualization tools to assign
            a color to masks and boxes. Note that class 0 is reserved for
            background in most instance seg models, so one can rearrange
            the class assignments of different shapes but 0 must correspond
            to "background". The utility of this Dict is to enable the user
            to change target colors, class assignments, and shape
            intensities. A valid example:
            class_map={
            0: {"name": "background","gs_range": (200, 255),"target_color": (255, 255, 255),},
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)}}.
        object_count (bool): whether or not the targets contain the
            object instance counts or not. Example below under the
            build_imgs_and_targets() method of the ImageClassification_DS .

    """

    def __init__(
        self,
        train_val_size=100,
        train_val_split=(0.80, 0.20),
        test_size=10,
        batch_size=8,
        dataloader_shuffle={"train": True, "val": False, "test": False},
        img_size=50,
        shapes_per_image=(1, 3),
        class_probs=(1, 1, 1),
        rand_seed=12345,
        class_map={
            0: {
                "name": "background",
                "gs_range": (200, 255),
                "target_color": (255, 255, 255),
            },
            1: {"name": "rectangle", "gs_range": (0, 100), "target_color": (255, 0, 0)},
            2: {"name": "line", "gs_range": (0, 100), "target_color": (0, 255, 0)},
            3: {"name": "donut", "gs_range": (0, 100), "target_color": (0, 0, 255)},
        },
        object_count=True,
    ):

        super().__init__()

        if sorted(list(dataloader_shuffle.keys())) != sorted(["train", "val", "test"]):
            raise ValueError(
                "Dict dataloader_shuffle must contain the keys: train, val, test."
            )
        # Attributes to define datamodule.
        self.train_val_size = train_val_size
        self.train_val_split = np.array(train_val_split)
        self.train_val_sizes = np.array(
            self.train_val_size * self.train_val_split, dtype=int
        )
        self.test_size = test_size
        self.batch_size = batch_size
        self.dataloader_shuffle = dataloader_shuffle

        # Attributes to define dataset.
        self.img_size = img_size
        self.rand_seed = rand_seed
        self.shapes_per_image = shapes_per_image
        self.class_probs = class_probs
        self.class_map = class_map
        self.object_count = object_count

    def setup(self, stage):
        if stage == "fit" or stage is None:
            print("Setting up fit stage.")

            self.train = ObjectCounting_DS(
                ds_size=self.train_val_sizes[0],
                img_size=self.img_size,
                rand_seed=self.rand_seed,
                shapes_per_image=self.shapes_per_image,
                class_probs=self.class_probs,
                class_map=self.class_map,
                object_count=self.object_count,
            )
            self.val = ObjectCounting_DS(
                ds_size=self.train_val_sizes[1],
                img_size=self.img_size,
                rand_seed=self.rand_seed + 111,
                shapes_per_image=self.shapes_per_image,
                class_probs=self.class_probs,
                class_map=self.class_map,
                object_count=self.object_count,
            )

        if stage == "test" or stage is None:
            print("Setting up test stage.")

            self.test = ObjectCounting_DS(
                ds_size=self.test_size,
                img_size=self.img_size,
                rand_seed=self.rand_seed + 222,
                shapes_per_image=self.shapes_per_image,
                class_probs=self.class_probs,
                class_map=self.class_map,
                object_count=self.object_count,
            )

        return None

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.dataloader_shuffle["train"],
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=self.dataloader_shuffle["val"],
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=self.dataloader_shuffle["test"],
        )
