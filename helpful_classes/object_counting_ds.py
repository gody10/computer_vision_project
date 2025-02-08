import torch
import numpy as np
from cv_ds_base import CV_DS_Base

class ObjectCounting_DS(CV_DS_Base):
    """
    Self contained PyTorch Dataset for testing object counting models.

    Args:
        ds_size (int): number of images in dataset.
        img_size (int): will build images of shape (3, img_size, img_size).
        shapes_per_image (Tuple[int, int]): will produce images containing
            minimum number of shapes Tuple[0] and maximum number of shapes
            Tuple[1]. For example shapes_per_image = (2,2) would create a
            dataset where each image contains exactly two shapes.
        class_probs (Tuple[float, float, float]): relative probability of
            each shape occuring in an image. Need not sum to 1. For example
            class_probs = (1,1,0) will create a dataset with 50% class 1
            shapes, 50% class 2 shapes, 0% class 3 shapes.
        rand_seed (int): used to instantiate a numpy random number generator.
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
            build_imgs_and_targets() method.
    """

    def __init__(
        self,
        ds_size=100,
        img_size=256,
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
        object_count=False,
    ):

        super().__init__(
            ds_size, img_size, shapes_per_image, class_probs, rand_seed, class_map
        )

        self.object_count = object_count
        self.num_classes = np.max(np.nonzero(self.class_probs)) + 1

        self.imgs, self.targets = self.build_imgs_and_targets()

    def build_imgs_and_targets(self):
        """
        Builds images and targets for object counting.

        Returns:
            imgs (torch.UInt8Tensor[ds_size, 3, img_size, img_size]): images
                containing different shapes. The images are gray-scale
                (each layer of the first (color) dimension is identical).
                This makes it easier to visualize targets and predictions.
            targets (torch.int64[ds_size, num_classes]): targets contian
                either the number of instances of each class (if
                object_count = True) or a binary value representing if
                any of the class are present in the image. For example
                if image i contains 3 instances of class 2 then
                targets[i][1] = 3 if object_count = True and
                targets[i][1] = 1 if object_count = False.
        """
        imgs = []
        targets = []

        for idx in range(self.ds_size):

            chosen_ids = self.chosen_ids_per_img[idx]

            # Creating an empty noisy img.
            img = self.rng.integers(
                self.class_map[0]["gs_range"][0],
                self.class_map[0]["gs_range"][1],
                (self.img_size, self.img_size),
            )

            target = np.zeros(self.num_classes)

            # Filling the noisy img with shapes and building targets.
            for i, class_id in enumerate(chosen_ids):
                shape = self.draw_shape(class_id)
                gs_range = self.class_map[class_id]["gs_range"]
                img[shape] = self.rng.integers(
                    gs_range[0], gs_range[1], img[shape].shape
                )
                target[class_id - 1] += 1

            # Convert from np to torch and assign appropriate dtypes.
            img = torch.from_numpy(img)
            img = img.unsqueeze(dim=0).repeat(3, 1, 1).type(torch.ByteTensor)

            target = torch.from_numpy(target).long()
            if not self.object_count:
                target = torch.clamp(target, max=1)
            imgs.append(img)
            targets.append(target)

        # Turn a list of imgs with shape (3, H, W) of len ds_size to a tensor
        # of shape (ds_size, 3, H, W)
        imgs = torch.stack(imgs)
        targets = torch.stack(targets)

        return imgs, targets