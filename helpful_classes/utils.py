import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
import cv2
import torch
import ipywidgets as widgets
from object_counting_dm import ObjectCounting_DM
from torchvision.utils import make_grid

def show(imgs, figsize=(10.0, 10.0)):
    """Displays a single image or list of images. Taken more or less from
    the pytorch docs:
    https://pytorch.org/vision/main/auto_examples/plot_visualization_utils.html#visualizing-a-grid-of-images

    Args:
        imgs (Union[List[torch.Tensor], torch.Tensor]): A list of images
            of shape (3, H, W) or a single image of shape (3, H, W).
        figsize (Tuple[float, float]): size of figure to display.

    Returns:
        None
    """

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), figsize=figsize, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = TF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.show()

    return None

def add_labels(img, label, class_map, pred=False, object_count=False):
    """Takes a single image of shape (3, H, W) and adds labels directly
    onto the image using cv2. Used with ImageSegmentation_DS/DM but can 
    be used in other applicable computer vision tasks. 

    Args:
        img (torch.UInt8Tensor[3, H, W]): a pytorch image.
        label (torch.int64[ds_size, num_classes]): label contians
                either the number of instances of each class (if object_count
                = True) or a binary value representing if 
                any of the class are present in the image. For example
                if the image contains 3 instances of class 2 then 
                label[1] = 3 if object_count = True and 
                label[1] = 1 if object_count = False. Note that here 0 is
                not a valid class so if your class_map contains keys
                0,1,2,3,4 then num_classes = 4. 
        class_map (Dict[Dict]): the class map must contain keys that
            correspond to the labels provided. Inner Dict must contain
            "name" and "target_color". class 0 is reserved for the case
            where the image contains no objects (label.sum() == 0).
            A valid example:
            class_map={
            0: {"name": "background","target_color": (255, 255, 255),},
            1: {"name": "rectangle", "target_color": (255, 0, 0)},
            2: {"name": "line", "target_color": (0, 255, 0)},
            3: {"name": "donut", "target_color": (0, 0, 255)}}.
        pred (bool): whether or not the label provided is a prediction. 
            Predictions are printed in the bottom right of the image 
            whereas targets are printed in the top left. 
        object_count (bool): whether or not the label contains the 
            object instance counts or not. See above under label for an 
            example. 


    Returns:
        img (torch.UInt8Tensor[3, H, W]): a pytorch image with the names 
            (and possibly counts) corresponding to the provided label
            drawn over the image. 
    """
    img_size = img.shape[-1]
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8).copy()

    if label.sum() == 0:

        nonzero_classes = [0]
        label_colors = [class_map[0]["target_color"]]
        img_labels = ["background"]

    else:

        nonzero_classes = label.cpu().numpy().nonzero()[0] + 1
        label_colors = [class_map[indx]["target_color"] for indx in nonzero_classes]

        if object_count:
            img_labels = [
                class_map[indx]["name"] + ": {}".format(label[indx - 1])
                for indx in nonzero_classes
            ]
        else:
            img_labels = [class_map[indx]["name"] for indx in nonzero_classes]

    scaling_ratio = img_size / 512

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1 * scaling_ratio
    thickness = 1
    lineType = 1

    y0, x0, dy = 27 * scaling_ratio, 10 * scaling_ratio, 27 * scaling_ratio
    if pred:
        y0, x0 = 400 * scaling_ratio, 315 * scaling_ratio
        thickness = 2

    for i, (img_label, label_color, c) in enumerate(
        zip(img_labels, label_colors, nonzero_classes)
    ):
        y = y0 + c * dy
        fontColor = label_color

        cv2.putText(
            img,
            img_label,
            (int(x0), int(y)),
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )

    img = torch.from_numpy(img).permute(2, 0, 1)

    return img