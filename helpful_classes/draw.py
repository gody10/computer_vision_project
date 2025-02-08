# Necessary for creating our images.
from skimage.draw import line_aa
import numpy as np

class Draw:
    """
    Class used to draw shapes onto images. Methods return coordinates of 
    corresponding shape on a 2d np array of shape (img_size, img_size). 
    The np rng is used for enabling derministic behaviour. 

    Args:
    img_size (int): draws onto 2d array of shape (img_size, img_size).
    rng (Generator): used for enabling deterministic behaviour. Example 
        of valid rng: rng = np.random.default_rng(12345)

    """

    def __init__(self, img_size, rng):

        self.img_size = img_size
        self.rng = rng

        return None

    def rectangle(self):
        """
        Returns the image coordinates of a rectangle.
        """
        # Min and max rectangle height.
        a = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)
        b = self.rng.integers(self.img_size * 3 / 20, self.img_size * 7 / 20)

        # Initial coordinates of the rectangle.
        xx, yy = np.where(np.ones((a, b)) == 1)

        # Place the rectangle randomly in the image.
        cx = self.rng.integers(0 + a, self.img_size - a)
        cy = self.rng.integers(0 + b, self.img_size - b)

        rectangle = xx + cx, yy + cy

        return rectangle

    def line(self):
        """
        Returns the image coordinates of a line.
        """
        # Randomly choose the start and end coordinates.
        a, b = self.rng.integers(0, self.img_size / 3, size=2)
        c, d = self.rng.integers(self.img_size / 2, self.img_size, size=2)

        # Flip a coin to see if slope of line is + or -.
        coin_flip = self.rng.integers(low=0, high=2)
        # Use a skimage.draw method to draw the line.
        if coin_flip:
            xx, yy, _ = line_aa(a, b, c, d)
        else:
            xx, yy, _ = line_aa(a, d, c, b)

        line = xx, yy

        return line

    def donut(self):
        """
        Returns the image coordinates of an elliptical donut.
        """
        # Define a grid
        xx, yy = np.mgrid[: self.img_size, : self.img_size]
        cx = self.rng.integers(0, self.img_size)
        cy = self.rng.integers(0, self.img_size)

        # Define the width of the donut.
        width = self.rng.uniform(self.img_size / 3, self.img_size)

        # Give the donut some elliptical nature.
        e0 = self.rng.uniform(0.1, 5)
        e1 = self.rng.uniform(0.1, 5)

        # Use the forumula for an ellipse.
        ellipse = e0 * (xx - cx) ** 2 + e1 * (yy - cy) ** 2

        donut = (ellipse < (self.img_size + width)) & (
            ellipse > (self.img_size - width)
        )

        return donut