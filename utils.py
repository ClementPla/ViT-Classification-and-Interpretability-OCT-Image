import cv2
import numpy as np
from nntools.dataset.image_tools import nntools_wrapper


@nntools_wrapper
def to_rgb(image):
    if image.ndim != 3:

        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        return image

@nntools_wrapper
def crop_fundus(image):
    try:
        lower_img = cv2.cvtColor(image[425:, 200:], cv2.COLOR_BGR2GRAY)
    except:
        lower_img = image[425:, 200:]

    indices = lower_img == 255
    not_null_pixels = np.nonzero(indices)
    if not_null_pixels[1].any():
        x = np.min(not_null_pixels[1])
    else:
        x = 0
    return image[:, 200 + x - 10:]


@nntools_wrapper
def quick_resize(image):
    shape = (224, 224)
    return cv2.resize(image, dsize=shape, interpolation=cv2.INTER_LINEAR)


@nntools_wrapper
def fundus_autocrop(image):
    mean_img = np.mean(image, axis=2)
    threhshold_img = mean_img > 15
    not_null_pixels = np.nonzero(threhshold_img)
    if not_null_pixels[0].size == 0:
        return image
    x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
    y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
    return image[y_range[0]:y_range[1], x_range[0]:x_range[1]]