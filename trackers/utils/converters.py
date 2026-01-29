import numpy as np


def xyxy_to_xcycsr(xyxy: np.ndarray) -> np.ndarray:
    """
    Converts bounding box coordinates from `(x_min, y_min, x_max, y_max)`
    format to `(x_center, y_center, scale, aspect_ratio)` format.

    Args:
        xyxy: A numpy array of shape `(4)` whichcorresponds bounding box
            in the format `(x_min, y_min, x_max, y_max)`.

    Returns:
        A numpy array of shape `(4)` that  corresponds to a bounding box
        in the format `(x_min, y_min, x_max, y_max)`.
    """
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]
    x = xyxy[0] + w / 2.0
    y = xyxy[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-6)
    return np.array([x, y, s, r])


def xcycsr_to_xyxy(xcycsr: np.ndarray) -> np.ndarray:
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
      [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    Args:
        bbox: bounding box in the form [x,y,s,r]
    Returns:
        np.ndarray: bounding box in the form [x1,y1,x2,y2]
    """
    w = np.sqrt(xcycsr[2] * xcycsr[3])
    h = xcycsr[2] / w
    return np.array(
        [
            xcycsr[0] - w / 2.0,
            xcycsr[1] - h / 2.0,
            xcycsr[0] + w / 2.0,
            xcycsr[1] + h / 2.0,
        ]
    )
