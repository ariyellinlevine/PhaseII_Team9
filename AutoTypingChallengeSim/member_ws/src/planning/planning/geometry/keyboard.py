"""Where the keys are: the public key layout (INTERFACES.md 6.3) and where the key grid sits on the panel."""
import numpy as np

# INTERFACES.md section 6.3: 1u = 19.05 mm. Only the keys a launch key can use (A-Z, 0-9).
PITCH = 0.01905

# Where the key grid's top-left corner sits on the panel, in metres in the board frame (x right, y down from the
# panel's top-left corner). INTERFACES.md does not publish this. It was measured from the camera image at the home
# pose (matching the grid photo to a top-down view of the panel) and came out the same, to 0.1 mm, on every episode.
KEYBOARD_ORIGIN = [0.0406, 0.0422]

# (top of the row in units, x of the first key in units, keys in order); every one of these keys is 1u wide.
ROWS = [
    (1.25, 1.0, '1234567890'),
    (2.25, 1.5, 'QWERTYUIOP'),
    (3.25, 1.75, 'ASDFGHJKL'),
    (4.25, 2.25, 'ZXCVBNM'),
]

CELLS = {key: (x + i, y) for y, x, keys in ROWS for i, key in enumerate(keys)}


def key_center(key):
    """Centre of a key in metres, in the key grid's frame: origin at its top-left corner, x right, y down."""
    x, y = CELLS[key]
    return np.array([x + 0.5, y + 0.5]) * PITCH
