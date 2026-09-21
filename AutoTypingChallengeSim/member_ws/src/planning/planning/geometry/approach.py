"""How to approach a key and when it is safe to press it."""
import itertools

import numpy as np
from scipy.spatial.transform import Rotation

from planning.geometry import kinematics as kin
from planning.geometry.panel import Z_AXIS

# A press is only attempted well inside the simulator's limits (0.05-0.35 m, 55 deg, 0.02 rad/s).
PRESS_RANGE = (0.08, 0.32)
PRESS_MAX_INCIDENCE = np.deg2rad(45.0)
MAX_JOINT_SPEED = 0.012
# A key cap is about 17 mm across, and the estimate of the keyboard is a couple of millimetres off.
AIM_TOLERANCE = 0.003
# Pan and tilt are kept this far from their stops so the aim never sits against one.
PAN_TILT_MARGIN = np.deg2rad(5.0)

# Head distances from a target to try, in order: the middle of the stylus window first.
STANDOFFS = [0.25, 0.20, 0.30, 0.15]
# How far the approach may lean off square-on, about the panel's x and y axes.
LEAN_ANGLES = np.deg2rad([0.0, 10.0, -10.0, 20.0, -20.0, 30.0, -30.0])


def press_window_violations(hit):
    """Why the stylus range or angle at `hit` is outside the comfortable press window."""
    problems = []
    if not PRESS_RANGE[0] <= hit.range <= PRESS_RANGE[1]:
        problems.append(f'range {hit.range:.3f} m is outside the press window')
    if hit.incidence > PRESS_MAX_INCIDENCE:
        problems.append(f'incidence {np.rad2deg(hit.incidence):.1f} deg is too glancing')
    return problems


def press_problems(hit, key_center, joint_speed):
    """Everything that keeps a press at `key_center` from being safe right now, or [] when nothing does.

    `hit` is where the stylus meets the keyboard (a `Hit`, or None if it does not), `joint_speed` the fastest joint.
    """
    if hit is None:
        return ['the stylus does not point at the panel']

    problems = press_window_violations(hit)
    error = np.linalg.norm(hit.xy - key_center)
    if error > AIM_TOLERANCE:
        problems.append(f'aim is {error * 1000:.1f} mm off')
    if joint_speed > MAX_JOINT_SPEED:
        problems.append(f'speed {joint_speed:.3f} rad/s')
    return problems


def can_press_from(q_arm, target, panel):
    """Whether pan and tilt alone can aim at the target from this arm pose, with room to spare."""
    pan, tilt, stylus_range = kin.aim_inverse(q_arm, target)
    if np.any(np.abs([pan, tilt]) > kin.Q_MAX[3:] - PAN_TILT_MARGIN):   # the pan and tilt limits are symmetric
        return False
    head = kin.head_position(q_arm)
    hit = panel.hit(head, (np.asarray(target) - head) / stylus_range)
    return hit is not None and not press_window_violations(hit)


def approach_point(target, panel):
    """A head position in front of `target`, as square-on as the arm allows, that can still aim at it."""
    target = np.asarray(target)
    leans = sorted(itertools.product(LEAN_ANGLES, repeat=2), key=lambda lean: np.hypot(*lean))
    for lean in leans:
        direction = (panel.rotation * Rotation.from_euler('xy', lean)).apply(-Z_AXIS)
        for standoff in STANDOFFS:
            p_head = target + standoff * direction
            try:
                q = kin.position_ik(p_head)
            except kin.IKError:
                continue
            if not kin.aim_violations(*kin.aim_inverse(q, target)):
                return p_head
    raise kin.IKError('no head position in front of the target can reach it and aim at it')
