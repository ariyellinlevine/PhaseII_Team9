"""The 5-joint arm: forward kinematics, inverse kinematics for the head, and aiming the stylus."""
from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = ['base_yaw', 'shoulder_pitch', 'elbow_pitch', 'head_pan', 'head_tilt']

BASE_HEIGHT = 0.30
UPPER_ARM = 0.60
FOREARM = 0.40
STYLUS_MIN = 0.05
STYLUS_MAX = 0.35

Q_MIN = np.deg2rad([-120.0, -30.0, -140.0, -45.0, -35.0])
Q_MAX = np.deg2rad([120.0, 100.0, 0.0, 45.0, 35.0])
Q_HOME = np.deg2rad([0.0, 85.0, -113.0, 0.0, 0.0])

TOLERANCE = 1e-9
FORWARD = np.array([1.0, 0.0, 0.0])


class IKError(ValueError):
    """The requested head position cannot be reached inside the joint limits."""


class Aim(NamedTuple):
    pan: float
    tilt: float
    range: float


def link_rotation(yaw, pitch):
    # A positive pitch raises the link, which is a negative rotation about Y.
    return Rotation.from_euler('ZY', [yaw, -pitch])


def head_position(q):
    yaw, shoulder, elbow = q[:3]
    upper_arm = link_rotation(yaw, shoulder).apply(UPPER_ARM * FORWARD)
    forearm = link_rotation(yaw, shoulder + elbow).apply(FOREARM * FORWARD)
    return np.array([0.0, 0.0, BASE_HEIGHT]) + upper_arm + forearm


def head_rotation(q):
    return link_rotation(q[0], q[1] + q[2])


def aim_direction(q):
    pan, tilt = q[3:5]
    return head_rotation(q).apply(link_rotation(pan, tilt).apply(FORWARD))


def position_ik(p_head):
    """The arm joints (yaw, shoulder, elbow) that put the head at `p_head`, elbow bent down."""
    x, y, z = p_head
    reach = np.hypot(x, y)
    height = z - BASE_HEIGHT
    distance = np.hypot(reach, height)

    cos_elbow = (distance ** 2 - UPPER_ARM ** 2 - FOREARM ** 2) / (2 * UPPER_ARM * FOREARM)
    if cos_elbow > 1 + TOLERANCE:
        raise IKError(f'head target is too far: {distance:.3f} m from the shoulder, '
                      f'the arm reaches {UPPER_ARM + FOREARM:.2f} m')
    if cos_elbow < -1 - TOLERANCE:
        raise IKError(f'head target is too close: {distance:.3f} m from the shoulder, '
                      f'the arm folds to {UPPER_ARM - FOREARM:.2f} m')

    elbow = -np.arccos(np.clip(cos_elbow, -1.0, 1.0))
    yaw = np.arctan2(y, x)
    shoulder = (np.arctan2(height, reach)
                - np.arctan2(FOREARM * np.sin(elbow), UPPER_ARM + FOREARM * np.cos(elbow)))

    q = np.array([yaw, shoulder, elbow])
    for name, angle, low, high in zip(JOINT_NAMES, q, Q_MIN, Q_MAX):
        if not low - TOLERANCE <= angle <= high + TOLERANCE:
            raise IKError(f'{name} would need {np.rad2deg(angle):.1f} deg, '
                          f'its limits are {np.rad2deg(low):.0f} to {np.rad2deg(high):.0f} deg')
    return q


def aim_inverse(q_arm, target):
    """The pan, tilt and stylus range that point the head at `target` from the arm pose `q_arm`."""
    offset = np.asarray(target) - head_position(q_arm)
    x, y, z = head_rotation(q_arm).inv().apply(offset)
    return Aim(np.arctan2(y, x), np.arctan2(z, np.hypot(x, y)), np.linalg.norm(offset))


def aim_violations(pan, tilt, stylus_range):
    problems = []
    if not Q_MIN[3] <= pan <= Q_MAX[3]:
        problems.append(f'pan {np.rad2deg(pan):.1f} deg is outside its limits')
    if not Q_MIN[4] <= tilt <= Q_MAX[4]:
        problems.append(f'tilt {np.rad2deg(tilt):.1f} deg is outside its limits')
    if not STYLUS_MIN <= stylus_range <= STYLUS_MAX:
        problems.append(f'range {stylus_range:.3f} m is outside the stylus window')
    return problems
