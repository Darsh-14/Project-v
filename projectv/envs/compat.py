"""Compatibility shim for gymnasium_robotics.utils.mujoco_utils joint helpers.

On the Kaggle image (Linux, Python 3.13.15, mujoco 3.17.0, gymnasium-robotics 1.4.2) comparing
`model.jnt_type[i]` (numpy int32) with `mujoco.mjtJoint.mjJNT_FREE` evaluates False for the free
joint, so `FetchPush` construction hits the hinge/slide assertion. The same code works on
Windows. The shim passes the helpers a model proxy whose `jnt_type` holds Python ints, which
compare correctly with the enum everywhere. Behaviour is otherwise unchanged.
"""

from __future__ import annotations

import functools

from gymnasium_robotics.utils import mujoco_utils

_PATCHED = ("get_joint_qpos", "set_joint_qpos", "get_joint_qvel", "set_joint_qvel")


class _IntJointTypes:
    def __init__(self, model):
        self._model = model
        self.jnt_type = [int(t) for t in model.jnt_type]

    def __getattr__(self, name):
        return getattr(self._model, name)


def _wrap(fn):
    @functools.wraps(fn)
    def inner(model, *args, **kwargs):
        return fn(_IntJointTypes(model), *args, **kwargs)

    inner._pv_patched = True
    return inner


def apply() -> None:
    for name in _PATCHED:
        fn = getattr(mujoco_utils, name)
        if not getattr(fn, "_pv_patched", False):
            setattr(mujoco_utils, name, _wrap(fn))
