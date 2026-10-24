import subprocess
def sh(c): print("+", c, flush=True); subprocess.run(c, shell=True)
sh("pip install -q mujoco==3.17.0 gymnasium==1.4.0 gymnasium-robotics==1.4.2")
code = r'''
import mujoco, gymnasium_robotics, os, numpy as np
from importlib.metadata import version
print("versions", version("gymnasium-robotics"), version("mujoco"), mujoco.__file__)
from gymnasium_robotics.envs.fetch import push
path = os.path.join(os.path.dirname(push.__file__), "..", "assets", "fetch", "push.xml")
m = mujoco.MjModel.from_xml_path(os.path.abspath(path))
print("njnt", m.njnt, "jnt_type dtype", m.jnt_type.dtype)
for j in range(m.njnt):
    print(j, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_type[j]), repr(m.jnt_type[j]))
print("enum", int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
'''
open("/kaggle/tmp_probe.py", "w").write(code)
sh("python /kaggle/tmp_probe.py")
