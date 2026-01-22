# MIT License
#
# Copyright (c) 2024 <COPYRIGHT_HOLDERS>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

"""
OmniGraph core Python API:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph/latest/Overview.html

OmniGraph attribute data types:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/attribute_types.html

Collection of OmniGraph code examples in Python:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.docs/latest/dev/ogn/ogn_code_samples_python.html

Collection of OmniGraph tutorials:
  https://docs.omniverse.nvidia.com/kit/docs/omni.graph.tutorials/latest/Overview.html
"""

import math

import numpy as np
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.numpy.rotations import (
    euler_angles_to_quats,
    quats_to_euler_angles,
)
from scipy.spatial.transform import Rotation as R


class OgnPowCameraTeleopRosPyInternalState:
    """Convenience class for maintaining per-node state information"""

    def __init__(self):
        """Instantiate the per-node state information"""
        self.target_prim: XFormPrim = None
        self.target_prim_path: str = ""
        self.last_time: float = 0.0


class OgnPowCameraTeleopRosPy:
    """The Ogn node class"""

    @staticmethod
    def internal_state():
        """Returns an object that contains per-node state information"""
        return OgnPowCameraTeleopRosPyInternalState()

    @staticmethod
    def compute(db) -> bool:
        """Compute the output based on inputs and internal state"""
        state = db.internal_state
        input_target_prim_path = db.inputs.targetPrim[0].GetString()

        if not db.inputs.targetPrim:
            db.log_error("Target Prim input is not set.")
            return False

        try:
            # Input values from .ogn (except execIn, targetPrim)
            angularX = db.inputs.angularX
            angularY = db.inputs.angularY
            angularZ = db.inputs.angularZ
            linearX = db.inputs.linearX
            linearY = db.inputs.linearY
            linearZ = db.inputs.linearZ
            time = db.inputs.time

            if state.target_prim_path != input_target_prim_path:
                state.target_prim_path = input_target_prim_path
                state.target_prim = XFormPrim(state.target_prim_path)

            # Approximate delta time to 60 FPS (0.016667s) on first run
            dt = max(time - state.last_time, 0.016667)

            # Compute delta rotation quaternion from angular velocity
            angle_x_rad = angularX * dt
            angle_y_rad = angularY * dt
            angle_z_rad = angularZ * dt

            # Compute delta position from linear velocity
            px = linearX * dt
            py = linearY * dt
            pz = linearZ * dt

            current_pos, current_rot_quat = state.target_prim.get_world_poses()

            # Update rotation
            euler_current = quats_to_euler_angles(current_rot_quat)
            euler_delta = np.array([[angle_x_rad, angle_y_rad, angle_z_rad]])
            euler_new = euler_current + euler_delta
            updated_rot_quat = euler_angles_to_quats(euler_new)

            # Update position, FPS-style movement:
            # Use the new Yaw to determine movement direction on the ground plane.
            yaw_new = euler_new[:, 2]  # Z-axis rotation (yaw)
            r_yaw = R.from_euler("z", yaw_new)

            delta_pos_target = np.array([[px, py, pz]])
            delta_pos_world = r_yaw.apply(delta_pos_target)
            updated_pos = current_pos + delta_pos_world

            state.last_time = time

            state.target_prim.set_world_poses(
                updated_pos,
                updated_rot_quat,
            )

        except Exception as e:
            db.log_error(f"Computation error: {e}")
            return False
        return True
