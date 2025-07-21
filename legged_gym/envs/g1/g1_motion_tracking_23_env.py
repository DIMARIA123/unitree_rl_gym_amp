from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from isaacgym import gymapi
from motion_lib.motion_lib_robot import MotionLibRobot

class G1_23MotionTracking(G1Robot):
    
    def __init__(
        self, 
        cfg: LeggedRobotCfg, 
        sim_params: gymapi.SimParams,
        physics_engine: int = gymapi.SIM_PHYSX, 
        sim_device: str = 'cuda:0', 
        headless: bool = True
    ):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._motion_lib = MotionLibRobot(self.cfg.robot, num_envs=self.num_envs, device=self.device)
        motion = self._motion_lib.load_motions()
        # motion_sk = self._motion_lib.load_motion_with_skeleton()
        print("debug")