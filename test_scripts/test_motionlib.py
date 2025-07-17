# 在 main.py 中使用 motion_lib
import sys
import os

# 将项目根目录加入 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from legged_gym.envs.g1.g1_motion_tracking_23_env import G1_23MotionTracking
from legged_gym.envs.g1.g1_motion_tracking_23_config import G1_23MotionTrackingCfg
from legged_gym.utils.helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, set_seed, parse_sim_params
from isaacgym import gymapi

g1_config = G1_23MotionTrackingCfg()

sim_params = {"sim": class_to_dict(g1_config.sim)}
args = get_args()
sim_params = parse_sim_params(args, sim_params)

g1 = G1_23MotionTracking(g1_config, sim_params, gymapi.SIM_PHYSX, 'cuda:0', False)
