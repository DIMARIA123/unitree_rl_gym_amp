from setuptools import find_packages
from distutils.core import setup

setup(name='unitree_rl_gym',
      version='1.0.0',
      author='Unitree Robotics',
      license="BSD-3-Clause",
      packages=find_packages(where="utils"),         # 在 utils 目录下查找所有子包
      package_dir={"": "utils"},                     # 告诉 pip 包的根路径是 utils/
      author_email='support@unitree.com',
      description='Template RL environments for Unitree Robots',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.24', 'tensorboard', 'mujoco==3.2.3', 'pyyaml'])
