#!/usr/bin/env python3
import time
from hpaf.core.io import load_config
from hpaf.robot.piper_backend import PiperArm


def main():
    cfg = load_config('configs/demo.yaml')
    arm = PiperArm(
        can_name=cfg.robot.can_name,
        speed_percent=cfg.robot.speed_percent,
        move_mode=cfg.robot.move_mode,
        pose_wait_s=cfg.robot.pose_wait_s,
        grip_wait_s=cfg.robot.grip_wait_s,
        pose_command_hz=cfg.robot.pose_command_hz,
        pose_command_duration_s=cfg.robot.pose_command_duration_s,
        gripper_command_hz=cfg.robot.gripper_command_hz,
        gripper_command_duration_s=cfg.robot.gripper_command_duration_s,
        gripper_max_width_mm=cfg.robot.gripper_max_width_mm,
        gripper_mm_to_ctrl=cfg.robot.gripper_mm_to_ctrl,
        gripper_open_force=cfg.robot.gripper_open_force,
        gripper_close_force=cfg.robot.gripper_close_force,
    )
    arm.connect()
    while True:
        print('open 45mm')
        arm.open_gripper(width_mm=45.0)
        time.sleep(1.0)
        print('open 25mm')
        arm.open_gripper(width_mm=25.0)
        time.sleep(1.0)
        print('close')
        arm.close_gripper(force=cfg.robot.gripper_close_force)
        time.sleep(1.0)


if __name__ == '__main__':
    main()
