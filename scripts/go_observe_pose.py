#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from piper_sdk import *

# =========================
# 需要你按自己机械臂实际情况微调的参数
# =========================

CAN_PORT = "can0"

# 全局观察位姿
# 单位：
#   X Y Z: mm
#   RX RY RZ: deg
#
# 这里先给一个比较常见、偏高一点的观察位姿示例
# 你后面可以直接改这组数
OBSERVE_POSE = {
    "x": 188,
    "y": -13,
    "z": 286,
    "rx": 180,
    "ry": 15,
    "rz": 178,
}
# 夹爪张开宽度，单位 mm
# 你说想开大一点提高容错，这里先给 50mm
GRIPPER_OPEN_MM = 70.0

# 连续发送控制指令的次数
SEND_CYCLES = 120

# 每次发送间隔
DT = 0.01


def gripper_mm_to_cmd(width_mm: float) -> int:
    """
    根据你测试成功的 demo 语义：
    50 mm -> 0.05 * 1000 * 1000 = 50000
    所以这里直接按 mm * 1000 转成控制值
    """
    width_mm = max(0.0, width_mm)
    return round(width_mm * 1000.0)


def endpose_to_cmd(pose: dict):
    """
    EndPoseCtrl 的输入格式：
    X,Y,Z,RX,RY,RZ 都是整数
    其中位置和角度都按 *1000
    """
    factor = 1000.0
    x = round(pose["x"] * factor)
    y = round(pose["y"] * factor)
    z = round(pose["z"] * factor)
    rx = round(pose["rx"] * factor)
    ry = round(pose["ry"] * factor)
    rz = round(pose["rz"] * factor)
    return x, y, z, rx, ry, rz


def main():
    print("[INFO] Connecting to PiPER...")
    piper = C_PiperInterface_V2(CAN_PORT)
    piper.ConnectPort()

    while not piper.EnablePiper():
        time.sleep(0.01)

    print("[INFO] PiPER enabled.")

    # 夹爪初始化，按你验证通过的 demo 方式来
    print("[INFO] Initializing gripper...")
    piper.GripperCtrl(0, 1000, 0x02, 0)
    time.sleep(0.2)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    time.sleep(0.2)

    # 先张开夹爪
    gripper_cmd = gripper_mm_to_cmd(GRIPPER_OPEN_MM)
    print(f"[INFO] Opening gripper to about {GRIPPER_OPEN_MM:.1f} mm (cmd={gripper_cmd})")

    # 观察位姿
    x, y, z, rx, ry, rz = endpose_to_cmd(OBSERVE_POSE)
    print("[INFO] Moving to observe pose:")
    print(f"       X={OBSERVE_POSE['x']} mm, Y={OBSERVE_POSE['y']} mm, Z={OBSERVE_POSE['z']} mm")
    print(f"       RX={OBSERVE_POSE['rx']} deg, RY={OBSERVE_POSE['ry']} deg, RZ={OBSERVE_POSE['rz']} deg")

    # 连续发送，保证机械臂和夹爪都稳定执行到位
    for i in range(SEND_CYCLES):
        piper.MotionCtrl_2(0x01, 0x00, 100, 0x00)
        piper.EndPoseCtrl(x, y, z, rx, ry, rz)
        piper.GripperCtrl(abs(gripper_cmd), 1000, 0x01, 0)
        time.sleep(DT)

    print("[INFO] Done. Arm should now be at observe pose with gripper open.")


if __name__ == "__main__":
    main()