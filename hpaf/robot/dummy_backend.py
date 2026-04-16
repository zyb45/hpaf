from hpaf.core.models import Pose

class DummyArm:
    def connect(self):
        print('[DummyArm] connect')

    def move_to_pose(self, pose: Pose):
        print('[DummyArm] move_to_pose', pose)

    def open_gripper(self):
        print('[DummyArm] open_gripper')

    def close_gripper(self, force=800):
        print('[DummyArm] close_gripper', force)
