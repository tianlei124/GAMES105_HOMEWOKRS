import math
import numpy as np
from scipy.spatial.transform import Rotation as R


def modulus_length(v):
    return np.sqrt(np.sum(v**2))

def normalize(v):
    return v / modulus_length(v)

def calculate_rotation(v1, v2):
    u = normalize(np.cross(v1, v2))
    theta = math.acos(np.dot(v1, v2)/(modulus_length(v1)*modulus_length(v2)))
    return R.from_rotvec(u*theta).as_quat()


class InverseKinematics:
    def __init__(self, meta_data, joint_positions, joint_orientations, target_pose):
        self.path, self.path_name, self.path1, self.path2 = meta_data.get_path_from_root_to_end()
        self.path1 = list(reversed(self.path1))
        self.joint_name = meta_data.get_joint_name()
        self.joint_offset = meta_data.get_joint_offset()
        self.joint_parent = meta_data.get_joint_parent()
        self.joint_positions = joint_positions
        self.joint_orientations = joint_orientations
        self.target_pose = target_pose
        print(self.path, self.path_name, self.path1, self.path2)

    def ccd_method(self):
        target_distance = modulus_length(
            self.joint_positions[self.path[-1]] - self.target_pose)
        while target_distance > 0.01:
            index = len(self.path) - 2
            while index >= 0:
                # print(f"Index: {self.path[index]}, target distance: {target_distance}")
                current_position = self.joint_positions[self.path[index]]
                target_direction = normalize(
                    self.target_pose - current_position)
                current_direction = normalize(
                    self.joint_positions[self.path[-1]] - current_position)
                rotation = R.from_quat(calculate_rotation(
                    current_direction, target_direction))
                self.update_joint_pose(index, rotation)
                # if index < (len(self.path) - len(self.path1) -1 ):
                #     break
                index -= 1
                target_distance = modulus_length(
                    self.joint_positions[self.path[-1]] - self.target_pose)
            # break
        return self.joint_positions, self.joint_orientations

    def update_joint_pose(self, index, rotation):
        divide = len(self.path) - len(self.path1)
        if (index >= divide):
            self.update_upper_pose(index - len(self.path2), rotation)
        else:
            self.updaet_lower_pose(index, rotation)

    def update_upper_pose(self, index, rotation):
        origin_orientatins = self.joint_orientations
        origin_rotation = R.from_matrix(np.transpose(R.from_quat(
            origin_orientatins[self.path1[index-1]]).as_matrix())) * R.from_quat(origin_orientatins[self.path1[index]])
        if self.path1[index] > 0:
            self.joint_orientations[self.path1[index]] = (R.from_quat(
                self.joint_orientations[self.path1[index-1]]) * origin_rotation * rotation).as_quat()
        else:
            self.joint_orientations[self.path1[index]] = rotation.as_quat()
        for j in range(self.path1[index+1], len(self.joint_orientations)):
            self.joint_positions[j] = self.joint_positions[self.joint_parent[j]] + R.from_quat(
                self.joint_orientations[self.joint_parent[j]]).apply(self.joint_offset[j])
            local_rotation = R.from_matrix(np.transpose(R.from_quat(
                origin_orientatins[self.joint_parent[j]]).as_matrix())) * R.from_quat(origin_orientatins[j])
            self.joint_orientations[j] = (R.from_quat(
                self.joint_orientations[self.joint_parent[j]]) * local_rotation).as_quat()

    def updaet_lower_pose(self, index, rotation):
        origin_orientatins = self.joint_orientations
        origin_rotation = R.from_matrix(np.transpose(R.from_quat(
            origin_orientatins[self.path2[index-1]]).as_matrix())) * R.from_quat(origin_orientatins[self.path2[index]])
        if self.path2[index] > 0:
            self.joint_orientations[self.path2[index]] = (R.from_quat(
                self.joint_orientations[self.path2[index-1]]) * origin_rotation * rotation.inv()).as_quat()
            for j in range(self.path2[index-1], len(self.joint_orientations)):
                self.joint_positions[j] = self.joint_positions[self.joint_parent[j]] + R.from_quat(
                    self.joint_orientations[self.joint_parent[j]]).apply(self.joint_offset[j])
                local_rotation = R.from_matrix(np.transpose(R.from_quat(
                    origin_orientatins[self.joint_parent[j]]).as_matrix())) * R.from_quat(origin_orientatins[j])
                self.joint_orientations[j] = (R.from_quat(
                    self.joint_orientations[self.joint_parent[j]]) * local_rotation).as_quat()
        else:
            self.joint_orientations[self.path2[index]] = rotation.as_quat()
            self.joint_positions[self.path2[index]] = self.joint_positions[self.path2[index+1]] + R.from_quat(
                    self.joint_orientations[self.path2[index+1]]).apply(self.joint_offset[index])


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """

    return InverseKinematics(meta_data, joint_positions, joint_orientations, target_pose).ccd_method()

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations