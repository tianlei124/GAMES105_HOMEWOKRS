import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    joint_name = []
    joint_parent = []
    joint_offset = []
    index = -1
    stack = [index]
    with open(bvh_file_path, "r") as bvh:
        for des in bvh.readlines():
            if "ROOT" in des or "JOINT" in des:
                joint_parent.append(stack[-1])
                joint_name.append(des.split(" ")[-1].strip())
            if "End Site" in des:
                joint_parent.append(stack[-1])
                joint_name.append(joint_name[joint_parent[-1]] + "_end")
            if "{" in des:
                index += 1
                stack.append(index)
            if "OFFSET" in des:
                joint_offset.append([float(x) for x in des.split("  ")[-3:]])
            if "}" in des:
                stack.pop()
            if "MOTION" in des:
                break
    return joint_name, joint_parent, np.array(joint_offset)


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    current_motion = motion_data[frame_id]
    joint_positions = [current_motion[:3] + joint_offset[0]]
    joint_orientations = [R.from_euler(
        "XYZ", current_motion[3:6], degrees=True).as_quat()]
    j = 1
    for i in range(1, len(joint_name)):
        index = joint_parent[i]
        joint_positions.append(
            joint_positions[index] + R.from_quat(joint_orientations[index]).apply(joint_offset[i]))
        if "_end" in joint_name[i]:
            joint_orientations.append(
                np.zeros(joint_orientations[index].shape))
        else:
            local_rotation = R.from_euler(
                "XYZ", current_motion[3*(j+1):3*(j+2)], degrees=True).as_quat()
            joint_orientations.append(
                (R.from_quat(joint_orientations[index]) * R.from_quat(local_rotation)).as_quat())
            j += 1
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    joint_name_t, joint_parent_t, joint_offset_t = part1_calculate_T_pose(T_pose_bvh_path)
    joint_name_a = []
    Rl = R.from_euler("XYZ", [0, 0, -45.], degrees=True) # lShoulder Ri(TPose->APose)
    Rr = R.from_euler("XYZ", [0, 0, 45.0], degrees=True) # rShoulder Ri(TPose->APose)
    motion_data = []
    joint_count, li, ri = 0, 0, 0
    with open(A_pose_bvh_path, "r") as bvh:
        valid_data = False
        for line in bvh.readlines():
            if not valid_data:
                if "ROOT" in line or "JOINT" in line:
                    joint_count += 1
                    if "lShoulder" in line:
                        li = joint_count
                    if "rShoulder" in line:
                        ri = joint_count
                    joint_name_a.append(line.split()[-1].strip())
                if "Frame Time:" in line:
                    valid_data = True
                continue
            data = [float(x) for x in line.split()]
            if len(data) > 0:
                data[3 * li : 3 * (li + 1)] = (Rl * R.from_euler("XYZ", data[3 * li : 3 * (li + 1)], degrees=True)).as_euler("XYZ", degrees=True)
                data[3 * ri : 3 * (ri + 1)] = (Rr * R.from_euler("XYZ", data[3 * ri : 3 * (ri + 1)], degrees=True)).as_euler("XYZ", degrees=True)
                correct_data = data[:3]
                for name in joint_name_t:
                    if "_end" not in name:
                        index = joint_name_a.index(name)
                        correct_data.extend(data[(index + 1)*3:(index+2)*3])
                motion_data.append(correct_data)
    motion_data = np.array(motion_data)
    return motion_data
