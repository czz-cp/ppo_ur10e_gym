#!/usr/bin/env python3

"""
URDF验证脚本
验证UR10e URDF文件的结构和内容
"""

import os
import xml.etree.ElementTree as ET

def validate_urdf(urdf_path):
    """验证URDF文件"""
    print(f"🔍 验证URDF文件: {urdf_path}")

    if not os.path.exists(urdf_path):
        print(f"❌ URDF文件不存在: {urdf_path}")
        return False

    try:
        # 解析XML
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        # 统计元素
        links = root.findall('link')
        joints = root.findall('joint')

        print(f"✅ URDF文件解析成功")
        print(f"   机器人名称: {root.get('name', 'Unknown')}")
        print(f"   连接数量: {len(links)}")
        print(f"   关节数量: {len(joints)}")

        # 显示关节信息
        print(f"\n📋 关节列表:")
        for i, joint in enumerate(joints):
            name = joint.get('name', f'joint_{i}')
            joint_type = joint.get('type', 'unknown')
            parent = joint.find('parent')
            child = joint.find('child')
            parent_link = parent.get('link') if parent is not None else 'unknown'
            child_link = child.get('link') if child is not None else 'unknown'

            print(f"   {i+1:2d}. {name:20s} ({joint_type:10s}) {parent_link:15s} -> {child_link:15s}")

        # 显示连接信息
        print(f"\n📋 连接列表:")
        for i, link in enumerate(links):
            name = link.get('name', f'link_{i}')
            inertial = link.find('inertial')
            mass = inertial.find('mass') if inertial is not None else None
            mass_value = mass.get('value', '0') if mass is not None else 'unknown'

            print(f"   {i+1:2d}. {name:20s} (质量: {mass_value:6s} kg)")

        # 检查UR10e特定结构
        required_joints = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        joint_names = [joint.get('name') for joint in joints]

        print(f"\n🎯 UR10e关节检查:")
        for req_joint in required_joints:
            if req_joint in joint_names:
                print(f"   ✅ {req_joint}")
            else:
                print(f"   ❌ {req_joint} (缺失)")

        missing_joints = [j for j in required_joints if j not in joint_names]

        if not missing_joints:
            print(f"🎉 所有UR10e必需关节都存在！")
            return True
        else:
            print(f"⚠️ 缺少 {len(missing_joints)} 个必需关节")
            return False

    except Exception as e:
        print(f"❌ URDF文件解析失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 URDF验证工具")
    print("=" * 50)

    # 验证两个URDF文件
    urdf_files = [
        "ur10e.urdf",          # 从isaac_gym_manipulator复制的原始文件
        "ur10e_isaac.urdf"     # 为Isaac Gym优化的简化文件
    ]

    for urdf_file in urdf_files:
        if os.path.exists(urdf_file):
            print(f"\n{'='*20} {urdf_file} {'='*20}")
            validate_urdf(urdf_file)
        else:
            print(f"\n❌ 文件不存在: {urdf_file}")

    print(f"\n{'='*60}")
    print("✅ URDF验证完成")

if __name__ == "__main__":
    main()