import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, PathJoinSubstitution
from launch_ros.actions import Node

def generate_launch_description():
    pkg_path = get_package_share_directory('unitree_g1_description')
    rviz_pkg_path = get_package_share_directory('unitree_g1_rviz')

    # 1. Define the 'model' argument with strict choices
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='29dof',
        choices=['29dof'],  # add '23dof' once g1_23dof.urdf is in urdf_xml/
        description='G1 variant (URDF file g1_<model>.urdf)'
    )

    # 2. Dynamically build the URDF filename based on the argument
    # model:=29dof -> urdf_xml/g1_29dof.urdf
    model_config = LaunchConfiguration('model')
    urdf_filename = [ 'g1_', model_config, '.urdf' ]
    urdf_path = PathJoinSubstitution([pkg_path, 'urdf_xml', urdf_filename])

    # 3. Robot State Publisher Node
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': Command(['xacro ', urdf_path])}]
    )

    # 4. Joint State Publisher GUI (Sliders will update automatically)
    joint_state_publisher_gui = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui'
    )

    # 5. RViz Node
    rviz2 = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', os.path.join(rviz_pkg_path, 'rviz', 'rviz.rviz')]
    )

    return LaunchDescription([
        model_arg,
        robot_state_publisher,
        joint_state_publisher_gui,
        rviz2
    ])