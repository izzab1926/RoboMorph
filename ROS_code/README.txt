paste the script "interactive_marker.py" inside:

/ws_panda/src/franka_ros/franka_example_controllers/scripts

(In order to not substite the original one, rename _ORIGINAL the already present.)

where ws_panda is the catkin_space from:

https://github.com/frankaemika/franka_ros

-

Execute from terminal after building catkin workspace.

roslaunch franka_gazebo panda.launch x:=-0.5 controller:=cartesian_impedance_example_controller     rviz:=true
