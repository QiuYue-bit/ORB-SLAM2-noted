cd /home/divenire/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments
rosrun ORB_SLAM2 RGBD ./Vocabulary/ORBvoc.txt ./Examples/ROS/ORB_SLAM2/D435i.yaml \
/camera/rgb/image_raw:=/camera/color/image_raw   camera/depth_registered/image_raw:=/camera/depth/image_rect_raw

cd /home/divenire/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments
rosrun ORB_SLAM2 Stereo ./Vocabulary/ORBvoc.txt ./Examples/Stereo/EuRoC.yaml true /camera/left/image_raw:=/camera/infra1/image_rect_raw  camera/right/image_raw:=/camera/infra2/image_rect_raw