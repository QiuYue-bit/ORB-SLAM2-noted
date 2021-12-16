###
 # @Author: your name
 # @Date: 2021-08-23 22:12:59
 # @LastEditTime: 2021-08-26 14:53:36
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /undefined/home/divenire/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments/run_rgbd_dataset.sh
### 

#Tum dataset
#in dir: /home/divenire/Divenire_ws/dataset/tum/tool

# 1.Associate
cd /home/divenire/Divenire_ws/dataset/tum/tool
python associate.py ../rgbd_dataset_freiburg1_xyz/rgb.txt ../rgbd_dataset_freiburg1_xyz/depth.txt > rgbd_dataset_freiburg1_xyz_association.txt
cp ./rgbd_dataset_freiburg1_xyz_association.txt \
~/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments/results/rgbd/

mv rgbd_dataset_freiburg1_xyz_association.txt ../rgbd_dataset_freiburg1_xyz/

# 2.run
cd ~/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments/
./Examples/RGB-D/rgbd_tum \
Vocabulary/ORBvoc.txt \
Examples/RGB-D/TUM1.yaml \
~/Divenire_ws/dataset/tum/rgbd_dataset_freiburg1_xyz \
./results/rgbd/rgbd_dataset_freiburg1_xyz_association.txt

# 3.save results
cp CameraTrajectory2.txt results/rgbd/tum_frb1xyz_orb2_cam.txt
cp KeyFrameTrajectory.txt results/rgbd/tum_frb1xyz_orb2_keyframe.txt