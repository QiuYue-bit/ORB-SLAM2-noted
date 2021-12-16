###
 # @Author: your name
 # @Date: 2021-08-23 22:12:59
 # @LastEditTime: 2021-08-30 10:14:17
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /undefined/home/divenire/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments/run_mono_dataset.sh
### 

# tum数据集
./Examples/Monocular/mono_tum ./Vocabulary/ORBvoc.txt ./Examples/Monocular/TUM1.yaml ~/Divenire_ws/dataset/tum/rgbd_dataset_freiburg1_xyz

# kitti数据集
./Examples/Monocular/mono_kitti Vocabulary/ORBvoc.txt Examples/Monocular/KITTI00-02.yaml ~/Divenire_ws/dataset/KITTI/dataset/sequences/00

# Euroc
./Examples/Monocular/mono_euroc Vocabulary/ORBvoc.txt Examples/Monocular/EuRoC.yaml ~/Divenire_ws/dataset/EuRoC/V1_02_medium/mav0/cam0/data ./Examples/Monocular/EuRoC_TimeStamps/V102.txt

