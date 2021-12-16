###
 # @Author: Divenire
 # @Date: 2021-08-23 22:12:59
 # @LastEditTime: 2021-09-15 20:34:56
 # @LastEditors: Please set LastEditors
 # @Description: run dataset
 # @FilePath: /undefined/home/divenire/Divenire_ws/workingProgram/ORB_SLAM/ORB_SLAM2_detailed_comments/run_stereo_dataset.sh
### 

# EuRoC Dataset v1 and v2
# ./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml \
# ~/Divenire_ws/dataset/EuRoC/V1_02_medium/mav0/cam0/data \
# ~/Divenire_ws/dataset/EuRoC/V1_02_medium/mav0/cam1/data \
# Examples/Stereo/EuRoC_TimeStamps/V102.txt

# EuRoC Dataset MH
# ./Examples/Stereo/stereo_euroc Vocabulary/ORBvoc.txt Examples/Stereo/EuRoC.yaml \
# ~/Divenire_ws/dataset/EuRoC/MH_01_easy/mav0/cam0/data \
# ~/Divenire_ws/dataset/EuRoC/MH_01_easy/mav0/cam1/data \
# ./Examples/Stereo/EuRoC_TimeStamps/MH01.txt


# KITTI 
./Examples/Stereo/stereo_kitti ./Vocabulary/ORBvoc.txt \
./Examples/Stereo/KITTI00-02.yaml \
~/Divenire_ws/dataset/KITTI/dataset/sequences/00