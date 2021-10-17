cmake_minimum_required(VERSION 3.1)
project(ORB_SLAM2)

############################################################
# Build Type
############################################################
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

############################################################
#  C/C++ Flags
############################################################
# -O3 编译等级优化为3上 开启了这个就不能用debug,会丢失信息了
# -Wall 开启所有警告
# -march=native, which automatically detects the features your CPU supports and sets the options appropriately.
if(CMAKE_BUILD_TYPE MATCHES Release)
   MESSAGE("开启编译器优化")
   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
endif()
########################################s####################
# Check C++14
############################################################
# 设置指定的C++编译器版本是必须的，如果不设置，或者为OFF，则指定版本不可用时，会使用上一版本。
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 指定为C++14 版本
set(CMAKE_CXX_STANDARD 14)

# 程序中有使用这个宏定义
add_definitions(-DCOMPILEDWITHC11)
############################################################
# Cmake Moudule Path
############################################################
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
message(${CMAKE_MODULE_PATH})

############################################################
# 寻找库并包含路径
############################################################
# OpenCV
set(OpenCV_DIR "/home/divenire/Divenire_ws/libsStore/OPENCV/4.2.0/opencv-4.2.0/build")
#set(OpenCV_DIR "/home/divenire/Divenire_ws/libsStore/OPENCV/3.4.12/opencv-3.4.12/build")
find_package(OpenCV  REQUIRED)

# if(NOT OpenCV_FOUND)
#    find_package(OpenCV 2.4.3 QUIET)
#    if(NOT OpenCV_FOUND)
#       message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
#    endif()
# endif()

#Eigen
# include_directories(/home/divenire/Divenire_ws/libsStore/EIGEN/eigen-3.2.1/eigen-3.2.1)
include_directories(/home/divenire/Divenire_ws/libsStore/EIGEN/eigen-3.3.8)

#PANGOLIN
#set(Pangolin_DIR "/home/divenire/myDisk/WorkSpace/LIB_store/Pangolin/build/src")
find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})



include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
)

############################################################
# 设置库文件输出路径，生成库文件
############################################################
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
src/System.cc
src/Tracking.cc
src/LocalMapping.cc
src/LoopClosing.cc
src/ORBextractor.cc
src/ORBmatcher.cc
src/FrameDrawer.cc
src/Converter.cc
src/MapPoint.cc
src/KeyFrame.cc
src/Map.cc
src/MapDrawer.cc
src/Optimizer.cc
src/PnPsolver.cc
src/Frame.cc
src/KeyFrameDatabase.cc
src/Sim3Solver.cc
src/Initializer.cc
src/Viewer.cc
)

target_link_libraries(${PROJECT_NAME}
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

############################################################
# 输出可执行文件
############################################################
## 二进制文件输出路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(rgbd_tum
Examples/RGB-D/rgbd_tum.cc)
target_link_libraries(rgbd_tum ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(stereo_kitti
Examples/Stereo/stereo_kitti.cc)
target_link_libraries(stereo_kitti ${PROJECT_NAME})

add_executable(stereo_euroc
Examples/Stereo/stereo_euroc.cc)
target_link_libraries(stereo_euroc ${PROJECT_NAME})


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)

add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
target_link_libraries(mono_tum ${PROJECT_NAME})

add_executable(mono_kitti
Examples/Monocular/mono_kitti.cc)
target_link_libraries(mono_kitti ${PROJECT_NAME})

add_executable(mono_euroc
Examples/Monocular/mono_euroc.cc)
target_link_libraries(mono_euroc ${PROJECT_NAME})

# Self example
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/myvideo)
add_executable(myvideo Examples/myvideo/myvideo.cpp)
target_link_libraries(myvideo ${PROJECT_NAME})

add_executable(myslam Examples/myvideo/myslam.cpp)
target_link_libraries(myslam ${PROJECT_NAME})

# D435i
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Sensors)
add_executable(mono_d435i Examples/Sensors/mono_d435i.cc)
target_link_libraries(mono_d435i ${PROJECT_NAME})
