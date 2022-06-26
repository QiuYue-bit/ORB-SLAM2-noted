#ifndef __UTILS_H__
#define __UTILS_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

#include "Thirdparty/g2o/g2o/types/se3quat.h"

class Converter{
public:
    
    static g2o::SE3Quat toSE3Quat(const Eigen::Matrix4d &Twc)
    {
        Eigen::Matrix<double,3,3> R = Twc.block(0, 0, 3, 3);
        Eigen::Matrix<double,3,1> t = Twc.block(0, 3, 3, 1);

        return g2o::SE3Quat(R,t);
    }

    static Eigen::Matrix4d toEigenMat(const g2o::SE3Quat &SE3)
    {
        return SE3.to_homogeneous_matrix();
    }

};

#endif