#ifndef __OPTIMIER_H__
#define __OPTIMIER_H__

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include "define.h"

void two_view_ba(Frame &frame_last, Frame &frame_curr, LoaclMap &map, std::vector<FMatch> &matches, int n_iter = 20);

#endif //__OPTIMIER_H__