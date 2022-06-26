/*
 * @Author: Divenire qiuyue_online@163.com
 * @Date: 2022-06-22 20:03:43
 * @LastEditors: Divenire qiuyue_online@163.com
 * @LastEditTime: 2022-06-22 21:59:37
 * @FilePath: /hw1/src/two_view_geometry.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef __TWO_VIEW_GEOMETRY__
#define __TWO_VIEW_GEOMETRY__

#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <opencv2/highgui.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>
#include <opencv2/core/eigen.hpp>

namespace TwoViewGeometry
{

    using namespace std;

    void FindFundamental(vector<cv::KeyPoint> mvKeys1, vector<cv::KeyPoint> mvKeys2,
                         vector<bool> &vbInliers, float &score, cv::Mat &F21);

    bool ReconstructF(vector<cv::KeyPoint> mvKeys1, vector<cv::KeyPoint> mvKeys2,
                      vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated);

}

#endif