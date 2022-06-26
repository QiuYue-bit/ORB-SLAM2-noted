#ifndef __TWO_VIEW_GEOMETRY__
#define __TWO_VIEW_GEOMETRY__

#include <vector>
#include <opencv2/opencv.hpp>
#include<opencv2/features2d/features2d.hpp>

namespace TwoViewGeometry {

    using namespace std;

    void FindFundamental(const vector<cv::Point2f>& vPts1, const vector<cv::Point2f>& vPts2, const vector<cv::DMatch> &vMatches12, vector<bool> &vbMatchesInliers,
                         cv::Mat &F21, float &score, float sigma = 1.0f, int maxIterations = 200);

    bool ReconstructF(const vector<cv::Point2f>& vPts1, const vector<cv::Point2f>& vPts2, const vector<cv::DMatch> &vMatches12, vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K, cv::Mat &R21, cv::Mat &t21,
                      vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax = 0.0f, int minTriangulated = 50, float sigma = 1.0f);

    void Triangulate(const cv::Point2f &pt1, const cv::Point2f &pt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
}


#endif