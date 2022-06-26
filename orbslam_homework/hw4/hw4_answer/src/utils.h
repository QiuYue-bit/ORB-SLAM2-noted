#ifndef __UTILS_H__
#define __UTILS_H__

#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

void saveTrajectoryTUM(const std::string &file_name, const std::vector<Eigen::Matrix4d> &v_Twc);

void createLandmarks(std::vector<Eigen::Vector3d> &points);

void loadMeshAndPerprocess(cv::viz::Mesh &mesh, const std::string &ply_file);

void createLandmarksFromMesh(const cv::viz::Mesh &mesh, std::vector<Eigen::Vector3d> &points, std::vector<Eigen::Vector3d> &normals);

void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc, const Eigen::Vector3d &point_focus);

#endif