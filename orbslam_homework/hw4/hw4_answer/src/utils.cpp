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

#include "define.h"
#include "utils.h"

void saveTrajectoryTUM(const std::string &file_name, const std::vector<Eigen::Matrix4d> &v_Twc)
{
    std::ofstream f;
    f.open(file_name.c_str());
    f << std::fixed;

    for (size_t i = 0; i < v_Twc.size(); i++)
    {
        const Eigen::Matrix4d Twc = v_Twc[i];
        const Eigen::Vector3d t = Twc.block(0, 3, 3, 1);
        const Eigen::Matrix3d R = Twc.block(0, 0, 3, 3);
        const Eigen::Quaterniond q = Eigen::Quaterniond(R);

        f << std::setprecision(6) << i << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }
    f.close();
    std::cout << "save traj to " << file_name << " done." << std::endl;
}

void createLandmarks(std::vector<Eigen::Vector3d> &points)
{
    points.clear();

#if 0
    float scale = 5; 
    const double k = 0.5;
    for (float y = -1.0f; y <= 1.0f; y+=0.2)
    {
        Eigen::Vector3d pt;
        pt[0] = y > 0 ? k*y : -k*y;
        pt[1] = y;
        for (float z = -1.0f; z <= 1.0f; z+=0.2)
        {
            pt[2] = z;
            points.push_back(pt * scale);
        }
    }
#else

    std::mt19937 gen{12345};
    std::normal_distribution<double> d_x{0.0, 4.0};
    std::normal_distribution<double> d_y{0.0, 10.0};
    std::normal_distribution<double> d_z{0.0, 10.0};
    for (int i = 0; i < 200; i++)
    {
        Eigen::Vector3d pt;
        pt[0] = std::round(d_x(gen));
        pt[1] = std::round(d_y(gen));
        pt[2] = std::round(d_z(gen));
        points.push_back(pt);
    }

#endif
}

void loadMeshAndPerprocess(cv::viz::Mesh &mesh, const std::string &ply_file)
{
    mesh = cv::viz::Mesh::load(std::string(WORKSPACE_DIR) + ply_file);
    mesh.cloud *= 100;
    computeNormals(mesh, mesh.normals);

    const int N = mesh.cloud.cols;
    cv::Vec3f *p_cloud = mesh.cloud.ptr<cv::Vec3f>(0);
    // cv::Vec3f *p_normals = mesh.normals.ptr<cv::Vec3f>(0);

    cv::Vec3f min(FLT_MAX, FLT_MAX, FLT_MAX);
    cv::Vec3f max(FLT_MIN, FLT_MIN, FLT_MIN);
    std::cout << "max:" << max << std::endl
              << "min:" << min << std::endl;
    for (int n = 0; n < N; n++)
    {
        min[0] = min[0] > p_cloud[n][0] ? p_cloud[n][0] : min[0];
        min[1] = min[1] > p_cloud[n][1] ? p_cloud[n][1] : min[1];
        min[2] = min[2] > p_cloud[n][2] ? p_cloud[n][2] : min[2];

        max[0] = max[0] < p_cloud[n][0] ? p_cloud[n][0] : max[0];
        max[1] = max[1] < p_cloud[n][1] ? p_cloud[n][1] : max[1];
        max[2] = max[2] < p_cloud[n][2] ? p_cloud[n][2] : max[2];
    }

    std::cout << "max:" << max << std::endl
              << "min:" << min << std::endl;
    cv::Vec3f offset = 0.5 * (min + max);
    for (int n = 0; n < N; n++)
    {
        p_cloud[n] = p_cloud[n] - offset;
    }
}

void createLandmarksFromMesh(const cv::viz::Mesh &mesh, std::vector<Eigen::Vector3d> &points, std::vector<Eigen::Vector3d> &normals)
{
    points.clear();
    normals.clear();

    const int N = mesh.cloud.cols;

    cv::Mat _points;
    cv::Mat _normals;
    mesh.cloud.convertTo(_points, CV_64FC3);
    mesh.normals.convertTo(_normals, CV_64FC3);

    points.resize(N);
    normals.resize(N);

    _points.copyTo(cv::Mat(_points.size(), _points.type(), points.data()));
    _normals.copyTo(cv::Mat(_normals.size(), _normals.type(), normals.data()));
}

void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc, const Eigen::Vector3d &point_focus)
{
    float x_offset = 0;
    float y_offset = 0;
    float z_offset = 0;
    float scale = 30;


    // i的标记
    static int i = 0;

    static const Eigen::Vector3d b_cam_z(0, 0, 1);
    static const Eigen::Matrix3d R_w_base = (Eigen::Matrix3d() << 0, 0, -1, 1, 0, 0, 0, -1, 0).finished(); // row major
    // std::cout << (R_w_base * Eigen::Vector3d(0,0,1)).transpose() << std::endl;

    v_Twc.clear();
    float cycle = 2;
    for (float angle = -cycle * 360; angle < cycle * 360; angle += 15)
    {
        float theta = angle * 3.14159 / 180.0f;

        Eigen::Vector3d pt;
        pt[0] = cos(theta);
        pt[1] = sin(theta);
        pt[2] = theta / 30;

        pt = scale * pt;
        pt[0] += x_offset;
        pt[1] += y_offset;
        pt[2] += z_offset;

        Eigen::Vector3d b_cam_z_cur = R_w_base.transpose() * (point_focus - pt);
        Eigen::Matrix3d R_cur_base(Eigen::Quaterniond::FromTwoVectors(b_cam_z_cur, b_cam_z));
        // std::cout << pt.transpose() << ", " << (R_cur_base * b_cam_z_cur).transpose() << std::endl;
        Eigen::Matrix3d Rwc(R_w_base * R_cur_base.transpose());

        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = Rwc;
        Twc.block(0, 3, 3, 1) = pt;
        v_Twc.push_back(Twc);

        i++;
//        if(i>=2)
            // break;
    }
}