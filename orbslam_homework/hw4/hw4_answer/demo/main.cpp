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
#include "two_view_geometry.h"
#include "optimizer.h"

/**
 * @description: 将三维地图点投影到图像中生成特征点
 * @param {Matrix4d} &Twc
 * @param {Matrix3d} &K
 * @param {vector<Eigen::Vector3d>} &landmarks
 * @param {vector<Eigen::Vector3d>} &normals
 * @param {  } std
 * @param {vector<int32_t>} &matched
 * @param {bool} add_noise
 * @return {*}
 */
void detectFeatures(const Eigen::Matrix4d &Twc, const Eigen::Matrix3d &K,
                    const std::vector<Eigen::Vector3d> &landmarks,
                    const std::vector<Eigen::Vector3d> &normals,
                    std::vector<Eigen::Vector2i> &features,
                    std::vector<int32_t> &matched, bool add_noise = true)
{
    assert(landmarks.size() == normals.size());

    // 噪声水平
    std::mt19937 gen{12345};
    const float pixel_sigma = 1.0;
    std::normal_distribution<> d{0.0, pixel_sigma};

    // 当前帧位姿
    Eigen::Matrix3d Rwc = Twc.block(0, 0, 3, 3);
    Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;

    // 默认都不匹配上
    features.resize(landmarks.size());
    matched = std::vector<int32_t>(landmarks.size(), -1);

    for (size_t l = 0; l < landmarks.size(); ++l)
    {
        matched[l] = -1;

        // 地图点
        Eigen::Vector3d wP = landmarks[l];
        Eigen::Vector3d cP = Rcw * wP + tcw;

        // 深度值检验
        if (cP[2] < 0)
            continue;

#if 1
        /* check landmark in sight by mesh normal */
        // t_lc_w
        Eigen::Vector3d obv_dir = twc - landmarks[l];
        // 观测方向与物体发现夹角应该小于60度
        double costheta = obv_dir.dot(normals[l]) / (obv_dir.norm() * normals[l].norm());
        if (costheta < 0.5)
            continue;
#endif

        float noise_u = add_noise ? std::round(d(gen)) : 0.0f;
        float noise_v = add_noise ? std::round(d(gen)) : 0.0f;

        Eigen::Vector3d ft = K * cP;
        int u = ft[0] / ft[2] + 0.5 + noise_u;
        int v = ft[1] / ft[2] + 0.5 + noise_v;
        Eigen::Vector2i obs(u, v);

        features[l] = obs;
        matched[l] = l;
    }
}

std::vector<FMatch> feature_match(const Frame &frame1, const Frame &frame2, float outlier_rate = 0.0f)
{
    const std::vector<int32_t> &obvs1 = frame1.fts_obvs_;
    const std::vector<int32_t> &obvs2 = frame2.fts_obvs_;

    std::map<int32_t, int32_t> obvs_map;

    // 对当前帧的每一个特征点
    for (size_t n = 0; n < obvs1.size(); n++)
    {
        // 这个地图点无法被观察到
        if (obvs1[n] < 0)
            continue;

        // 地图点-特征的匹配关系
        obvs_map.emplace(obvs1[n], n);
    }

    //
    std::vector<FMatch> matches;
    matches.reserve(obvs_map.size());

    // 当前帧的特征点在上一帧寻找匹配关系
    for (size_t n = 0; n < obvs2.size(); n++)
    {
        // 如果这个特征点无法被上一帧观测到
        if (obvs2[n] < 0)
            continue;

        // 当前帧无法看到这个地图点，跳过
        if (!obvs_map.count(obvs2[n]))
            continue;

        // 建立两帧之间的匹配关系
        // featrue id - feature id
        int32_t idx1 = obvs_map[obvs2[n]];
        int32_t idx2 = n;
        matches.emplace_back(idx1, idx2, false);
    }

    // if add outliers
    size_t outlier_num = outlier_rate * matches.size();
    static std::mt19937 gen{12345};
    std::uniform_int_distribution<> d{0, matches.size() - 1};

    std::set<size_t> fts_set;
    for (size_t i = 0; i < outlier_num; i++)
    {
        size_t id1 = d(gen);
        // 防止重复添加
        if (fts_set.count(id1))
        {
            continue;
        }
        fts_set.insert(id1);

        size_t id2;
        do
        {
            id2 = d(gen);
        } while (obvs2[id2] < 0 || id2 == matches[id1].second);

        // 随机生成错误的匹配关系
        matches[id1].second = id2;
        matches[id1].outlier_gt = true;
    }

    return matches;
}

void outlier_rejection(Frame &frame_last, Frame &frame_curr, LoaclMap &map, std::vector<FMatch> &matches)
{
    const double sigma = 1.0;
    const double thr = sigma * 4;
    double rpj_err;
    for (size_t n = 0; n < matches.size(); n++)
    {
        uint32_t idx_curr = matches[n].first;
        uint32_t idx_last = matches[n].second;
        int32_t mpt_idx = frame_last.mpt_track_[idx_last];

        if (mpt_idx < 0)
        {
            continue;
        }
        Eigen::Vector3d &mpt = map.mpts_[mpt_idx];
        // TODO homework 重投影误差剔除外点
        Eigen::Matrix4d T_cw = frame_curr.Twc_.inverse();

        Eigen::Vector3d c2_P = frame_curr.K_ * (T_cw.block(0, 0, 3, 3) * mpt + T_cw.block(0, 3, 3, 1));
        Eigen::Vector2d pre = {c2_P[0] / c2_P[2], c2_P[1] / c2_P[2]};

        Eigen::Vector2d obs;
        obs[0] = frame_curr.fts_[idx_curr][0];
        obs[1] = frame_curr.fts_[idx_curr][1];

        rpj_err = (pre - obs).norm();

        if (rpj_err > thr)
        {
            matches[n].outlier = true;
        }
    }

    for (size_t n = 0; n < matches.size(); n++)
    {
        if (matches[n].outlier)
        {
            uint32_t idx_curr = matches[n].first;
            frame_curr.mpt_track_[idx_curr] = -1;
        }
    }
}

bool createInitMap(Frame &frame_last, Frame &frame_curr, LoaclMap &map, std::vector<FMatch> &matches)
{
    static int i = 0;
    i++;

    cv::Mat cv_K;
    Eigen::Matrix3f temp_K = frame_last.K_.cast<float>();
    cv::eigen2cv(temp_K, cv_K);

    cv::Mat R21, t21;

    /* 获取两帧之间的匹配关系 */
    std::vector<cv::Point2f> point_curr;
    std::vector<cv::Point2f> point_last;
    point_curr.reserve(matches.size());
    point_last.reserve(matches.size());
    for (size_t n = 0; n < matches.size(); n++)
    {
        uint32_t idx_curr = matches[n].first;
        uint32_t idx_last = matches[n].second;
        point_curr.push_back(cv::Point2f(frame_curr.fts_[idx_curr][0], frame_curr.fts_[idx_curr][1]));
        point_last.push_back(cv::Point2f(frame_last.fts_[idx_last][0], frame_last.fts_[idx_last][1]));
    }

    // OPENCV恢复位姿
    std::vector<bool> vbMatchesInliers;
    Eigen::Matrix4d T_curr_last;

    cv::Mat inlier_mask;
    cv::Mat cv_E = cv::findEssentialMat(point_last, point_curr, cv_K, cv::RANSAC, 0.999, 1.0, inlier_mask);

    int num_inlier_F = cv::countNonZero(inlier_mask);
    printf("%d - F inlier: %d(%d)\n", i, num_inlier_F, (int)point_curr.size());

    Eigen::Matrix3d R;
    Eigen::Vector3d t;

    // 计算出来的是
    // Tc2c1
    // point_curr = Tc2c1 * point_last
    cv::recoverPose(cv_E, point_last, point_curr, cv_K, R21, t21, inlier_mask);

    int num_inlier_Pose = cv::countNonZero(inlier_mask);
    printf("%d - RT inlier: %d(%d)\n", i, num_inlier_Pose, num_inlier_F);
    vbMatchesInliers.resize(point_curr.size());
    for (size_t n = 0; n < point_curr.size(); n++)
    {
        vbMatchesInliers[n] = inlier_mask.at<uint8_t>(n) != 0;
    }
    cv::cv2eigen(R21, R);
    cv::cv2eigen(t21, t);

    /* comput Twc_cur = Twc_last * T_last_curr */
    T_curr_last = Eigen::Matrix4d::Identity();
    T_curr_last.block(0, 0, 3, 3) = R;
    T_curr_last.block(0, 3, 3, 1) = t.normalized();

    /* 根据恢复的位姿三角化得到地图点 */
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));

    // P2 = K [R_c2c1 | t_c2c1]
    cv_K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    cv::Mat P2(3, 4, CV_32F);
    R21.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    t21.copyTo(P2.rowRange(0, 3).col(3));
    P2 = cv_K * P2;

    map.clear();

    // 记录每一帧的地图点对应关系
    frame_last.mpt_track_ = std::vector<int32_t>(frame_last.N_, -1);
    frame_curr.mpt_track_ = std::vector<int32_t>(frame_curr.N_, -1);

    // 对于每一对匹配
    for (size_t n = 0; n < matches.size(); n++)
    {

        //
        matches[n].outlier = true;

        // 错误的匹配不创建三维点
        if (!vbMatchesInliers[n])
        {
            continue;
        }

        /* if fts in last frame has been matched, then excaped */
        uint32_t idx_curr = matches[n].first;
        uint32_t idx_last = matches[n].second;

        // TODO 我觉得这句话注释掉也没事
        if (frame_last.mpt_track_[idx_last] > 0)
        {
            continue;
        }

        cv::Mat p3d_c1;
        // 三角化地图点，在C1坐标系下
        TwoViewGeometry::Triangulate(point_last[n], point_curr[n], P1, P2, p3d_c1);

        // 地图点有效性
        if (!std::isfinite(p3d_c1.at<float>(0)) ||
            !std::isfinite(p3d_c1.at<float>(1)) ||
            !std::isfinite(p3d_c1.at<float>(2)))
        {
            continue;
        }

        // 记录匹配关系
        // 保存地图点
        Eigen::Vector3d mpt;
        mpt[0] = p3d_c1.at<float>(0);
        mpt[1] = p3d_c1.at<float>(1);
        mpt[2] = p3d_c1.at<float>(2);
        int32_t mpt_idx = map.add(mpt);
        frame_curr.mpt_track_[idx_curr] = mpt_idx;
        frame_last.mpt_track_[idx_last] = mpt_idx;
        matches[n].outlier = false;
    }

    // 第一帧得位姿设置为单位矩阵
    frame_last.Twc_ = Eigen::Matrix4d::Identity();

    frame_curr.Twc_ = frame_last.Twc_ * T_curr_last.inverse();

    // 剔除外点，BA，再剔除外点
    outlier_rejection(frame_last, frame_curr, map, matches);
    two_view_ba(frame_last, frame_curr, map, matches);
    outlier_rejection(frame_last, frame_curr, map, matches);

    return true;
}

int main()
{
    //
    cv::viz::Viz3d window("window");
    cv::viz::WCoordinateSystem world_coord(1.0), camera_coord(0.5);
    window.showWidget("Coordinate", world_coord);
    window.showWidget("Camera", camera_coord);

    cv::viz::Mesh mesh;
    loadMeshAndPerprocess(mesh, "/data/bun_zipper_res3.ply");

    // window.showWidget("mesh", cv::viz::WMesh(mesh));
    // window.showWidget("normals", cv::viz::WCloudNormals(mesh.cloud, mesh.normals, 1, 1.0f, cv::viz::Color::green()));
    // window.setRenderingProperty("normals", cv::viz::LINE_WIDTH, 2.0);

    std::cout << mesh.normals.size() << std::endl;
    std::cout << mesh.normals.type() << std::endl;
    // std::cout << CV_32FC3 << std::endl;
    // std::cout << CV_64FC3 << std::endl;

    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Vector3d> normals;
    std::vector<Eigen::Matrix4d> v_Twc;

    // 从mesh中生成路标点
    createLandmarksFromMesh(mesh, landmarks, normals);

    // 创建相机的位姿真值
    createCameraPose(v_Twc, Eigen::Vector3d(0, 0, 0));

    const size_t pose_num = v_Twc.size();

    /* 在VIZ中显示点云真值 */
    {
        cv::Mat point_cloud = cv::Mat::zeros(landmarks.size(), 1, CV_32FC3);
        for (size_t i = 0; i < landmarks.size(); i++)
        {
            point_cloud.at<cv::Vec3f>(i)[0] = landmarks[i][0];
            point_cloud.at<cv::Vec3f>(i)[1] = landmarks[i][1];
            point_cloud.at<cv::Vec3f>(i)[2] = landmarks[i][2];
        }
        cv::viz::WCloud cloud(point_cloud);
        window.showWidget("__cloud", cloud);
        // 设置点云的大小
        window.setRenderingProperty("__cloud", cv::viz::POINT_SIZE, 3.0);
    }

#if 1
    /*  VIZ显示相机轨迹真值 */
    for (size_t i = 0; i < pose_num - 1; i++)
    {
        Eigen::Vector3d twc0 = v_Twc[i].block(0, 3, 3, 1);
        Eigen::Vector3d twc1 = v_Twc[i + 1].block(0, 3, 3, 1);
        cv::Point3d pose_begin(twc0[0], twc0[1], twc0[2]);
        cv::Point3d pose_end(twc1[0], twc1[1], twc1[2]);
        cv::viz::WLine trag_line(pose_begin, pose_end, cv::viz::Color::green());
        window.showWidget("gt_trag_" + std::to_string(i), trag_line);
    }

    static const Eigen::Vector3d cam_z_dir(0, 0, 1);
    for (size_t i = 0; i < pose_num; i++)
    {
        Eigen::Matrix3d Rwc = v_Twc[i].block(0, 0, 3, 3);
        Eigen::Vector3d twc = v_Twc[i].block(0, 3, 3, 1);
        Eigen::Vector3d w_cam_z_dir = Rwc * cam_z_dir;
        cv::Point3d obvs_dir(w_cam_z_dir[0], w_cam_z_dir[1], w_cam_z_dir[2]);
        cv::Point3d obvs_begin(twc[0], twc[1], twc[2]);
        cv::viz::WLine obvs_line(obvs_begin, obvs_begin + obvs_dir, cv::viz::Color::blue());
        window.showWidget("gt_cam_z_" + std::to_string(i), obvs_line);
    }
#endif

    // 相机内参
    cv::Mat cv_K = (cv::Mat_<double>(3, 3) << 480, 0, 320, 0, 480, 240, 0, 0, 1);
    // cv::Mat cv_K = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 480, 240, 0, 0, 1);
    Eigen::Matrix3d K;
    cv::cv2eigen(cv_K, K);

    std::vector<Frame, Eigen::aligned_allocator<Frame>> frames;
    frames.reserve(v_Twc.size());

    // 新建一个帧并进行特征检测
    // 三维点投影到相机平面上
    // 关键帧的特征点存储在fts_中
    // fts_obvs_存储特征点的观测情况
    frames.emplace_back(Frame(0, landmarks.size(), K));
    frames.back().Twc_ = v_Twc[0];
    detectFeatures(v_Twc[0], K, landmarks, normals, frames.back().fts_, frames.back().fts_obvs_);

    /* start odomerty */
    std::vector<Eigen::Matrix4d> pose_est;
    std::vector<Eigen::Matrix4d> pose_gt;
    pose_est.reserve(v_Twc.size());
    pose_gt.reserve(v_Twc.size());

    // 第一帧位姿用真值
    pose_est.push_back(v_Twc[0]);
    pose_gt.push_back(v_Twc[0]);

    bool init_flag = false;

    // 创建初始地图
    LoaclMap map(landmarks.size());

    // 开始对每一帧进行处理
    for (size_t i = 1; i < pose_num; i++)
    {
        /* get features of current frame */
        Frame &frame_last = frames.back();
        Frame frame_curr(i, landmarks.size(), K);

        // 三维点投影到相机平面上
        // 关键帧的特征点存储在fts_中
        // fts_obvs_存储地图点的观测情况
        detectFeatures(v_Twc[i], K, landmarks, normals, frame_curr.fts_, frame_curr.fts_obvs_);

        /* get matched features */
        frame_curr.mpt_track_ = std::vector<int32_t>(landmarks.size(), -1);

        // 特征匹配
        std::vector<FMatch> matches = feature_match(frame_curr, frame_last, 0.05);
        assert(!matches.empty());

        // 帧的ID号 -- 匹配点对数量--
        std::cout << "[" << std::setw(3) << frame_curr.idx_ << "] match features " << matches.size() << std::endl;

        Eigen::Matrix4d Twc_curr;

        /* 创建初始地图 */
        if (!init_flag)
        {
            bool is_success = false;

            // * 两视图几何创建初始地图,得到初始地图点
            // 并进行BA和外点剔除
            is_success = createInitMap(frame_last, frame_curr, map, matches);
            if (!is_success)
            {
                continue;
            }
            init_flag = is_success;

            //! frame_last.Twc_ set to Identity in init
            Eigen::Matrix4d T_curr_last = frame_curr.Twc_.inverse() * frame_last.Twc_;
            frame_last.Twc_ = v_Twc[0];
            frame_curr.Twc_ = frame_last.Twc_ * T_curr_last.inverse();

            double t_scale = 1.0;
            {
                // 使用真值地图的两帧位移作为尺度
#if 0
                Eigen::Matrix4d gt_Twc_last = frame_last.Twc_;
                Eigen::Matrix4d gt_Twc_curr = v_Twc[i];
                Eigen::Matrix4d gt_T_cur_last = gt_Twc_curr.inverse() * gt_Twc_last;
                t_scale = gt_T_cur_last.block(0, 3, 3, 1).norm();

                //
#else
                double gt_dist = 0.0;
                Eigen::Matrix4d last_Tcw = frame_last.Twc_.inverse();
                for (size_t n = 0; n < landmarks.size(); n++)
                {
                    int32_t mpt_idx = frame_curr.mpt_track_[n];
                    if (mpt_idx < 0)
                        continue;

                    // 真实的3D路标点
                    Eigen::Vector3d mpt = landmarks[n];
                    // 转换到第一帧相机坐标系下
                    mpt = last_Tcw.block(0, 0, 3, 3) * mpt + last_Tcw.block(0, 3, 3, 1);
                    gt_dist += mpt[2];
                }

                double est_dist = 0.0;
                for (size_t n = 0; n < landmarks.size(); n++)
                {
                    int32_t mpt_idx = frame_curr.mpt_track_[n];
                    if (mpt_idx < 0)
                        continue;

                    // 三角化得到的无尺度信息的地图点（现在是在第一帧的相机坐标系下的）
                    Eigen::Vector3d mpt = map.mpts_[mpt_idx];
                    est_dist += mpt[2];
                }

                // 计算尺度
                t_scale = gt_dist / est_dist;
#endif
            }

            // 位姿恢复尺度
            T_curr_last.block(0, 3, 3, 1) *= t_scale;

            // Twc2 = Twc1 * Tc2c1.tanspose()
            frame_curr.Twc_ = frame_last.Twc_ * T_curr_last.inverse();

            // 地图点恢复尺度
            // 转回世界坐标系下
            int mpts_count = 0;
            for (size_t n = 0; n < landmarks.size(); n++)
            {
                int32_t mpt_idx = frame_curr.mpt_track_[n];
                if (mpt_idx < 0)
                    continue;
                Eigen::Vector3d &mpt = map.mpts_[mpt_idx];
                mpt *= t_scale;
                mpt = frame_last.Twc_.block(0, 0, 3, 3) * mpt + frame_last.Twc_.block(0, 3, 3, 1);
                mpts_count++;
            }

            // 输出初始化帧的ID，以及创建了多少个地图点
            std::cout << "[" << std::setw(3) << frame_curr.idx_ << "] create init mappoints " << mpts_count << std::endl;
        }
        /* 创建好了初始化帧 */
        else
        {

            // 当前帧跟踪到的地图点
            std::vector<cv::Point3f> obj_pts;
            std::vector<cv::Point2f> img_pts;

            // 得到当前帧得3D-2D对应关系
            for (size_t n = 0; n < matches.size(); n++)
            {
                uint32_t idx_curr = matches[n].first;
                uint32_t idx_last = matches[n].second;

                // 上一帧特征点对应的地图点
                int32_t mpt_idx = frame_last.mpt_track_[idx_last];

                if (mpt_idx < 0)
                {
                    continue;
                }
                Eigen::Vector3d &mpt = map.mpts_[mpt_idx];
                obj_pts.emplace_back(mpt[0], mpt[1], mpt[2]);
                img_pts.push_back(cv::Point2f(frame_curr.fts_[idx_curr][0], frame_curr.fts_[idx_curr][1]));
            }

            // 当前帧跟踪到的地图点数量
            std::cout << "[" << std::setw(3) << frame_curr.idx_ << "] track landmarks " << obj_pts.size() << std::endl;

            // 地图点越跟越少了
            if (obj_pts.size() < 5)
            {
                std::cout << "!!!!!!!!!!! track lost !!!!!!!!!!!!" << std::endl;
                window.spin();
            }

            // OpenCV求解PNP问题得到当前帧得位姿
            // R_cw
            // t_cw
            cv::Mat rvec, tvec;
            cv::solvePnPRansac(obj_pts, img_pts, cv_K, cv::Mat(), rvec, tvec, false,
                               100, 8.0, 0.99, cv::noArray(), cv::SOLVEPNP_ITERATIVE);

            cv::Mat Rmat(3, 3, CV_64FC1);
            Rodrigues(rvec, Rmat);
            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            cv::cv2eigen(Rmat, R);
            cv::cv2eigen(tvec, t);
            frame_curr.Twc_.block(0, 0, 3, 3) = R.transpose();
            frame_curr.Twc_.block(0, 3, 3, 1) = -R.transpose() * t;
            using namespace std;
            /* create new mappoints */
            // TODO homework 根据当前帧的位姿计算投影矩阵
            // P = K[R_cw|T_cw]
            // T_c2c1 = Tc2w * Twc1
            auto T_c2c1 = frame_curr.Twc_.inverse() * frame_last.Twc_;
            // cout<<"T_c1c2 cal :"<<T_c2c1<<endl;
            // cout << "T_c1c2 true: " << v_Twc[i].inverse() * v_Twc[i - 1] << endl;

            Eigen::Matrix3d R_c2c1 = T_c2c1.block(0, 0, 3, 3);
            Eigen::Vector3d t_c2c1 = T_c2c1.block(0, 3, 3, 1);

            cv::Mat cv_R;
            cv::Mat cv_t;
            cv::eigen2cv(R_c2c1, cv_R);
            cv::eigen2cv(t_c2c1, cv_t);
            // cout << "Eigen R is " << R_c2c1 << endl;
            // cout << "cv_R is " << cv_R << endl;
            // cout << "Eigen t is " << t_c2c1 << endl;
            // cout << "cv_t is " << cv_t.t() << endl;

            // P_c1c1 K[I,0]
            cv::Mat cv_P1(3, 4, CV_32F, cv::Scalar(0));
            cv_K.copyTo(cv_P1.rowRange(0, 3).colRange(0, 3));

            // P_c2c1 K[R,T]
            cv::Mat cv_P2(3, 4, CV_32F);
            cv_R.copyTo(cv_P2.rowRange(0, 3).colRange(0, 3));
            cv_t.copyTo(cv_P2.rowRange(0, 3).col(3));
            // !易错点，矩阵数据格式调整，需要为32F
            cv::Mat cv_K1;
            Eigen::Matrix3f temp_K = frame_last.K_.cast<float>();
            cv::eigen2cv(temp_K, cv_K1);
            cv_P2 = cv_K1 * cv_P2;

            // cout<<"cv_P1"<<cv_P1<<endl;
            // cout<<"cv_P2"<<cv_P2<<endl;

            // 对所有 有匹配关系的特征点，但是在当前帧没有对应的地图点
            // 进行三角化
            for (size_t n = 0; n < matches.size(); n++)
            {
                if (matches[n].outlier)
                {
                    //
                    continue;
                }

                uint32_t idx_curr = matches[n].first;
                uint32_t idx_last = matches[n].second;
                int32_t mpt_idx = -1;

                // 如果当前帧的特征点在上一帧有对应的地图点
                // 直接用上一帧的地图点
                if (frame_last.mpt_track_[idx_last] >= 0)
                {
                    // use last frame's mpts, not re-triangulate
                    // 取出上一帧特征对应的地图点id
                    // 设置为当前帧的地图点
                    mpt_idx = frame_last.mpt_track_[idx_last];
                    frame_curr.mpt_track_[idx_curr] = mpt_idx;
                    continue;
                }

                cv::Mat p3d_c1;

                // TODO homework 根据投影矩阵进行三角化地图点
                // 得到的三角化点实在上一帧坐标系下的
                cv::Point2f feature_lastframe, feature_curframe;
                feature_curframe = cv::Point2f(frame_curr.fts_[idx_curr][0], frame_curr.fts_[idx_curr][1]);
                feature_lastframe = cv::Point2f(frame_last.fts_[idx_last][0], frame_last.fts_[idx_last][1]);
                // cout<<feature_lastframe<<" "<<feature_curframe<<endl;
                TwoViewGeometry::Triangulate(feature_lastframe, feature_curframe, cv_P1, cv_P2, p3d_c1);

                // 调好以后注释掉continue
                // continue;

                if (!std::isfinite(p3d_c1.at<float>(0)) ||
                    !std::isfinite(p3d_c1.at<float>(1)) ||
                    !std::isfinite(p3d_c1.at<float>(2)))
                {
                    cout << "this is outlier" << endl;
                    frame_curr.mpt_track_[idx_curr] = -1;
                    matches[n].outlier = true;
                    continue;
                }

                // 把上一帧坐标系下的地图点恢复到3维空间中
                Eigen::Vector3d mpt = Eigen::Vector3f(p3d_c1.at<float>(0), p3d_c1.at<float>(1), p3d_c1.at<float>(2)).cast<double>();
                mpt = frame_last.Twc_.block(0, 0, 3, 3) * mpt + frame_last.Twc_.block(0, 3, 3, 1);
                mpt_idx = map.add(mpt);

                frame_curr.mpt_track_[idx_curr] = mpt_idx;
                frame_last.mpt_track_[idx_last] = mpt_idx;
            }
            outlier_rejection(frame_last, frame_curr, map, matches);

            std::cout << "[" << std::setw(3) << frame_curr.idx_ << "] inlier landmarks "
                      << std::count_if(frame_curr.mpt_track_.begin(), frame_curr.mpt_track_.end(), [](int32_t idx)
                                       { return idx >= 0; })
                      << std::endl;

            two_view_ba(frame_last, frame_curr, map, matches);
        }

        /* show landmarks in sight */
        {
            cv::Mat point_cloud = cv::Mat::zeros(landmarks.size(), 1, CV_32FC3);
            for (size_t k = 0; k < landmarks.size(); k++)
            {
                if (frame_curr.fts_obvs_[k] < 0)
                {
                    continue;
                }
                point_cloud.at<cv::Vec3f>(k)[0] = landmarks[k][0];
                point_cloud.at<cv::Vec3f>(k)[1] = landmarks[k][1];
                point_cloud.at<cv::Vec3f>(k)[2] = landmarks[k][2];
            }
            cv::viz::WCloud cloud(point_cloud, cv::viz::Color::yellow());
            window.showWidget("cloud", cloud);
            window.setRenderingProperty("cloud", cv::viz::POINT_SIZE, 3.0);
        }

        /* show estimated mappoints */
        {
            int N = std::count(map.status_.begin(), map.status_.end(), true);
            cv::Mat point_cloud = cv::Mat::zeros(N, 1, CV_32FC3);
            for (size_t k = 0, n = 0; k < landmarks.size(); k++)
            {
                int32_t mpt_idx = frame_curr.mpt_track_[k];
                if (mpt_idx < 0)
                {
                    continue;
                }
                point_cloud.at<cv::Vec3f>(n)[0] = map.mpts_[mpt_idx][0];
                point_cloud.at<cv::Vec3f>(n)[1] = map.mpts_[mpt_idx][1];
                point_cloud.at<cv::Vec3f>(n)[2] = map.mpts_[mpt_idx][2];
                n++;
            }
            cv::viz::WCloud cloud(point_cloud, cv::viz::Color::orange());
            // window.removeWidget("mappoints");
            window.showWidget("mappoints", cloud);
        }

        /* show estimated trajectory */
        {
            Eigen::Vector3d twc0 = frame_last.Twc_.block(0, 3, 3, 1);
            Eigen::Vector3d twc1 = frame_curr.Twc_.block(0, 3, 3, 1);
            cv::Point3d pose_begin(twc0[0], twc0[1], twc0[2]);
            cv::Point3d pose_end(twc1[0], twc1[1], twc1[2]);
            cv::viz::WLine trag_line(pose_begin, pose_end, cv::viz::Color(0, 255, 255));
            window.showWidget("trag_" + std::to_string(i), trag_line);

            Eigen::Matrix3d Rwc1 = frame_curr.Twc_.block(0, 0, 3, 3);
            Eigen::Vector3d w_cam_z_dir = Rwc1 * Eigen::Vector3d(0, 0, 1);
            cv::Point3d obvs_dir(w_cam_z_dir[0], w_cam_z_dir[1], w_cam_z_dir[2]);
            cv::Point3d obvs_begin(twc1[0], twc1[1], twc1[2]);
            cv::viz::WLine obvs_line(obvs_begin, obvs_begin + obvs_dir, cv::viz::Color(255, 0, 100));
            window.showWidget("cam_z_" + std::to_string(i), obvs_line);
        }

        /* update */
        map.update(frame_curr);
        frames.push_back(frame_curr);
        pose_est.push_back(frame_curr.Twc_);
        pose_gt.push_back(v_Twc[i]);
        window.spinOnce(10, true);
        // window.spin();
    }

    /* save trajectory for evalution */
    saveTrajectoryTUM("../results/frame_traj_gt.txt", pose_gt);
    saveTrajectoryTUM("../results/frame_traj_est.txt", pose_est);

    while (!window.wasStopped())
    {
        window.spinOnce(1, true);
    }
}