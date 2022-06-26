

#include "two_view_geometry.h"

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4d)

using namespace std;

// 轨迹保存为Tum格式
void saveTrajectoryTUM(const std::string &file_name, const std::vector<Eigen::Matrix4d> &v_Twc);

// 创建相机位姿
void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc);

// 特征检测
void detectFeatures(const Eigen::Matrix4d &Twc, const Eigen::Matrix3d &K,
                    const std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector2i> &features, bool add_noise = true);
// 创建地图点
void createLandmarks(std::vector<Eigen::Vector3d> &points);

// 匹配的点对
int pointPair = 0;

// 存储执行时间的容器
vector<std::chrono::duration<double>> time_costs;
std::chrono::steady_clock::time_point t1,t2;
int main()
{
    // 创建可视化窗口
    cv::viz::Viz3d window("window");
    cv::viz::WCoordinateSystem world_coord(3.0), camera_coord(1.0);

    // 构造坐标系显示到窗口中
    window.showWidget("Coordinate", world_coord);
    // window.showWidget("Camera", camera_coord);

    std::vector<Eigen::Vector3d> landmarks;
    std::vector<Eigen::Matrix4d> v_Twc;

    // 生成路标点
    createLandmarks(landmarks);

    // 创建相机位姿
    createCameraPose(v_Twc);

    // 一共有多少相机位姿
    const size_t pose_num = v_Twc.size();

    // 用于OPENCV显示的点云
    cv::Mat point_cloud = cv::Mat::zeros(landmarks.size(), 1, CV_32FC3);


#if 1
    /* show camera gt trajectory */
    // 显示相机的轨迹真值
    for (size_t i = 0; i < pose_num - 1; i++)
    {
        Eigen::Vector3d twc0 = v_Twc[i].block(0, 3, 3, 1);
        Eigen::Vector3d twc1 = v_Twc[i + 1].block(0, 3, 3, 1);
        cv::Point3d pose_begin(twc0[0], twc0[1], twc0[2]);
        cv::Point3d pose_end(twc1[0], twc1[1], twc1[2]);
        cv::viz::WLine trag_line(pose_begin, pose_end, cv::viz::Color::green());
        window.showWidget("gt_trag_" + std::to_string(i), trag_line);
    }

    // 绘制出相机的Z轴真值
    // static const Eigen::Vector3d cam_z_dir(0, 0, 1);
    // for (size_t i = 0; i < pose_num; i++)
    // {
    //     Eigen::Matrix3d Rwc = v_Twc[i].block(0, 0, 3, 3);
    //     Eigen::Vector3d twc = v_Twc[i].block(0, 3, 3, 1);
    //     Eigen::Vector3d w_cam_z_dir = Rwc * cam_z_dir;
    //     cv::Point3d obvs_dir(w_cam_z_dir[0], w_cam_z_dir[1], w_cam_z_dir[2]);
    //     cv::Point3d obvs_begin(twc[0], twc[1], twc[2]);
    //     cv::viz::WLine obvs_line(obvs_begin, obvs_begin + obvs_dir, cv::viz::Color::blue());
    //     window.showWidget("gt_cam_z_" + std::to_string(i), obvs_line);
    // }

#endif

    // 读取相机的内参数
    cv::Mat cv_K = (cv::Mat_<double>(3, 3) << 480, 0, 320, 0, 480, 240, 0, 0, 1);
    // cv::Mat cv_K = (cv::Mat_<double>(3, 3) << 1, 0, 1, 0, 480, 240, 0, 0, 1);
    Eigen::Matrix3d K;
    cv::cv2eigen(cv_K, K);

    // 将点云投影到相机上，得到观测值
    std::vector<cv::Point2f> point_last;
    std::vector<Eigen::Vector2i> features_curr;
    detectFeatures(v_Twc[0], K, landmarks, features_curr);
    for (size_t i = 0; i < features_curr.size(); i++)
    {
        point_last.push_back(cv::Point2f(features_curr[i][0], features_curr[i][1]));
    }

    /* start odomerty */
    std::vector<Eigen::Matrix4d> pose_est;
    pose_est.reserve(v_Twc.size());

    // 第一个相机位姿用真值
    Eigen::Matrix4d Twc_last = v_Twc[0];
    pose_est.push_back(Twc_last);

    for (size_t i = 1; i < pose_num; i++)
    {
        /* get scale form gt */
        double t_scale = 1.0;
        {
            Eigen::Matrix4d gt_Twc_last = v_Twc[i - 1];
            Eigen::Matrix4d gt_Twc_curr = v_Twc[i];
            Eigen::Matrix4d gt_T_cur_last = gt_Twc_curr.inverse() * gt_Twc_last;
            t_scale = gt_T_cur_last.block(0, 3, 3, 1).norm();
        }

        /* get features of current frame */
        detectFeatures(v_Twc[i], K, landmarks, features_curr);
        std::vector<cv::Point2f> point_curr;
        for (auto &i : features_curr)
        {
            point_curr.push_back(cv::Point2f(i[0], i[1]));
        }

        vector<cv::Point2f> feature_last;
        vector<cv::Point2f> features_curr;

        vector<cv::KeyPoint> KeyPoint_last;
        vector<cv::KeyPoint> KeyPoint_curr;


        // *剔除错误匹配
        for (int i = 0; i < landmarks.size() ; i++)
        {
            if (point_curr[i].x == 0 || point_last[i].x == 0)
                continue;
            else
            {
                feature_last.emplace_back(point_last[i]);
                features_curr.emplace_back(point_curr[i]);

                cv::KeyPoint kp_last;
                cv::KeyPoint kp_cur;
                kp_last.pt = point_last[i];
                kp_cur.pt = point_curr[i];
                KeyPoint_last.emplace_back(kp_last);
                KeyPoint_curr.emplace_back(kp_cur);
            }
        }


        Eigen::Matrix4d Twc_curr;

        // estimate fundamental matrix between frame i-1 and frame i, then recover pose from fundamental matrix
        {
            pointPair++;
            Eigen::Matrix3d F;
            Eigen::Matrix3d E;
            Eigen::Matrix3d R;
            Eigen::Vector3d t;

            //*计算两帧之间的基础矩阵
            cv::Mat inlier_mask;
            cv::Mat cv_F;
            float score;
            vector<bool> vbMatchesInliersF = vector<bool>(feature_last.size(), true) ;


            t1 = std::chrono::steady_clock::now();

            // *方案一 Opencv
            //  cv_F = cv::findFundamentalMat(feature_last, features_curr, cv::FM_RANSAC, 3, 0.99, inlier_mask);
            //  cv_F.convertTo(cv_F,CV_32F);

            // 方案二 ORBSLAM2
           TwoViewGeometry::FindFundamental(KeyPoint_last, KeyPoint_curr, vbMatchesInliersF, score, cv_F);
            // ORBSLAM返回的矩阵式CV_32,为了和OPENCV的计算位姿函数兼容，需要转换一下格式。
            // cv_F.convertTo(cv_F,CV_64F);

            t2 = std::chrono::steady_clock::now();
            auto time_cost = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1); 
            time_costs.emplace_back(time_cost);


            cout<<"cv_F type is "<<cv_F.type()<<endl;


            cv_K.convertTo(cv_K,CV_32F);

            // 从基础矩阵中恢复得本质矩阵
            cv::Mat cv_E = cv_K.t() * cv_F * cv_K;

            cv::Mat cv_R, cv_t;

            // *从本质矩阵中恢复位姿
            // *方案一 OPENCV
            // cv::recoverPose(cv_E, feature_last, features_curr, cv_K, cv_R, cv_t, inlier_mask);

            // *方案二 ORBSLAM2
            vector<cv::Point3f> vP3D;
            vector<bool> vbTriangulated;
            cout<<"hello"<<endl;
            bool flag = TwoViewGeometry::ReconstructF(KeyPoint_last,KeyPoint_curr,vbMatchesInliersF,cv_F,cv_K,cv_R,cv_t,vP3D,vbTriangulated,1,20);

            if(!flag)
            {
                //位姿恢复失败
                cout<<"ReconstructF fail"<<endl;
                continue;
            }


            cv::cv2eigen(cv_R, R);
            cv::cv2eigen(cv_t, t);

            // 恢复变换矩阵
            Eigen::Matrix4d T_curr_last = Eigen::Matrix4d::Identity();
            T_curr_last.block(0, 0, 3, 3) = R;
            T_curr_last.block(0, 3, 3, 1) = t * t_scale;
            // T_wc2 = T_wc1 * T_c1c2
            // T_curr_last = Tc2c1
            // 当前系的位姿
            Twc_curr = Twc_last * T_curr_last.inverse();

            // 测试位姿估计的效果
            {
                // word pos
                Eigen::Vector3d p_w = {111, 333, 444};

                // c1  true
                Eigen::Matrix3d R_wc1 = v_Twc[i - 1].block<3, 3>(0, 0);
                Eigen::Vector3d t_wc1 = v_Twc[i - 1].block<3, 1>(0, 3);
                Eigen::Matrix3d R_c1w = R_wc1.transpose();
                Eigen::Vector3d t_c1_w = -R_c1w * t_wc1;

                Eigen::Vector3d p_c1_true = R_c1w * p_w + t_c1_w;

                // c2 true
                Eigen::Matrix3d R_wc2 = v_Twc[i].block<3, 3>(0, 0);
                Eigen::Vector3d t_wc2 = v_Twc[i].block<3, 1>(0, 3);
                Eigen::Matrix3d R_c2w = R_wc2.transpose();
                Eigen::Vector3d t_c2_w = -R_c2w * t_wc2;

                Eigen::Vector3d p_c2_true = R_c2w * p_w + t_c2_w;

                // T_c2_c1 * c1
                Eigen::Vector3d p2_pre = R * p_c1_true + t * t_scale;

                //
                std::cout << std::endl;
                std::cout << "t_c1_w is " << p_c1_true.transpose() << std::endl;
                std::cout << "p_c2_true is " << p_c2_true.transpose() << std::endl;
                std::cout << "p2_pre is " << p2_pre.transpose() << std::endl;
                std::cout << std::endl;
                // std::cout<<"sub ="<<p1_pre-p_c1_true<<std::endl;
            }
        }
        /* show estimated trajectory */
        {
            // 显示估计的轨迹
            Eigen::Vector3d twc0 = Twc_last.block(0, 3, 3, 1);
            Eigen::Vector3d twc1 = Twc_curr.block(0, 3, 3, 1);
            cv::Point3d pose_begin(twc0[0], twc0[1], twc0[2]);
            cv::Point3d pose_end(twc1[0], twc1[1], twc1[2]);
            cv::viz::WLine trag_line(pose_begin, pose_end, cv::viz::Color(0, 255, 255));
            window.showWidget("trag_" + std::to_string(i), trag_line);

            // 估计轨迹的Z轴
            Eigen::Matrix3d Rwc1 = Twc_curr.block(0, 0, 3, 3);
            Eigen::Vector3d w_cam_z_dir = Rwc1 * Eigen::Vector3d(0, 0, 1);
            cv::Point3d obvs_dir(w_cam_z_dir[0], w_cam_z_dir[1], w_cam_z_dir[2]);
            cv::Point3d obvs_begin(twc1[0], twc1[1], twc1[2]);
            cv::viz::WLine obvs_line(obvs_begin, obvs_begin + obvs_dir, cv::viz::Color(255, 0, 0));
            window.showWidget("cam_z_" + std::to_string(i), obvs_line);
        }

        /* update */
        pose_est.push_back(Twc_curr);
        Twc_last = Twc_curr;
        point_last = point_curr;
    }

    /* save trajectory for evalution */
    saveTrajectoryTUM("../result/frame_traj_gt.txt", v_Twc);
    saveTrajectoryTUM("../result/frame_traj_est.txt", pose_est);

    cout << "一共有" << pointPair << "对匹配点参与了匹配" << endl;

    std::chrono::duration<double> time_span;
    for(auto i :time_costs)
    {
        time_span+=i;
    }
    auto F_time = time_span.count()/time_costs.size();
    cout<<"计算基础矩阵耗时"<<F_time<< " seconds."<<endl;

    while (!window.wasStopped())
    {
        window.spinOnce(1, true);
    }
}

/**
 * @description: 仿真生成路标点
 * @param {vector<Eigen::Vector3d>} &points 路标点的存储
 * @return {*}
 */
void createLandmarks(std::vector<Eigen::Vector3d> &points)
{
    points.clear();

// 生成阵列点云,两个交叉平面
#if 0

    // 点云缩放稀疏
    float scale = 5;

    // XY的比例
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

    // 伪随机数产生器
    std::mt19937 gen{12345};
    // 随机分布的XYZ点云
    // 参数： 均值 标准差
    std::normal_distribution<double> d_x{0.0, 4.0};
    std::normal_distribution<double> d_y{0.0, 10.0};
    std::normal_distribution<double> d_z{0.0, 10.0};

    for (int i = 0; i < 200; i++)
    {
        Eigen::Vector3d pt;
        // 取整
        pt[0] = std::round(d_x(gen));
        pt[1] = std::round(d_y(gen));
        pt[2] = std::round(d_z(gen));
        points.push_back(pt);
    }

#endif
}

/**
 * @description: 仿真得到相机的位姿
 * @param {vector<Eigen::Matrix4d>} &v_Twc
 * @param {Vector3d} &point_focus
 * @return {*}
 */
void createCameraPose(std::vector<Eigen::Matrix4d> &v_Twc)
{
    // 相机距离世界系原点的位置
    float x_offset = 20;
    float y_offset = 0;
    float z_offset = -5;
    float scale = 10;

    v_Twc.clear();

    for (float angle = 0; angle < 4 * 360; angle += 15)
    {
        // 角度转弧度
        float theta = angle * 3.14159 / 180.0f;

        // t_wc_w
        Eigen::Vector3d pt;
        pt[0] = cos(theta);
        pt[1] = sin(theta);
        pt[2] = theta / 20;
        pt = scale * pt;
        pt[0] += x_offset;
        pt[1] += y_offset;
        pt[2] += z_offset;
        Eigen::Vector3d t_wc = pt;

        // 相机系下的Z轴
        static const Eigen::Vector3d b_cam_z(0, 0, 1);

        // 相机系下的Z轴变换到平移向量上，
        // 也就是相机的Z轴 指向世界坐标系原点
        // 得到旋转四元数
        Eigen::Matrix3d R_cam_world(Eigen::Quaterniond::FromTwoVectors(-t_wc, b_cam_z));
        Eigen::Matrix3d Rwc(R_cam_world.transpose());

        Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
        Twc.block(0, 0, 3, 3) = Rwc;

        // t_wb_w
        Twc.block(0, 3, 3, 1) = t_wc;
        v_Twc.push_back(Twc);
    }
}

void detectFeatures(const Eigen::Matrix4d &Twc, const Eigen::Matrix3d &K,
                    const std::vector<Eigen::Vector3d> &landmarks, std::vector<Eigen::Vector2i> &features, bool add_noise)
{

    std::mt19937 gen{12345};

    // 图像特征点的噪声--测量噪声
    const float pixel_sigma = 2.0;
    std::normal_distribution<> d{0.0, pixel_sigma};

    // 相机的位姿
    Eigen::Matrix3d Rwc = Twc.block(0, 0, 3, 3);
    Eigen::Vector3d twc = Twc.block(0, 3, 3, 1);
    Eigen::Matrix3d Rcw = Rwc.transpose();
    Eigen::Vector3d tcw = -Rcw * twc;

    features.clear();

    for (size_t l = 0; l < landmarks.size(); ++l)
    {
        //
        Eigen::Vector3d wP = landmarks[l];
        Eigen::Vector3d cP = Rcw * wP + tcw;

        // 相机无法观测到该特征点,赋值给0，方便后面剔除
        if (cP[2] < 0)
        {
            std::cout << "skip this " << std::endl;
            features.emplace_back(0, 0);
            continue;
        }

        // 添加噪声
        float noise_u = add_noise ? std::round(d(gen)) : 0.0f;
        float noise_v = add_noise ? std::round(d(gen)) : 0.0f;

        // 真实的特征 + 测量
        Eigen::Vector3d ft = K * cP;
        int u = ft[0] / ft[2] + 0.5 + noise_u;
        int v = ft[1] / ft[2] + 0.5 + noise_v;

        // 组合一下变成观测值
        Eigen::Vector2i obs(u, v);
        features.push_back(obs);

        // 输出一下观测关系
        // std::cout << l << " " << obs.transpose() << std::endl;
    }
}

void saveTrajectoryTUM(const std::string &file_name, const std::vector<Eigen::Matrix4d> &v_Twc)
{
    std::ofstream f;
    f.open(file_name.c_str());

    // 流操作符fixed 表示浮点数出与固定点或者小数点表示法显示
    f << std::fixed;

    for (size_t i = 0; i < v_Twc.size(); i++)
    {
        const Eigen::Matrix4d Twc = v_Twc[i];
        const Eigen::Vector3d t = Twc.block(0, 3, 3, 1);
        const Eigen::Matrix3d R = Twc.block(0, 0, 3, 3);
        const Eigen::Quaterniond q = Eigen::Quaterniond(R);

        // 指定浮点数的保留位数
        f << std::setprecision(6) << i << " "
          << std::setprecision(9) << t[0] << " " << t[1] << " " << t[2] << " " << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }

    f.close();
    std::cout << "save traj to " << file_name << " done." << std::endl;
}