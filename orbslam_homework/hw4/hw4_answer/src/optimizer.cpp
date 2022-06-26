#include <cmath>
#include "Thirdparty/g2o/g2o/core/block_solver.h"
#include "Thirdparty/g2o/g2o/core/optimization_algorithm_levenberg.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_eigen.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/core/robust_kernel_impl.h"
#include "Thirdparty/g2o/g2o/solvers/linear_solver_dense.h"
#include "Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"
#include "define.h"
#include "optimizer.h"
#include "convertor.h"

void two_view_ba(Frame &frame_last, Frame &frame_curr, LoaclMap &map, std::vector<FMatch> &matches, int n_iter)
{
    // TODO homework
    // after you complete this funtion, remove the "return"
    // return;

    const double fx = frame_last.K_(0, 0);
    const double fy = frame_last.K_(1, 1);
    const double cx = frame_last.K_(0, 2);
    const double cy = frame_last.K_(1, 2);

    // Eigen线性求解器
    // LM算法后端
    g2o::SparseOptimizer optimizer;
    g2o::BlockSolver_6_3::LinearSolverType *linearSolver;
    linearSolver = new g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>();
    g2o::BlockSolver_6_3 *solver_ptr = new g2o::BlockSolver_6_3(linearSolver);
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);


    // 两帧的顶点
    Eigen::Matrix4d last_Tcw = frame_last.Twc_.inverse();
    Eigen::Matrix4d curr_Tcw = frame_curr.Twc_.inverse();

    // 两帧的观测
    const std::vector<Eigen::Vector2i> &features_last = frame_last.fts_;
    const std::vector<Eigen::Vector2i> &features_curr = frame_curr.fts_;

    int frame_id_last = frame_last.idx_;
    int frame_id_curr = frame_curr.idx_;
    // TODO 向求解器中添加顶点 pose
    // add frame pose Vertex to optimizer
    // example:
    // g2o::VertexSE3Expmap * vSE3 = new g2o::VertexSE3Expmap();
    // ...
    // optimizer.addVertex(vSE3);

    // 上一帧的位姿
    {
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setId(frame_id_last);
        // Tcw
        vSE3->setEstimate(g2o::SE3Quat(last_Tcw.block(0, 0, 3, 3), last_Tcw.block(0, 3, 3, 1)));

        // fix住上一帧的位姿
        vSE3->setFixed(true);

        optimizer.addVertex(vSE3);
    }

    // 当前帧的位姿
    {
        g2o::VertexSE3Expmap *vSE3 = new g2o::VertexSE3Expmap();
        vSE3->setId(frame_id_curr);
        // Tcw
        vSE3->setEstimate(g2o::SE3Quat(curr_Tcw.block(0, 0, 3, 3), curr_Tcw.block(0, 3, 3, 1)));
        optimizer.addVertex(vSE3);
    }

    const float thHuber2D = sqrt(5.99);
    const float thHuber3D = sqrt(7.815);
    bool bRobust = true;

    int max_frame_id = std::max(frame_id_last, frame_id_curr) + 1;

    // Set MapPoint vertices

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].outlier)
            continue;

        uint32_t idx_curr = matches[i].first;
        uint32_t idx_last = matches[i].second;

        // 取出当前帧特征点匹配的地图点id
        int32_t idx_mpt = frame_curr.mpt_track_[idx_curr];
        assert(idx_mpt >= 0);

        assert(true == map.status_[idx_mpt]);
        assert(idx_mpt == frame_last.mpt_track_[idx_last]);

        // 把地图点取出来
        Eigen::Vector3d &mpt = map.mpts_[idx_mpt];

        // TODO homework 向求解器中添加顶点 landmark
        // add mappoint Vertex to optimizer
        // example:
        // g2o::VertexSBAPointXYZ * vPoint = new g2o::VertexSBAPointXYZ();
        // ...
        // optimizer.addVertex(vPoint);
        const int id = idx_mpt + max_frame_id + 1;
        {
            g2o::VertexSBAPointXYZ *vPoint = new g2o::VertexSBAPointXYZ();
            vPoint->setEstimate(mpt);
            vPoint->setId(id);
            // vPoint->setFixed(true);
            vPoint->setMarginalized(true);
            optimizer.addVertex(vPoint);
        }

        // TODO homework 构建误差边
        // add edage to optimizer
        // example:
        // g2o::EdgeSE3ProjectXYZ* e = new g2o::EdgeSE3ProjectXYZ();
        // ...
        // optimizer.addEdge(e);

        // 上一帧的误差边
        {
            // 构造当前帧观测
            g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
            // 边连接的第0号顶点对应的是第id个地图点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
            // 链接地图点
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(frame_id_last)));
            // 设置测量值
            Eigen::Vector2d v;
            v[0] = frame_last.fts_[idx_last][0];
            v[1] = frame_last.fts_[idx_last][1];
            e->setMeasurement(v);

            // 信息矩阵
            e->setInformation(Eigen::Matrix2d::Identity());

            // 使用鲁棒核函数
            if (bRobust)
            {
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                // 这里的重投影误差，自由度为2，所以这里设置为卡方分布中自由度为2的阈值，如果重投影的误差大约大于1个像素的时候，就认为不太靠谱的点了，
                // 核函数是为了避免其误差的平方项出现数值上过大的增长
                rk->setDelta(thHuber2D);
            }

            // 设置相机内参
            e->fx = fx;
            e->fy = fy;
            e->cx = cx;
            e->cy = cy;

            optimizer.addEdge(e);
        }

        // 当前帧的误差边
        {
            // 构造当前帧观测
            g2o::EdgeSE3ProjectXYZ *e = new g2o::EdgeSE3ProjectXYZ();
            // 边连接的第0号顶点对应的是第id个地图点
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(id)));
            // 链接地图点
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex *>(optimizer.vertex(frame_id_curr)));
            // 设置测量值
            Eigen::Vector2d v;
            v[0] = frame_curr.fts_[idx_curr][0];
            v[1] = frame_curr.fts_[idx_curr][1];
            e->setMeasurement(v);
            // 信息矩阵
            e->setInformation(Eigen::Matrix2d::Identity());

            // 使用鲁棒核函数
            if (bRobust)
            {
                g2o::RobustKernelHuber *rk = new g2o::RobustKernelHuber;
                e->setRobustKernel(rk);
                // 这里的重投影误差，自由度为2，所以这里设置为卡方分布中自由度为2的阈值，如果重投影的误差大约大于1个像素的时候，就认为不太靠谱的点了，
                // 核函数是为了避免其误差的平方项出现数值上过大的增长
                rk->setDelta(thHuber2D);
            }

            // 设置相机内参
            e->fx = fx;
            e->fy = fy;
            e->cx = cx;
            e->cy = cy;

            optimizer.addEdge(e);
        }
    }

    // Optimize!
    optimizer.initializeOptimization();
    optimizer.optimize(n_iter);

    // TODO homework
    // Recover optimized data
    // 上一帧的相机位姿
    // {

    //     g2o::VertexSE3Expmap *vSE3 = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0));
    //     g2o::SE3Quat SE3quat = vSE3->estimate();
    //     frame_last.Twc_ = SE3quat.to_homogeneous_matrix().inverse();
    // }

    // 当前帧的相机位姿
    {
        g2o::VertexSE3Expmap *vSE3 = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(frame_id_curr));
        g2o::SE3Quat SE3quat = vSE3->estimate();
        frame_curr.Twc_ = SE3quat.to_homogeneous_matrix().inverse();
    }

    // Points
    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].outlier)
        {
            continue;
        }
        uint32_t idx_last = matches[i].second;
        int32_t idx_mpt = frame_last.mpt_track_[idx_last];

        Eigen::Vector3d &mpt = map.mpts_[idx_mpt];

        // 获取优化之后的地图点的位置
        g2o::VertexSBAPointXYZ *vPoint = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(idx_mpt + max_frame_id + 1));

        mpt = vPoint->estimate();
    }
}