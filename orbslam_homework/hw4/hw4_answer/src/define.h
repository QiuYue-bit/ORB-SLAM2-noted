/*
 * @Author: Divenire qiuyue_online@163.com
 * @Date: 2022-06-20 18:55:17
 * @LastEditors: Divenire qiuyue_online@163.com
 * @LastEditTime: 2022-06-24 15:25:11
 * @FilePath: /hw4/src/define.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#ifndef __DEFINE_H__
#define __DEFINE_H__

#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>
#include <Eigen/StdVector>

EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix3d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix4d)
EIGEN_DEFINE_STL_VECTOR_SPECIALIZATION(Eigen::Matrix2i)

struct FMatch {
    FMatch(uint32_t _first, uint32_t _second, bool _outlier) :
        first(_first), second(_second), outlier(_outlier), outlier_gt(false) {}
    uint32_t first;
    uint32_t second;
    bool outlier;
    bool outlier_gt;
};

// 新建帧
class Frame{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Frame(uint64_t idx, size_t N, const Eigen::Matrix3d &K) : idx_(idx), N_(N), K_(K) {}
    uint64_t idx_;
    const size_t N_;
    Eigen::Matrix3d K_; 
    Eigen::Matrix4d Twc_;
    std::vector<Eigen::Vector2i> fts_;
    std::vector<int32_t> fts_obvs_;          //! landmarks idx tracked
    std::vector<int32_t> mpt_track_;         //! mappoint idx tracked

private:
    Frame() = delete;
};

class LoaclMap
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LoaclMap(size_t N) : N_(N)
    {
        mpts_.resize(N);
        status_ = std::vector<bool>(N, false);
    }

    size_t add(const Eigen::Vector3d &mpt)
    {
        for (size_t i = 0; i < N_; i++)
        {
            if (false == status_[i])
            {
                mpts_[i] = mpt;
                status_[i] = true;
                return i;
            }
        }
        return -1;
    }

    void remove(size_t idx)
    {
        assert(idx < N_);
        status_[idx] = false;
    }

    void clear()
    {
        status_ = std::vector<bool>(N_, false);
    }

    // make sure that only the mpts tracked by curr frame are remained
    void update(const Frame &frame_curr)
    {
        status_ = std::vector<bool>(N_, false);
        for (size_t i = 0; i < frame_curr.mpt_track_.size(); i++)
        {
            int32_t mpt_idx = frame_curr.mpt_track_[i];
            if (mpt_idx < 0) { continue; }
            status_[mpt_idx] = true;
        }
    }

public:
    const size_t N_;
    std::vector<Eigen::Vector3d> mpts_;
    std::vector<bool> status_;

};

#endif //__DEFINE_H__