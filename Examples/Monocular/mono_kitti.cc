/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <iomanip>

#include <opencv2/core/core.hpp>

#include "System.h"

using namespace std;

/**
 * @brief 获取图像序列中每一张图像的访问路径和时间戳
 * @param[in]  strSequence              图像序列的存放路径
 * @param[out] vstrImageFilenames       图像序列中每张图像的存放路径
 * @param[out] vTimestamps              图像序列中每张图像的时间戳
 */
static void LoadImages(const string &strSequence, vector<string> &vstrImageFilenames,
                       vector<double> &vTimestamps);

// 主函数工作原理和 mono_euroc.cc 基本相同
int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cerr << endl
             << "Usage: ./mono_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // 读取图像路径和图像的时间戳
    vector<string> vstrImageFilenames;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // 创建SLAM系统，其初始化所有的线程，准备好处理图像数据
    ORB_SLAM2::System SLAM(argv[1], argv[2], ORB_SLAM2::System::MONOCULAR, true);

    // Vector for tracking time statistics
    // 存储每帧跟踪消耗的时间
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // 输出序列中的图像数目
    cout << endl
         << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl
         << endl;

    // 开始循环处理图像
    cv::Mat im;
    for (int ni = 0; ni < nImages; ni++)
    {
        // * Step 1 读取图像
        im = cv::imread(vstrImageFilenames[ni], CV_LOAD_IMAGE_UNCHANGED);

        // 图像的时间戳
        double tframe = vTimestamps[ni];

        if (im.empty())
        {
            cerr << endl
                 << "Failed to load image at: " << vstrImageFilenames[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // * Step 2前端跟踪
        SLAM.TrackMonocular(im, tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();

        // 存储每张图片的跟踪耗时
        vTimesTrack[ni] = ttrack;

        // 两帧之间的时间戳之差
        double T = 0;
        if (ni < nImages - 1)
        {
            // 两张图象的时间差
            T = vTimestamps[ni + 1] - tframe;
        }
        else if (ni > 0)
        {
            // 处理最后一帧
            T = tframe - vTimestamps[ni - 1];
        }

        // 延时处理，使得和图像具有相同的频率。
        if (ttrack < T)
        {
            // usleep((T - ttrack) * 1e6);
        }
        else
        {
            cout << "跟踪消耗的时间大于帧率，无法实时计算" << endl;
        }
    }

    // 停止所有的线程，回收资源
    SLAM.Shutdown();

    // Tracking time statistics
    // 耗时排个序
    sort(vTimesTrack.begin(), vTimesTrack.end());

    float totaltime = 0;
    for (int ni = 0; ni < nImages; ni++)
    {
        totaltime += vTimesTrack[ni];
    }

    // 输出前端处理的时间统计信息
    cout << "-------" << endl
         << endl;
    cout << "median tracking time: " << vTimesTrack[nImages / 2] << endl;
    cout << "mean tracking time: " << totaltime / nImages << endl;

    // 存储所有的关键帧的位姿 ， Tum 格式
    // mTimeStamp ---- t[3] ------ q[4]
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}

// 获取图像序列中每一张图像的访问路径和时间戳
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    // step 1 读取时间戳文件
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while (!fTimes.eof())
    {
        string s;
        getline(fTimes, s);
        // 当该行不为空的时候执行
        if (!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            // 保存时间戳
            vTimestamps.push_back(t);
        }
    }

    // step 1 使用左目图像, 生成左目图像序列中的每一张图像的文件名
    string strPrefixLeft = strPathToSequence + "/image_0/";

    const int nTimes = vTimestamps.size();
    vstrImageFilenames.resize(nTimes);

    for (int i = 0; i < nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageFilenames[i] = strPrefixLeft + ss.str() + ".png";
    }
}
