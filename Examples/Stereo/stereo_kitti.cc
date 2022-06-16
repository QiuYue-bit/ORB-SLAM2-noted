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


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include<opencv2/core/core.hpp>


#include<System.h>

using namespace std;

/**
 * @brief 加载图像
 * @param[in]  strPathLeft          保存左目图像文件名称的文件的路径
 * @param[in]  strPathRight         保存右目图像文件名称的文件的路径
 * @param[in]  strPathTimes         保存图像时间戳的文件的路径
 * @param[out] vstrImageLeft        左目图像序列中每张图像的文件名
 * @param[out] vstrImageRight       右目图像序列中每张图像的文件名
 * @param[out] vTimeStamps          图像序列中每张图像的时间戳(认为左目图像和右目图像的时间戳已经对齐)
 */
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc != 4)
    {
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1;
    }

    // 读取图像路径和图像的时间戳
    vector<string> vstrImageLeft;
    vector<string> vstrImageRight;
    vector<double> vTimestamps;
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps);

    const int nImages = vstrImageLeft.size();

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    // 创建SLAM系统，其初始化所有的线程，准备好处理图像数据
    
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true);
   
    // Vector for tracking time statistics
    // 存储每帧跟踪消耗的时间
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // 输出序列中的图像数目
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;   

    // NOTE 由于Kitti数据集的图像已经经过双目矫正的处理，所以这里就不需要再进行矫正的操作了
    // NOTE Euroc数据集还需要进行双目矫正

    // Main loop
    cv::Mat imLeft, imRight;
    for(int ni=0; ni<nImages; ni++)
    {
        // * Step 1 读取图像
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED);
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED);
        double tframe = vTimestamps[ni];

        if(imLeft.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();


        // * Step 2前端跟踪
        SLAM.TrackStereo(imLeft,imRight,tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        // 存储每张图片的跟踪耗时
        vTimesTrack[ni]=ttrack;

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
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }

    // 输出前端处理的时间统计信息
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;


    // 存储所有的关键帧的位姿 ， KITTI 格式
    // mTimeStamp ---- t[3] ------ q[4]
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");

    return 0;
}

// 类似 mono_kitti.cc， 不过是生成了双目的图像路径
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    ifstream fTimes;
    string strPathTimeFile = strPathToSequence + "/times.txt";
    fTimes.open(strPathTimeFile.c_str());
    while(!fTimes.eof())
    {
        string s;
        getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            ss >> t;
            vTimestamps.push_back(t);
        }
    }

    string strPrefixLeft = strPathToSequence + "/image_0/";
    string strPrefixRight = strPathToSequence + "/image_1/";

    const int nTimes = vTimestamps.size();
    vstrImageLeft.resize(nTimes);
    vstrImageRight.resize(nTimes);


    for(int i=0; i<nTimes; i++)
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i;
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png";
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png";
    }
}
