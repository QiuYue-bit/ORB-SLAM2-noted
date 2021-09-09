/*
 * @Author: your name
 * @Date: 2021-09-03 17:58:50
 * @LastEditTime: 2021-09-03 20:41:39
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: /ORB_SLAM2_detailed_comments/Examples/Sensors/mono_d435i.cc
 */
#include <iostream>
#include <chrono>

#include <librealsense2/rs.hpp>
#include <opencv2/core/core.hpp>
#include <System.h>

using namespace std;
using namespace cv;

#define width 1280
#define height 720
#define fps 30

bool SLAM_state  = false;


void enable_stream_init(rs2::config cfg)
{
    //Add desired streams to configuration
    cfg.enable_stream(RS2_STREAM_COLOR, width, height, RS2_FORMAT_BGR8, fps);//向配置添加所需的流
    cfg.enable_stream(RS2_STREAM_DEPTH, width, height, RS2_FORMAT_Z16,fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 1, width, height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_INFRARED, 2, width, height, RS2_FORMAT_Y8, fps);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
}

//按下s结束
void Stop_thread()
{
    while(1)
    {
        char c = getchar();
        if (c == 's')
        {
            SLAM_state = false;
        }
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);
    }
}


//字典 内参
int main(int argc, char **argv)
{
    if(argc != 3)
    {
        cerr << endl << "Usage: ./mono_tran path_to_vocabulary path_to_settings " << endl;
        return 1;
    }
    //vector<double> vTimestamps;

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::MONOCULAR,true);

    SLAM_state = true;

    //配置realsense
    rs2::context ctx;
    auto list = ctx.query_devices(); // Get a snapshot of currently connected devices
    if (list.size() == 0)
        throw std::runtime_error("No device detected. Is it plugged in?");
    rs2::device dev = list.front();

    rs2::frameset frames;
    //Contruct a pipeline which abstracts the device
    rs2::pipeline pipe;//创建一个通信管道//https://baike.so.com/doc/1559953-1649001.html pipeline的解释
    //Create a configuration for configuring the pipeline with a non default profile
    rs2::config cfg;//创建一个以非默认配置的配置用来配置管道
    enable_stream_init(cfg);
    // start stream
    pipe.start(cfg);//指示管道使用所请求的配置启动流
    for( int i = 0; i < 30 ; i ++)
    {
        frames = pipe.wait_for_frames();
    }

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;

    thread stop_thread(Stop_thread);

    while(SLAM_state)
    {
        frames = pipe.wait_for_frames();//等待所有配置的流生成框架
        //   Align to color
        rs2::align align_to_color(RS2_STREAM_COLOR);
        frames = align_to_color.process(frames);

        // Get imu data
        //Get_imu_data(frames);

        //Get each frame
        rs2::frame color_frame = frames.get_color_frame();

        //rs2::depth_frame depth_frame = frames.get_depth_frame();
        //rs2::video_frame ir_frame_left = frames.get_infrared_frame(1);
       // rs2::video_frame ir_frame_right = frames.get_infrared_frame(2);

        // Creating OpenCV Matrix from a color image
        Mat color(Size(width, height), CV_8UC3, (void*)color_frame.get_data(), Mat::AUTO_STEP);
        double tframe = color_frame.get_timestamp();

        //Mat pic_right(Size(width,height), CV_8UC1, (void*)ir_frame_right.get_data());
        //Mat pic_left(Size(width,height), CV_8UC1, (void*)ir_frame_left.get_data());
        //Mat pic_depth(Size(width,height), CV_16U, (void*)depth_frame.get_data(), Mat::AUTO_STEP);
        namedWindow("Display Image", WINDOW_AUTOSIZE );
        imshow("Display Image", color);


#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        SLAM.TrackMonocular(color,tframe);

        if(color.empty())
        {
            cerr << endl << "Failed to load image at: "
                 <<  tframe << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        //TODO:检测跟踪时间
        
    }

    // Stop all threads
    SLAM.Shutdown();
    cout << "-------" << endl << endl;
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");

    return 0;
}
