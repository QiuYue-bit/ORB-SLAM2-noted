## ORB-SLAM2 第一次作业

#### 说明

1. 本工程采用opencv viz作为3D viewer，opencv version>= **2.4.9**

2. 建议安装**evo**  [https://github.com/MichaelGrupp/evo](https://github.com/MichaelGrupp/evo) 作为轨迹精度评估工具，评估ape误差：

   ```shell
   evo_rpe tum -a frame_traj_gt.txt frame_traj_est.txt
   ```

3. 代码中需要补全部分使用TODO注释给出

#### 作业
1. 通过两帧间特征点的匹配求解F矩阵，并恢复R，t。使用opencv的`findFundamentalMat`和`recoverPose`函数完成`demo/main.cpp`中的代码，实现基于2D-2D的视觉里程计。
2. 尝试修改仿真的landmarks分布和范围，以及像素误差，对比分析轨迹精度。
3. 使用ORB-SLAM2的`Initializer.cc`中求解F矩阵以及恢复R，t的代码，替换opencv的求解函数，补全`src/two_view_geometry.cpp`中的`FindFundamental`和`ReconstructF`函数（也可以根据自己喜好来定义）。



代码运行结果如下，白色点为landmarks，绿色以及黄色轨迹分别是仿真的真值以及VO的估计轨迹。

![viewer](\doc\viewer.png)