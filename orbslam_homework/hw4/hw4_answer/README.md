## ORB-SLAM2 第四次作业

#### 作业
1. 补充三角化部分代码，使得相机能够完整跟踪轨迹

2. 补全`optimizer.cpp`中`two_view_ba`的代码，使用g2o实现两视图的ba优化(先编译Thirdparity中的g2o)。

3. 修改函数`feature_match`的传参`outlier_rate=0.05`，使得两视图中存在$5\%$的错匹配。

   补全`outlier_rejection`函数中使用重投影误差剔除outlier的代码，保证最终精度不会因为outlier而明显降低

代码中需要补全部分使用TODO注释给出。下面两幅图分别是完成作业1完整跟踪的结果，以及增加了ba优化后的结果

![hw4_pnp_track](\doc\hw4_pnp_track.png)