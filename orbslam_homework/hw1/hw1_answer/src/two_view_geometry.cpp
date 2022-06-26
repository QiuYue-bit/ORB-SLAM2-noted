/*
 * @Author: Divenire qiuyue_online@163.com
 * @Date: 2022-06-20 18:55:17
 * @LastEditors: Divenire qiuyue_online@163.com
 * @LastEditTime: 2022-06-22 21:58:21
 * @FilePath: /hw1/src/two_view_geometry.cpp
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "two_view_geometry.h"

namespace TwoViewGeometry
{

   void Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T) //将特征点归一化的矩阵
   {
      // 归一化的是这些点在x方向和在y方向上的一阶绝对矩（随机变量的期望）。

      // Step 1 计算特征点X,Y坐标的均值 meanX, meanY
      float meanX = 0;
      float meanY = 0;

      //获取特征点的数量
      const int N = vKeys.size();

      //设置用来存储归一后特征点的向量大小，和归一化前保持一致
      vNormalizedPoints.resize(N);

      //开始遍历所有的特征点
      for (int i = 0; i < N; i++)
      {
         //分别累加特征点的X、Y坐标
         meanX += vKeys[i].pt.x;
         meanY += vKeys[i].pt.y;
      }

      //计算X、Y坐标的均值
      meanX = meanX / N;
      meanY = meanY / N;

      // Step 2 计算特征点X,Y坐标离均值的平均偏离程度 meanDevX, meanDevY，注意不是标准差
      float meanDevX = 0;
      float meanDevY = 0;

      // 将原始特征点减去均值坐标，使x坐标和y坐标均值分别为0
      for (int i = 0; i < N; i++)
      {
         vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
         vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

         //累计这些特征点偏离横纵坐标均值的程度
         meanDevX += fabs(vNormalizedPoints[i].x);
         meanDevY += fabs(vNormalizedPoints[i].y);
      }

      // 求出平均到每个点上，其坐标偏离横纵坐标均值的程度；将其倒数作为一个尺度缩放因子
      meanDevX = meanDevX / N;
      meanDevY = meanDevY / N;
      float sX = 1.0 / meanDevX;
      float sY = 1.0 / meanDevY;

      // Step 3 将x坐标和y坐标分别进行尺度归一化，使得x坐标和y坐标的一阶绝对矩分别为1
      // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值（期望）
      for (int i = 0; i < N; i++)
      {
         //对，就是简单地对特征点的坐标进行进一步的缩放
         vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
         vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
      }

      // Step 4 计算归一化矩阵：其实就是前面做的操作用矩阵变换来表示而已
      // |sX  0  -meanx*sX|
      // |0   sY -meany*sY|
      // |0   0      1    |
      T = cv::Mat::eye(3, 3, CV_32F);
      T.at<float>(0, 0) = sX;
      T.at<float>(1, 1) = sY;
      T.at<float>(0, 2) = -meanX * sX;
      T.at<float>(1, 2) = -meanY * sY;
   }

   cv::Mat ComputeF21(
       const vector<cv::Point2f> &vP1, //归一化后的点, in reference frame
       const vector<cv::Point2f> &vP2) //归一化后的点, in current frame
   {

      const int N = vP1.size();

      //初始化A矩阵
      cv::Mat A(N, 9, CV_32F); // N*9维

      // 构造矩阵A，将每个特征点添加到矩阵A中的元素
      for (int i = 0; i < N; i++)
      {
         const float u1 = vP1[i].x;
         const float v1 = vP1[i].y;
         const float u2 = vP2[i].x;
         const float v2 = vP2[i].y;

         A.at<float>(i, 0) = u2 * u1;
         A.at<float>(i, 1) = u2 * v1;
         A.at<float>(i, 2) = u2;
         A.at<float>(i, 3) = v2 * u1;
         A.at<float>(i, 4) = v2 * v1;
         A.at<float>(i, 5) = v2;
         A.at<float>(i, 6) = u1;
         A.at<float>(i, 7) = v1;
         A.at<float>(i, 8) = 1;
      }

      //存储奇异值分解结果的变量
      cv::Mat u, w, vt;

      // 定义输出变量，u是左边的正交矩阵U， w为奇异矩阵，vt中的t表示是右正交矩阵V的转置
      cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
      // 转换成基础矩阵的形式
      cv::Mat Fpre = vt.row(8).reshape(0, 3); // v的最后一列

      //基础矩阵的秩为2,而我们不敢保证计算得到的这个结果的秩为2,所以需要通过第二次奇异值分解,来强制使其秩为2
      // 对初步得来的基础矩阵进行奇异值分解

      cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

      // 秩2约束，强制将第3个奇异值设置为0
      w.at<float>(2) = 0;

      // 重新组合好满足秩约束的基础矩阵，作为最终计算结果返回
      return u * cv::Mat::diag(w) * vt;
   }

   float CheckFundamental(
       vector<cv::KeyPoint> mvKeys1, vector<cv::KeyPoint> mvKeys2,
       const cv::Mat &F21,             //当前帧和参考帧之间的基础矩阵
       vector<bool> &vbMatchesInliers, //匹配的特征点对属于inliers的标记
       float sigma)
   {

      // 获取匹配的特征点对的总对数
      const int N = mvKeys1.size();

      // Step 1 提取基础矩阵中的元素数据
      const float f11 = F21.at<float>(0, 0);
      const float f12 = F21.at<float>(0, 1);
      const float f13 = F21.at<float>(0, 2);
      const float f21 = F21.at<float>(1, 0);
      const float f22 = F21.at<float>(1, 1);
      const float f23 = F21.at<float>(1, 2);
      const float f31 = F21.at<float>(2, 0);
      const float f32 = F21.at<float>(2, 1);
      const float f33 = F21.at<float>(2, 2);

      // 预分配空间
      vbMatchesInliers.resize(N);

      // 设置评分初始值（因为后面需要进行这个数值的累计）
      float score = 0;

      // 基于卡方检验计算出的阈值
      // 自由度为1的卡方分布，显著性水平为0.05，对应的临界阈值
      // 点到直线的自由度为1
      const float th = 3.841;

      // 自由度为2的卡方分布，显著性水平为0.05，对应的临界阈值
      // 作用是给F矩阵打分，主要是于H矩阵一致
      // TODO 注意，H矩阵和F矩阵评分的标准是不一样的 一个是双边点到直线的距离、一个是双边重投影误差
      const float thScore = 5.991;

      // 根据点到直线的距离标准差计算方差的逆
      const float invSigmaSquare = 1.0 / (sigma * sigma);

      // Step 2 计算img1 和 img2 在估计 F 时的score值
      for (int i = 0; i < N; i++)
      {
         //默认为这对特征点是Inliers
         bool bIn = true;

         // Step 2.1 提取参考帧和当前帧之间的特征匹配点对
         const cv::KeyPoint &kp1 = mvKeys1[i];
         const cv::KeyPoint &kp2 = mvKeys2[i];

         // 提取出特征点的坐标
         const float u1 = kp1.pt.x;
         const float v1 = kp1.pt.y;
         const float u2 = kp2.pt.x;
         const float v2 = kp2.pt.y;

         // ----------------------------- 计算图像2上的点，到极线的距离(高斯分布)，并进行卡方检验及评分

         // Reprojection error in second image
         // Step 2.2 计算 img1 上的点在 img2 上投影得到的极线 l2 = F21 * p1 = (a2,b2,c2)
         const float a2 = f11 * u1 + f12 * v1 + f13;
         const float b2 = f21 * u1 + f22 * v1 + f23;
         const float c2 = f31 * u1 + f32 * v1 + f33;

         // Step 2.3 计算误差 点到直线的距离，公式如下：
         // e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
         const float num2 = a2 * u2 + b2 * v2 + c2;
         const float squareDist1 = num2 * num2 / (a2 * a2 + b2 * b2);

         // 点到直线的距离服从 N(0,σ2)
         // 这儿进行归一化，使得squareDist1服从自由度为1得卡方分布
         const float chiSquare1 = squareDist1 * invSigmaSquare;

         // Step 2.4 卡方检验 95% 1自由度
         if (chiSquare1 > th)
            bIn = false;
         else
            // 误差越大，得分越低
            score += thScore - chiSquare1;

         // ----------------------------- 计算图像1上的点，到极线的距离，并进行卡方检验及评分

         // 计算img2上的点在 img1 上投影得到的极线 l1= p2 * F21 = (a1,b1,c1)
         const float a1 = f11 * u2 + f21 * v2 + f31;
         const float b1 = f12 * u2 + f22 * v2 + f32;
         const float c1 = f13 * u2 + f23 * v2 + f33;

         // 计算误差 e = (a * p2.x + b * p2.y + c) /  sqrt(a * a + b * b)
         const float num1 = a1 * u1 + b1 * v1 + c1;
         const float squareDist2 = num1 * num1 / (a1 * a1 + b1 * b1);

         // 带权重误差
         const float chiSquare2 = squareDist2 * invSigmaSquare;

         // 误差大于阈值就说明这个点是Outlier
         if (chiSquare2 > th)
            bIn = false;
         else
            score += thScore - chiSquare2;

         // Step 2.5 保存结果
         if (bIn)
            vbMatchesInliers[i] = true;
         else
            vbMatchesInliers[i] = false;
      }
      //  返回评分
      return score;
   }

   void FindFundamental(vector<cv::KeyPoint> mvKeys1, vector<cv::KeyPoint> mvKeys2,
                        vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
   {
      // 计算基础矩阵,其过程和上面的计算单应矩阵的过程十分相似.

      // Number of putative matches
      // 匹配的特征点对总数
      const int N = mvKeys1.size();

      // Normalize coordinates
      // Step 1 将当前帧和参考帧中的特征点坐标进行归一化，主要是平移和尺度变换
      // 具体来说,就是将mvKeys1和mvKey2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1、T2
      // 这里所谓的一阶绝对矩其实就是随机变量到取值的中心的绝对值的平均值
      // 归一化矩阵就是把上述归一化的操作用矩阵来表示。这样特征点坐标乘归一化矩阵可以得到归一化后的坐标

      vector<cv::Point2f> vPn1, vPn2;
      cv::Mat T1, T2;
      Normalize(mvKeys1, vPn1, T1);
      Normalize(mvKeys2, vPn2, T2);

      //这里的转职在后面反归一化会用到
      cv::Mat T2t = T2.t();

      // Best Results variables
      //最优结果
      score = 0.0;
      // 取得历史最佳评分时,特征点对的inliers标记
      vbMatchesInliers = vector<bool>(N, false);

      // Iteration variables
      // 某次迭代中，参考帧的特征点坐标
      vector<cv::Point2f> vPn1i(8);
      // 某次迭代中，当前帧的特征点坐标
      vector<cv::Point2f> vPn2i(8);
      // 某次迭代中，计算的基础矩阵
      cv::Mat F21i;

      // 每次RANSAC记录的Inliers与得分
      vector<bool> vbCurrentInliers(N, false);
      float currentScore;

      // Perform all RANSAC iterations and save the solution with highest score
      // 下面进行每次的RANSAC迭代
      std::mt19937 gen{12345};
      std::uniform_int_distribution<int> d(0, N - 1);

      // !写死了 最大迭代300次
      for (int it = 0; it < 300; it++)
      {
         // Select a minimum set
         // Step 2 选择8个归一化之后的点对进行迭代

         for (int j = 0; j < 8; j++)
         {
            int idx = d(gen);

            // vPn1i和vPn2i为匹配的特征点对的归一化后的坐标
            // 首先根据这个特征点对的索引信息分别找到两个特征点在各自图像特征点向量中的索引，然后读取其归一化之后的特征点坐标
            vPn1i[j] = vPn1[idx]; // first存储在参考帧1中的特征点索引
            vPn2i[j] = vPn2[idx]; // second存储在参考帧1中的特征点索引
         }

         // Step 3 八点法计算基础矩阵
         cv::Mat Fn = ComputeF21(vPn1i, vPn2i);

         // 基础矩阵约束：p2^t*F21*p1 = 0，其中p1,p2 为齐次化特征点坐标
         // 特征点归一化：vPn1 = T1 * mvKeys1, vPn2 = T2 * mvKeys2
         // 根据基础矩阵约束得到:(T2 * mvKeys2)^t* Hn * T1 * mvKeys1 = 0
         // 进一步得到:mvKeys2^t * T2^t * Hn * T1 * mvKeys1 = 0
         F21i = T2t * Fn * T1;

         // Step 4 利用重投影误差为当次RANSAC的结果评分
         // !写死了 噪声水平为1个像素
         currentScore = CheckFundamental(mvKeys1, mvKeys2, F21i, vbCurrentInliers, 1);

         // Step 5 更新具有最优评分的基础矩阵计算结果,并且保存所对应的特征点对的内点标记
         if (currentScore > score)
         {
            //如果当前的结果得分更高，那么就更新最优计算结果
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
         }
      }
   }

   // ======================================= 由F矩阵恢复位姿===========================================
   void Triangulate(
       const cv::KeyPoint &kp1, //特征点, in reference frame
       const cv::KeyPoint &kp2, //特征点, in current frame
       const cv::Mat &P1,       //投影矩阵P1
       const cv::Mat &P2,       //投影矩阵P2
       cv::Mat &x3D)            //三维点
   {

      cv::Mat A(4, 4, CV_32F);

      //构造参数矩阵A
      A.row(0) = kp1.pt.x * P1.row(2) - P1.row(0);
      A.row(1) = kp1.pt.y * P1.row(2) - P1.row(1);
      A.row(2) = kp2.pt.x * P2.row(2) - P2.row(0);
      A.row(3) = kp2.pt.y * P2.row(2) - P2.row(1);

      //奇异值分解的结果
      cv::Mat u, w, vt;
      //对系数矩阵A进行奇异值分解
      cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

      //根据前面的结论，奇异值分解右矩阵的最后一行其实就是解，原理类似于前面的求最小二乘解，四个未知数四个方程正好正定
      //别忘了我们更习惯用列向量来表示一个点的空间坐标
      x3D = vt.row(3).t();

      //为了符合其次坐标的形式，使最后一维为1，最后一维归一化
      x3D = x3D.rowRange(0, 3) / x3D.at<float>(3);
   }

   void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
   {

      // 对本质矩阵进行奇异值分解
      //准备存储对本质矩阵进行奇异值分解的结果
      cv::Mat u, w, vt;
      //对本质矩阵进行奇异值分解
      cv::SVD::compute(E, w, u, vt);

      // 左奇异值矩阵U的最后一列就是t，对其进行归一化
      u.col(2).copyTo(t);
      t = t / cv::norm(t);

      // 构造一个绕Z轴旋转pi/2的旋转矩阵W，按照下式组合得到旋转矩阵 R1 = u*W*vt
      //计算完成后要检查一下旋转矩阵行列式的数值，使其满足行列式为1的约束
      cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
      W.at<float>(0, 1) = -1;
      W.at<float>(1, 0) = 1;
      W.at<float>(2, 2) = 1;

      //计算
      R1 = u * W * vt;
      //旋转矩阵有行列式为+1的约束，所以如果算出来为负值，需要取反
      if (cv::determinant(R1) < 0)
         R1 = -R1;

      // 同理将矩阵W取转置来按照相同的公式计算旋转矩阵R2 = u*W.t()*vt
      R2 = u * W.t() * vt;
      //旋转矩阵有行列式为1的约束
      if (cv::determinant(R2) < 0)
         R2 = -R2;
   }

   int CheckRT(const cv::Mat &R, const cv::Mat &t,
               const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
               vector<bool> &vbMatchesInliers,
               const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
   {
      // 对给出的特征点对及其R t , 通过三角化检查解的有效性，也称为 cheirality check

      // Calibration parameters
      //从相机内参数矩阵获取相机的校正参数
      const float fx = K.at<float>(0, 0);
      const float fy = K.at<float>(1, 1);
      const float cx = K.at<float>(0, 2);
      const float cy = K.at<float>(1, 2);

      //特征点是否是good点的标记，这里的特征点指的是参考帧中的特征点
      vbGood = vector<bool>(vKeys1.size(), false);
      //重设存储空间坐标的点的大小
      vP3D.resize(vKeys1.size());

      //存储计算出来的每对特征点的视差
      vector<float> vCosParallax;
      vCosParallax.reserve(vKeys1.size());

      // Camera 1 Projection Matrix K[I|0]
      // Step 1：计算相机的投影矩阵
      cv::Mat P1(3, 4,           //矩阵的大小是3x4
                 CV_32F,         //数据类型是浮点数
                 cv::Scalar(0)); //初始的数值是0
                                 //将整个K矩阵拷贝到P1矩阵的左侧3x3矩阵，因为 K*I = K
      K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
      // 第一个相机的光心设置为世界坐标系下的原点
      cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);

      // Camera 2 Projection Matrix K[R|t]
      // 计算第二个相机的投影矩阵 P2=K*[R|t]
      cv::Mat P2(3, 4, CV_32F);
      R.copyTo(P2.rowRange(0, 3).colRange(0, 3));
      t.copyTo(P2.rowRange(0, 3).col(3));
      //最终结果是K*[R|t]
      P2 = K * P2;
      // 第二个相机的光心在世界坐标系下的坐标
      cv::Mat O2 = -R.t() * t;

      //在遍历开始前，先将good点计数设置为0
      int nGood = 0;

      // 开始遍历所有的特征点对
      for (size_t i = 0, iend = vKeys1.size(); i < iend; i++)
      {

         // 跳过outliers
         if (!vbMatchesInliers[i])
            continue;

         // Step 2 获取特征点对，调用Triangulate() 函数进行三角化，得到三角化测量之后的3D点坐标
         // kp1和kp2是匹配好的有效特征点
         const cv::KeyPoint &kp1 = vKeys1[i];
         const cv::KeyPoint &kp2 = vKeys2[i];
         //存储三维点的的坐标
         cv::Mat p3dC1;

         // 利用三角法恢复三维点p3dC1
         Triangulate(kp1, kp2, //特征点
                     P1, P2,   //投影矩阵
                     p3dC1);   //输出，三角化测量之后特征点的空间坐标

         // Step 3 第一关：检查三角化的三维点坐标是否合法（非无穷值）
         // 只要三角测量的结果中有一个是无穷大的就说明三角化失败，跳过对当前点的处理，进行下一对特征点的遍历
         if (!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
         {
            //其实这里就算是不这样写也没问题，因为默认的匹配点对就不是good点
            continue;
         }

         // Check parallax
         // Step 4 第二关：通过三维点深度值正负、两相机光心视差角大小来检查是否合法

         //得到向量PO1
         cv::Mat normal1 = p3dC1 - O1;
         //求取模长，其实就是距离
         float dist1 = cv::norm(normal1);

         //同理构造向量PO2
         cv::Mat normal2 = p3dC1 - O2;
         //求模长
         float dist2 = cv::norm(normal2);

         //根据公式：a.*b=|a||b|cos_theta 可以推导出来下面的式子
         float cosParallax = normal1.dot(normal2) / (dist1 * dist2);

         // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
         // 如果深度值为负值，为非法三维点跳过该匹配点对
         // 视差比较小时，重投影误差比较大。这里0.99998 对应的角度为0.36°
         if (p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

         // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
         // 讲空间点p3dC1变换到第2个相机坐标系下变为p3dC2
         cv::Mat p3dC2 = R * p3dC1 + t;
         //判断过程和上面的相同
         if (p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
            continue;

         // Step 5 第三关：计算空间点在参考帧和当前帧上的重投影误差，如果大于阈值则舍弃
         // Check reprojection error in first image
         // 计算3D点在第一个图像上的投影误差
         //投影到参考帧图像上的点的坐标x,y
         float im1x, im1y;
         //这个使能空间点的z坐标的倒数
         float invZ1 = 1.0 / p3dC1.at<float>(2);
         //投影到参考帧图像上。因为参考帧下的相机坐标系和世界坐标系重合，因此这里就直接进行投影就可以了
         im1x = fx * p3dC1.at<float>(0) * invZ1 + cx;
         im1y = fy * p3dC1.at<float>(1) * invZ1 + cy;

         //参考帧上的重投影误差，这个的确就是按照定义来的
         float squareError1 = (im1x - kp1.pt.x) * (im1x - kp1.pt.x) + (im1y - kp1.pt.y) * (im1y - kp1.pt.y);

         // 重投影误差太大，跳过淘汰
         // TODO 调试来看一下这儿到底有多少被淘汰的
         if (squareError1 > th2)
            continue;

         // Check reprojection error in second image
         // 计算3D点在第二个图像上的投影误差，计算过程和第一个图像类似
         float im2x, im2y;
         // 注意这里的p3dC2已经是第二个相机坐标系下的三维点了
         float invZ2 = 1.0 / p3dC2.at<float>(2);
         im2x = fx * p3dC2.at<float>(0) * invZ2 + cx;
         im2y = fy * p3dC2.at<float>(1) * invZ2 + cy;

         // 计算重投影误差
         float squareError2 = (im2x - kp2.pt.x) * (im2x - kp2.pt.x) + (im2y - kp2.pt.y) * (im2y - kp2.pt.y);

         // 重投影误差太大，跳过淘汰
         if (squareError2 > th2)
            continue;

         // Step 6 统计经过检验的3D点个数，记录3D点视差角
         // 如果运行到这里就说明当前遍历的这个特征点对靠谱，经过了重重检验，说明是一个合格的点，称之为good点
         vCosParallax.push_back(cosParallax);
         //存储这个三角化测量后的3D点在世界坐标系下的坐标
         vP3D[i] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
         // good点计数++
         nGood++;

         //判断视差角，只有视差角稍稍大一丢丢的才会给打good点标记
         //? bug 我觉得这个写的位置不太对。你的good点计数都++了然后才判断，不是会让good点标志和good点计数不一样吗
         if (cosParallax < 0.99998)
            vbGood[i] = true;
      }

      // Step 7 得到3D点中较大的视差角，并且转换成为角度制表示
      if (nGood > 0)
      {
         // 从小到大排序，注意vCosParallax值越大，视差越小
         sort(vCosParallax.begin(), vCosParallax.end());

         // !排序后并没有取最小的视差角，而是取一个较小的视差角
         // 作者的做法：如果经过检验过后的有效3D点小于50个，那么就取最后那个最小的视差角(cos值最大)
         // 如果大于50个，就取排名第50个的较小的视差角即可，为了避免3D点太多时出现太小的视差角
         // 为什么不取中值？
         size_t idx = min(50, int(vCosParallax.size() - 1));
         //将这个选中的角弧度制转换为角度制
         parallax = acos(vCosParallax[idx]) * 180 / CV_PI;
      }
      else
         //如果没有good点那么这个就直接设置为0了
         parallax = 0;

      //返回good点计数
      return nGood;
   }

   bool ReconstructF(vector<cv::KeyPoint> mvKeys1, vector<cv::KeyPoint> mvKeys2,
                     vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                     cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
   {
      // Step 1 统计有效匹配点个数，并用 N 表示
      // vbMatchesInliers 中存储匹配点对是否是有效
      int N = 0;
      for (auto && vbMatchesInlier : vbMatchesInliers)
         if (vbMatchesInlier)
            N++;

      // Step 2 根据基础矩阵和相机的内参数矩阵计算本质矩阵
      cv::Mat E21 = K.t() * F21 * K;

      // 定义本质矩阵分解结果，形成四组解,分别是：
      // (R1, t) (R1, -t) (R2, t) (R2, -t)
      cv::Mat R1, R2, t;

      // Step 3 从本质矩阵求解两个R解和两个t解，共四组解
      DecomposeE(E21, R1, R2, t);
      cv::Mat t1 = t;
      cv::Mat t2 = -t;

      // Reconstruct with the 4 hyphoteses and check
      // Step 4 分别验证求解的4种R和t的组合，选出最佳组合
      vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;

      // 定义四组解分别对同一匹配点集的有效三角化结果，True or False
      vector<bool> vbTriangulated1, vbTriangulated2, vbTriangulated3, vbTriangulated4;

      // 定义四种解对应的比较大的特征点对视差角
      float parallax1, parallax2, parallax3, parallax4;

      // Step 4.1 使用同样的匹配点分别检查四组解，记录当前计算的3D点在摄像头前方且投影误差小于阈值的个数，记为有效3D点个数
      int nGood1 = CheckRT(R1, t1,           //当前组解
                           mvKeys1, mvKeys2, //参考帧和当前帧中的特征点
                           vbMatchesInliers, //特征点的匹配关系和Inliers标记
                           K,                //相机的内参数矩阵
                           vP3D1,            //存储三角化以后特征点的空间坐标
                           4.0,              //三角化测量过程中允许的最大重投影误差
                           vbTriangulated1,  //参考帧中被成功进行三角化测量的特征点的标记
                           parallax1);       //认为某对特征点三角化测量有效的比较大的视差角
      int nGood2 = CheckRT(R2, t1, mvKeys1, mvKeys2, vbMatchesInliers, K, vP3D2, 4.0, vbTriangulated2, parallax2);
      int nGood3 = CheckRT(R1, t2, mvKeys1, mvKeys2, vbMatchesInliers, K, vP3D3, 4.0, vbTriangulated3, parallax3);
      int nGood4 = CheckRT(R2, t2, mvKeys1, mvKeys2, vbMatchesInliers, K, vP3D4, 4.0, vbTriangulated4, parallax4);

      // Step 4.2 选取最大可三角化测量的点的数目
      int maxGood = max(nGood1, max(nGood2, max(nGood3, nGood4)));

      // 重置变量，并在后面赋值为最佳R和T
      R21 = cv::Mat();
      t21 = cv::Mat();

      // 统计四组解中重建的有效3D点个数 > 0.7 * maxGood 的解的数目
      // 如果有多个解同时满足该条件，认为结果太接近，nsimilar++，nsimilar>1就认为有问题了，后面返回false
      int nsimilar = 0;
      if (nGood1 > 0.7 * maxGood)
         nsimilar++;
      if (nGood2 > 0.7 * maxGood)
         nsimilar++;
      if (nGood3 > 0.7 * maxGood)
         nsimilar++;
      if (nGood4 > 0.7 * maxGood)
         nsimilar++;

      // Step 4.3 确定最小的可以三角化的点数
      // 在0.9倍的内点数 和 指定值minTriangulated =50 中取最大的，也就是说至少50个
//      int nMinGood = max(static_cast<int>(0.1 * N), minTriangulated);
        int nMinGood = minTriangulated;

      // Step 4.4 四个结果中如果没有明显的最优结果，或者没有足够数量的三角化点，则返回失败
      // 条件1: 如果四组解能够重建的最多3D点个数小于所要求的最少3D点个数（mMinGood），失败
      // 条件2: 如果存在两组及以上的解能三角化出 >0.7*maxGood的点，说明没有明显最优结果，失败
      if (maxGood < nMinGood || nsimilar > 1)
      {
         return false;
      }

      //  Step 4.5 选择最佳解记录结果
      // 条件1: 有效重建最多的3D点，即maxGood == nGoodx，也即是位于相机前方的3D点个数最多
      // 条件2: 三角化视差角 parallax 必须大于最小视差角 minParallax，角度越大3D点越稳定

      //看看最好的good点是在哪种解的条件下发生的
      if (maxGood == nGood1)
      {
         //如果该种解下的parallax大于函数参数中给定的最小值
         // TODO
         if (parallax1 > minParallax)
         {
            // 存储3D坐标
            vP3D = vP3D1;

            // 获取特征点向量的三角化测量标记
            vbTriangulated = vbTriangulated1;

            // 存储相机姿态
            R1.copyTo(R21);
            t1.copyTo(t21);

            // 结束
            return true;
         }
      }
      else if (maxGood == nGood2)
      {
         if (parallax2 > minParallax)
         {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
         }
      }
      else if (maxGood == nGood3)
      {
         if (parallax3 > minParallax)
         {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
         }
      }
      else if (maxGood == nGood4)
      {
         if (parallax4 > minParallax)
         {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
         }
      }

      // 如果有最优解但是不满足对应的parallax>minParallax，那么返回false表示求解失败
      return false;
   }

}