#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
#include <pcl/filters/extract_indices.h>

struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000;


// Net output params
struct OutputSeg
{
    int id;             // 结果类别id
    float confidence;   // 结果置信度
    cv::Rect box;       // 矩形框
    cv::Mat boxMask;    // 矩形框内mask
};

// mask params
struct MaskParams
{
    int segChannels = 32;
    int segWidth    = 160;
    int segHeight   = 160;
    int netWidth    = 640;
    int netHeight   = 640;
    float maskThreshold = 0.5;
    cv::Size srcImgShape;
    cv::Vec4d params;
};

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock;
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength];
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    // 当期帧运动畸变校正之后的激光点云
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    // 从fullCloud中提取有效点
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag;
    cv::Mat rangeMat;

    bool odomDeskewFlag;
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo;
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;

    vector<int> columnIdnCountVec;

    // ****************************************** semantic param *************************************
    // 发布带语义的点云
    ros::Publisher pubSemanticCloud;
    // 发布分割图像
    ros::Publisher pubSemanticImg;
    ros::Publisher pubCloudInfo;
    ros::Publisher pubCamViewCloud;

    // std_msgs::Header cloudHeader;
    // lio_sam::cloud_info cloudInfo;    

    const int INPUT_WIDTH = 640;
    const int INPUT_HEIGHT = 640;
    const float SCORE_THRESHOLD = 0.5;
    const float NMS_THRESHOLD = 0.45;
    const float CONFIDENCE_THRESHOLD = 0.45;

    // 分割类别名称
    std::vector<std::string> class_name;

    // 推理网络模型
    cv::dnn::Net net;

    // 存放网络推理结果
    std::vector<cv::Mat> detections;

    // 存图像放检查结果 0是图像+mask 1是mask
    std::vector<cv::Mat> imgAndMask;

    // kitti双目相机参数
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;

    cv::Mat mask;
    cv::Mat depth;

public:
    ImageProjection():
    deskewFlag(0)
    {
        std::ifstream ifs("/home/baobaizhong/catkin_ws/src/LIO-SAM/src/class_seg.txt");
        if (!ifs.is_open())
        {
            std::cerr << "Can't open file!" << std::endl;
        }
        std::string line;

        while (getline(ifs, line))
        {
            class_name.push_back(line);
        }

        net = cv::dnn::readNetFromONNX("/home/baobaizhong/catkin_ws/src/LIO-SAM/src/yolov5n-seg.onnx");
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        // subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory();
        resetParameters();

        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);

        // 消息同步
        message_filters::Subscriber<sensor_msgs::Image> subImageLeft(nh, "kitti/camera_color_left/image_raw", 1, ros::TransportHints().tcpNoDelay());
        message_filters::Subscriber<sensor_msgs::Image> subImageRight(nh, "kitti/camera_color_right/image_raw", 1, ros::TransportHints().tcpNoDelay());
        message_filters::Subscriber<sensor_msgs::PointCloud2> subCloudInfo(nh, pointCloudTopic, 1, ros::TransportHints().tcpNoDelay());
        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, sensor_msgs::Image, sensor_msgs::Image> mySyncPolicy;
        message_filters::Synchronizer<mySyncPolicy> sync(mySyncPolicy(8), subCloudInfo, subImageLeft, subImageRight);
        sync.registerCallback(boost::bind(&ImageProjection::cloudInfoImagehandler, this, _1, _2, _3));

        pubSemanticCloud    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/semantic/pointcloud", 1);
        pubSemanticImg      = nh.advertise<sensor_msgs::Image>("lio_sam/semantic/img", 1);
        pubCloudInfo        = nh.advertise<lio_sam::cloud_info>("lio_sam/semantic/cloud_info", 1);
        pubCamViewCloud     = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/semantic/camViewCloud", 1);

        ROS_INFO("\033[1;32m----> Semantic Segmentation Started.\033[0m");
        // 注意！由于作用域原因，必须放在订阅函数之后（不能放在主函数中，否则无法订阅到话题）！
        ros::MultiThreadedSpinner spinner(3);
        spinner.spin();
    }

    void allocateMemory()
    {
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        cloudInfo.startRingIndex.assign(N_SCAN, 0);
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0);
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters();
    }

    void resetParameters()
    {
        laserCloudIn->clear();
        extractedCloud->clear();
        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX));

        imuPointerCur = 0;
        firstPointFlag = true;
        odomDeskewFlag = false;

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }

        columnIdnCountVec.assign(N_SCAN, 0);
    }

    ~ImageProjection(){}

    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg);

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        if (!cachePointCloud(laserCloudMsg))
            return;

        if (!deskewInfo())
            return;

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();
    }

    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2)
            return false;

        // convert cloud
        currentCloudMsg = std::move(cloudQueue.front());
        cloudQueue.pop_front();
        if (sensor == SensorType::VELODYNE || sensor == SensorType::LIVOX)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        cloudHeader = currentCloudMsg.header;
        timeScanCur = cloudHeader.stamp.toSec();
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        if (laserCloudIn->is_dense == false)
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0;
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    bool deskewInfo()
    {
        std::lock_guard<std::mutex> lock1(imuLock);
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        imuDeskewInfo();

        odomDeskewInfo();

        return true;
    }

    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0;

        for (int i = 0; i < (int)imuQueue.size(); ++i)
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit);

            if (currentImuTime > timeScanEnd + 0.01)
                break;

            if (imuPointerCur == 0){
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur;

        if (imuPointerCur <= 0)
            return;

        cloudInfo.imuAvailable = true;
    }

    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur)
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        cloudInfo.odomAvailable = true;

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd)
            return;

        nav_msgs::Odometry endOdomMsg;

        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        float rollIncre, pitchIncre, yawIncre;
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true;
    }

    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        int imuPointerFront = 0;
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront])
                break;
            ++imuPointerFront;
        }

        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            int imuPointerBack = imuPointerFront - 1;
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    PointType deskewPoint(PointType *point, double relTime)
    {
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        if (firstPointFlag == true)
        {
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    void projectPointCloud()
    {
        //store calibration data in Opencv matrices
        cv::Mat P_rect_00(3,4,cv::DataType<double>::type);//3×4 projection matrix after rectification
        cv::Mat R_rect_00(4,4,cv::DataType<double>::type);//3×3 rectifying rotation to make image planes co-planar
        cv::Mat RT(4,4,cv::DataType<double>::type);//rotation matrix and translation vector
        
        RT.at<double>(0,0) = 7.533745e-03;RT.at<double>(0,1) = -9.999714e-01;RT.at<double>(0,2) = -6.166020e-04;RT.at<double>(0,2) = -4.069766e-03;
        RT.at<double>(1,0) = 1.480249e-02;RT.at<double>(1,1) = 7.280733e-04;RT.at<double>(1,2) = -9.998902e-01;RT.at<double>(1,3) = -7.631618e-02;
        RT.at<double>(2,0) = 9.998621e-01;RT.at<double>(2,1) = 7.523790e-03;RT.at<double>(2,2) = 1.480755e-02;RT.at<double>(2,3) = -2.717806e-01;
        RT.at<double>(3,0) = 0.0;RT.at<double>(3,1) = 0.0;RT.at<double>(3,2) = 0.0;RT.at<double>(3,3) = 1.0;
        
        R_rect_00.at<double>(0,0) = 9.999239e-01;R_rect_00.at<double>(0,1) = 9.837760e-03;R_rect_00.at<double>(0,2) = -7.445048e-03;R_rect_00.at<double>(0,3) = 0.0;
        R_rect_00.at<double>(1,0) = -9.869795e-03;R_rect_00.at<double>(1,1) = 9.999421e-01;R_rect_00.at<double>(1,2) = -4.278459e-03;R_rect_00.at<double>(1,3) = 0.0;
        R_rect_00.at<double>(2,0) = 7.402527e-03;R_rect_00.at<double>(2,1) = 4.351614e-03;R_rect_00.at<double>(2,2) = 9.999631e-01;R_rect_00.at<double>(2,3) = 0.0;
        R_rect_00.at<double>(3,0) = 0.0;R_rect_00.at<double>(3,1) = 0.0;R_rect_00.at<double>(3,2) = 0.0;R_rect_00.at<double>(3,3) = 1.0;

        P_rect_00.at<double>(0,0) = 7.215377e+02;P_rect_00.at<double>(0,1) = 0.000000e+00;P_rect_00.at<double>(0,2) = 6.095593e+02;P_rect_00.at<double>(0,3) = 0.000000e+00;
        P_rect_00.at<double>(1,0) = 0.000000e+00;P_rect_00.at<double>(1,1) = 7.215377e+02;P_rect_00.at<double>(1,2) = 1.728540e+02;P_rect_00.at<double>(1,3) = 0.000000e+00;
        P_rect_00.at<double>(2,0) = 0.000000e+00;P_rect_00.at<double>(2,1) = 0.000000e+00;P_rect_00.at<double>(2,2) = 1.000000e+00;P_rect_00.at<double>(2,3) = 0.000000e+00;
        cv::Mat X(4, 1, cv::DataType<double>::type);
        cv::Mat Y(4, 1, cv::DataType<double>::type);
        pcl::PointIndices::Ptr deleteIndices(new pcl::PointIndices());
        pcl::ExtractIndices<PointXYZIRT> extract;
        int cloudSize = laserCloudIn->points.size();
        for (int i = 0; i < cloudSize; i++)
        {
            PointXYZIRT point(laserCloudIn->points[i]);
            if (point.x < 0.0)
            {
                deleteIndices->indices.push_back(i);
                continue;
            } 

            X.at<double>(0,0) = point.x;
            X.at<double>(1,0) = point.y;
            X.at<double>(2,0) = point.z;
            X.at<double>(3,0) = point.intensity;

            //apply the projection equation to map X onto the image plane of the camera. Store the result in Y
            Y=P_rect_00*R_rect_00*RT*X;
            // transform Y back into Euclidean coordinates and store the result in the variable pt
            cv::Point pt;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

            if (pt.x < 0 || pt.x > 1225 || pt.y < 0 || pt.y > 369)
            {
                continue;
            };
            // camViewCloud->points.push_back(*iter);
            int cls = mask.at<uchar>(int(pt.y), int(pt.x));
            float depthVal = depth.at<float>(int(pt.y), int(pt.x));
            // 根据lidar点云深度与相机点云深度过滤一部分深度不匹配的点云
            if (cls != 255) {
                // point.intensity = cls;
                deleteIndices->indices.push_back(i);
                // cloudOut->points.push_back(point);
            }
        }
        extract.setInputCloud(laserCloudIn);
        extract.setIndices(deleteIndices);
        extract.setNegative(true);
        extract.filter(*laserCloudIn);



        cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;
            // if(thisPoint.x < 0) continue;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            int columnIdn = -1;
            if (sensor == SensorType::VELODYNE || sensor == SensorType::OUSTER)
            {
                float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
                static float ang_res_x = 360.0/float(Horizon_SCAN);
                columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;
                if (columnIdn >= Horizon_SCAN)
                    columnIdn -= Horizon_SCAN;
            }
            else if (sensor == SensorType::LIVOX)
            {
                columnIdn = columnIdnCountVec[rowIdn];
                columnIdnCountVec[rowIdn] += 1;
            }
            
            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            rangeMat.at<float>(rowIdn, columnIdn) = range;

            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    
    }

    void cloudExtraction()
    {
        int count = 0;
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    cloudInfo.pointColInd[count] = j;
                    // save range info
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);
                    // save extracted cloud
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);
                    // size of extracted cloud
                    ++count;
                }
            }
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        cloudInfo.cloud_deskewed  = publishCloud(pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        pubLaserCloudInfo.publish(cloudInfo);
    }

    void cloudInfoImagehandler(const sensor_msgs::PointCloud2ConstPtr& msgIn, const sensor_msgs::ImageConstPtr& imgLeft, const sensor_msgs::ImageConstPtr& imgRight)
    {
        if (!cachePointCloud(msgIn))
            return;

        if (!deskewInfo())
            return;

        detections.clear();
        imgAndMask.clear();
        // process left image and detectection(segmantation)
        cv_bridge::CvImagePtr cvPtrLeft = cv_bridge::toCvCopy(imgLeft, sensor_msgs::image_encodings::TYPE_8UC3);
        cv_bridge::CvImagePtr cvPtrRight = cv_bridge::toCvCopy(imgRight, sensor_msgs::image_encodings::TYPE_8UC3);
        cv::Mat imageLeft = cvPtrLeft->image;
        cv::Mat imageRight = cvPtrRight->image;
        cv::Mat blob;
        cv::Vec4d params;
        preProcess(imageLeft, blob, params);
        process(blob, net, detections);
        imgAndMask = post_process(imageLeft, detections, class_name, params);
        sensor_msgs::ImagePtr msgImg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgAndMask[0]).toImageMsg();
        pubSemanticImg.publish(msgImg);
        //process left and right image to get stereoVision
        //to gray image
        cv::Mat leftGray, rightGray;
        cv::cvtColor(imageLeft, leftGray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imageRight, rightGray, cv::COLOR_BGR2GRAY);
        cv::Mat imgDepth(leftGray.rows, leftGray.cols, CV_32F, cv::Scalar(0.));
        cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(
        0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32);    // 神奇的参数
        cv::Mat disparity_sgbm, disparity;
        sgbm->compute(leftGray, rightGray, disparity_sgbm);
        disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
        imgDepth = fx * b / disparity;
        depth = imgDepth;

        // process lidar point
        cloudHeader = msgIn->header; // new cloud header
        mask = imgAndMask[1];

        projectPointCloud();

        cloudExtraction();

        publishClouds();

        resetParameters();     
    }

    // LetterBox处理
    void letterBox(const cv::Mat& image, cv::Mat& outImage,
                    cv::Vec4d& params,
                    const cv::Size& newShape = cv::Size(640, 640),
                    bool autoShape = false,
                    bool scaleFill = false,
                    bool scaleUp = true,
                    int stride = 32,
                    const cv::Scalar& color = cv::Scalar(114, 114, 114))
    {
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
        if (!scaleUp)
        {
            r = std::min(r, 1.0f);
        }

        float ratio[2]{ r, r };
        int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

        auto dw = (float)(newShape.width - new_un_pad[0]);
        auto dh = (float)(newShape.height - new_un_pad[1]);

        if (autoShape)
        {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        }
        else if (scaleFill)
        {
            dw = 0.0f;
            dh = 0.0f;
            new_un_pad[0] = newShape.width;
            new_un_pad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
            cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
        else 
            outImage = image.clone();

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        params[0] = ratio[0];
        params[1] = ratio[1];
        params[2] = left;
        params[3] = top;
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }
    
    // 预处理
    void preProcess(cv::Mat& image, cv::Mat& blob, cv::Vec4d& params)
    {
        cv::Mat input_image;
        letterBox(image, input_image, params, cv::Size(INPUT_WIDTH, INPUT_HEIGHT));
        cv::dnn::blobFromImage(input_image, blob, 1. / 255., cv::Size(INPUT_WIDTH, INPUT_HEIGHT), cv::Scalar(0, 0, 0), true, false);
    }

    // 网络推理
    void process(cv::Mat& blob, cv::dnn::Net& net, std::vector<cv::Mat>& outputs)
    {
        // std::lock_guard<std::mutex> lock(semanLock);
        
        if (net.empty())
        {
            std::cerr << "Net is empty" << std::endl;
        }
        
        if (blob.empty())
        {
            std::cerr << "Input is empty" << std::endl;
        }
        // int n, c, h, w, id;
        // //四维数据的访问为：
        // for (n = 0; n < 1; n++)
        // {
        //     for (c = 0; c < 3; c++)
        //     {
        //         for (h = 0; h < 640; h++)
        //         {
        //             for (w = 0; w < 640; w++)
        //             {
        //                 id = blob.step[0] * n + blob.step[1] * c + blob.step[2] * h + w * blob.step[3];
        //                 //cout << id << endl;
        //                 float *p = (float*)(blob.data + id);
        //                 cout << *p << endl;
        //             }
        //         }
        //     }
        // }
        net.setInput(blob);
        std::vector<std::string> output_layer_names{ "output0","output1" };
        net.enableWinograd(false);
        net.forward(outputs, output_layer_names);
    }

    //取得掩膜
    void getMask(const cv::Mat& maskProposals, const cv::Mat& mask_protos, OutputSeg& output, const MaskParams& maskParams)
    {
        int seg_channels = maskParams.segChannels;
        int net_width = maskParams.netWidth;
        int seg_width = maskParams.segWidth;
        int net_height = maskParams.netHeight;
        int seg_height = maskParams.segHeight;
        float mask_threshold = maskParams.maskThreshold;
        cv::Vec4f params = maskParams.params;
        cv::Size src_img_shape = maskParams.srcImgShape;
        cv::Rect temp_rect = output.box;

        //crop from mask_protos
        int rang_x = floor((temp_rect.x * params[0] + params[2]) / net_width * seg_width);
        int rang_y = floor((temp_rect.y * params[1] + params[3]) / net_height * seg_height);
        int rang_w = ceil(((temp_rect.x + temp_rect.width) * params[0] + params[2]) / net_width * seg_width) - rang_x;
        int rang_h = ceil(((temp_rect.y + temp_rect.height) * params[1] + params[3]) / net_height * seg_height) - rang_y;

        rang_w = MAX(rang_w, 1);
        rang_h = MAX(rang_h, 1);
        if (rang_x + rang_w > seg_width)
        {
            if (seg_width - rang_x > 0)
                rang_w = seg_width - rang_x;
            else
                rang_x -= 1;
        }
        if (rang_y + rang_h > seg_height)
        {
            if (seg_height - rang_y > 0)
                rang_h = seg_height - rang_y;
            else
                rang_y -= 1;
        }

        std::vector<cv::Range> roi_rangs;
        roi_rangs.push_back(cv::Range(0, 1));
        roi_rangs.push_back(cv::Range::all());
        roi_rangs.push_back(cv::Range(rang_y, rang_h + rang_y));
        roi_rangs.push_back(cv::Range(rang_x, rang_w + rang_x));

        //crop
        cv::Mat temp_mask_protos = mask_protos(roi_rangs).clone();
        cv::Mat protos = temp_mask_protos.reshape(0, { seg_channels,rang_w * rang_h });
        cv::Mat matmul_res = (maskProposals * protos).t();
        cv::Mat masks_feature = matmul_res.reshape(1, { rang_h,rang_w });
        cv::Mat dest, mask;

        //sigmoid
        cv::exp(-masks_feature, dest);
        dest = 1.0 / (1.0 + dest);

        int left = floor((net_width / seg_width * rang_x - params[2]) / params[0]);
        int top = floor((net_height / seg_height * rang_y - params[3]) / params[1]);
        int width = ceil(net_width / seg_width * rang_w / params[0]);
        int height = ceil(net_height / seg_height * rang_h / params[1]);

        cv::resize(dest, mask, cv::Size(width, height), cv::INTER_NEAREST);
        mask = mask(temp_rect - cv::Point(left, top)) > mask_threshold;
        output.boxMask = mask;
    }

    // 获取分割后的图像以及掩膜
    std::vector<cv::Mat> drawResult(cv::Mat & image, std::vector<OutputSeg> result, std::vector<std::string> class_name)
    {
        std::vector<cv::Scalar> color;
        srand(time(0));
        for (int i = 0; i < class_name.size(); i++)
        {
            color.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
        }

        cv::Mat mask = image.clone();
        cv::Mat blankImg(image.rows, image.cols, CV_8UC1, cv::Scalar(255)); // modify
        for (int i = 0; i < result.size(); i++)
        {
            cv::rectangle(image, result[i].box, cv::Scalar(255, 0, 0), 2);
            mask(result[i].box).setTo(color[result[i].id], result[i].boxMask);
            // std::cout << result[i].boxMask.size << std::endl;   // modify
            blankImg(result[i].box).setTo(result[i].id, result[i].boxMask); // modify
            std::string label = class_name[result[i].id] + ":" + cv::format("%.2f", result[i].confidence);
            int baseLine;
            cv::Size label_size = cv::getTextSize(label, 0.8, 0.8, 1, &baseLine);
            cv::putText(image, label, cv::Point(result[i].box.x, result[i].box.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, color[result[i].id], 1);
        }
        // cv::imshow("mask", blankImg);   // modify
        // cv::imwrite("mask_007000_seg.png", blankImg);
        addWeighted(image, 0.5, mask, 0.5, 0, image);
        return {image, blankImg};
    }
    //后处理
    std::vector<cv::Mat> post_process(cv::Mat& image, std::vector<cv::Mat>& outputs, const std::vector<std::string>& class_name, cv::Vec4d& params)
    {
        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;
        std::vector<std::vector<float>> picked_proposals;

        float* data = (float*)outputs[0].data;

        const int dimensions = 117;	//5+80+32
        const int rows = 25200; 	//(640/8)*(640/8)*3+(640/16)*(640/16)*3+(640/32)*(640/32)*3
        for (int i = 0; i < rows; ++i)
        {
            float confidence = data[4];
            if (confidence >= CONFIDENCE_THRESHOLD)
            {
                float* classes_scores = data + 5;
                cv::Mat scores(1, class_name.size(), CV_32FC1, classes_scores);
                cv::Point class_id;
                double max_class_score;
                cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
                if (max_class_score > SCORE_THRESHOLD)
                {
                    float x = (data[0] - params[2]) / params[0];  
                    float y = (data[1] - params[3]) / params[1]; 
                    float w = data[2] / params[0];
                    float h = data[3] / params[1];
                    int left = std::max(int(x - 0.5 * w), 0);
                    int top = std::max(int(y - 0.5 * h), 0);
                    int width = int(w);
                    int height = int(h);
                    boxes.push_back(cv::Rect(left, top, width, height));
                    confidences.push_back(confidence);
                    class_ids.push_back(class_id.x);

                    std::vector<float> temp_proto(data + class_name.size() + 5, data + dimensions);
                    picked_proposals.push_back(temp_proto);
                }
            }
            data += dimensions;
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, SCORE_THRESHOLD, NMS_THRESHOLD, indices);

        std::vector<OutputSeg> output;
        std::vector<std::vector<float>> temp_mask_proposals;
        cv::Rect holeImgRect(0, 0, image.cols, image.rows);
        for (int i = 0; i < indices.size(); ++i) 
        {
            int idx = indices[i];
            OutputSeg result;
            result.id = class_ids[idx];
            result.confidence = confidences[idx];
            result.box = boxes[idx] & holeImgRect;
            temp_mask_proposals.push_back(picked_proposals[idx]);
            output.push_back(result);
        }

        MaskParams mask_params;
        mask_params.params = params;
        mask_params.srcImgShape = image.size();
        for (int i = 0; i < temp_mask_proposals.size(); ++i)
        {
            getMask(cv::Mat(temp_mask_proposals[i]).t(), outputs[1], output[i], mask_params);
        }
        std::vector<cv::Mat> imgAndMask;
        imgAndMask = drawResult(image, output, class_name);

        return imgAndMask;
    }

    // lidar点云语义获取
    void getLidarSemantic(cv::Mat img, cv::Mat imgDepth, pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn, pcl::PointCloud<pcl::PointXYZI>::Ptr cloudOut)
    {
        //store calibration data in Opencv matrices
        cv::Mat P_rect_00(3,4,cv::DataType<double>::type);//3×4 projection matrix after rectification
        cv::Mat R_rect_00(4,4,cv::DataType<double>::type);//3×3 rectifying rotation to make image planes co-planar
        cv::Mat RT(4,4,cv::DataType<double>::type);//rotation matrix and translation vector
        
        RT.at<double>(0,0) = 7.533745e-03;RT.at<double>(0,1) = -9.999714e-01;RT.at<double>(0,2) = -6.166020e-04;RT.at<double>(0,2) = -4.069766e-03;
        RT.at<double>(1,0) = 1.480249e-02;RT.at<double>(1,1) = 7.280733e-04;RT.at<double>(1,2) = -9.998902e-01;RT.at<double>(1,3) = -7.631618e-02;
        RT.at<double>(2,0) = 9.998621e-01;RT.at<double>(2,1) = 7.523790e-03;RT.at<double>(2,2) = 1.480755e-02;RT.at<double>(2,3) = -2.717806e-01;
        RT.at<double>(3,0) = 0.0;RT.at<double>(3,1) = 0.0;RT.at<double>(3,2) = 0.0;RT.at<double>(3,3) = 1.0;
        
        R_rect_00.at<double>(0,0) = 9.999239e-01;R_rect_00.at<double>(0,1) = 9.837760e-03;R_rect_00.at<double>(0,2) = -7.445048e-03;R_rect_00.at<double>(0,3) = 0.0;
        R_rect_00.at<double>(1,0) = -9.869795e-03;R_rect_00.at<double>(1,1) = 9.999421e-01;R_rect_00.at<double>(1,2) = -4.278459e-03;R_rect_00.at<double>(1,3) = 0.0;
        R_rect_00.at<double>(2,0) = 7.402527e-03;R_rect_00.at<double>(2,1) = 4.351614e-03;R_rect_00.at<double>(2,2) = 9.999631e-01;R_rect_00.at<double>(2,3) = 0.0;
        R_rect_00.at<double>(3,0) = 0.0;R_rect_00.at<double>(3,1) = 0.0;R_rect_00.at<double>(3,2) = 0.0;R_rect_00.at<double>(3,3) = 1.0;

        P_rect_00.at<double>(0,0) = 7.215377e+02;P_rect_00.at<double>(0,1) = 0.000000e+00;P_rect_00.at<double>(0,2) = 6.095593e+02;P_rect_00.at<double>(0,3) = 0.000000e+00;
        P_rect_00.at<double>(1,0) = 0.000000e+00;P_rect_00.at<double>(1,1) = 7.215377e+02;P_rect_00.at<double>(1,2) = 1.728540e+02;P_rect_00.at<double>(1,3) = 0.000000e+00;
        P_rect_00.at<double>(2,0) = 0.000000e+00;P_rect_00.at<double>(2,1) = 0.000000e+00;P_rect_00.at<double>(2,2) = 1.000000e+00;P_rect_00.at<double>(2,3) = 0.000000e+00;
        cv::Mat X(4, 1, cv::DataType<double>::type);
        cv::Mat Y(4, 1, cv::DataType<double>::type);
        int cloudSize = cloudIn->points.size();
        for (int i = 0; i < cloudSize; i++)
        {
            pcl::PointXYZI point(cloudIn->points[i]);
            if (point.x < 0.0)
            {
                continue;
            } 

            X.at<double>(0,0) = point.x;
            X.at<double>(1,0) = point.y;
            X.at<double>(2,0) = point.z;
            X.at<double>(3,0) = point.intensity;

            //apply the projection equation to map X onto the image plane of the camera. Store the result in Y
            Y=P_rect_00*R_rect_00*RT*X;
            // transform Y back into Euclidean coordinates and store the result in the variable pt
            cv::Point pt;
            pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
            pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);

            if (pt.x < 0 || pt.x > 1225 || pt.y < 0 || pt.y > 369)
            {
                continue;
            };
            // camViewCloud->points.push_back(*iter);
            int cls = img.at<uchar>(int(pt.y), int(pt.x));
            float depth = imgDepth.at<float>(int(pt.y), int(pt.x));
            // 根据lidar点云深度与相机点云深度过滤一部分深度不匹配的点云
            if (cls != 255 && abs(depth - point.x) < 1.5) {
                // std::cout << "debug file:" << __FILE__ << " line: " << __LINE__ << std::endl;
                // std::cout << "image depth: " << depth << " lidar depth: " << point.x << " diff: " << abs(depth - point.x) << std::endl; 
                point.intensity = cls;
                cloudOut->points.push_back(point);
            }
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");

    // ros::MultiThreadedSpinner spinner(3);
    // spinner.spin();
    
    return 0;
}
