#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "message_filters/subscriber.h"
#include "message_filters/time_synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"
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

class MapSemantic : public ParamServer
{
private:
    // 订阅cloudinfo和用于检测的图像
    ros::Subscriber subCloudInfo;
    ros::Subscriber subImg;

    // 发布带语义的点云
    ros::Publisher pubSemanticCloud;
    // 发布分割图像
    ros::Publisher pubSemanticImg;
    ros::Publisher pubCloudInfo;

    std_msgs::Header cloudHeader;
    lio_sam::cloud_info cloudInfo;
    

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

    std::mutex lidarLock;
    std::mutex imgLock;

    Eigen::Matrix4d lidarCamExtrin;
    Eigen::Affine3d transformA = Eigen::Affine3d::Identity();
    

public:
    // 构造函数
    MapSemantic()
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

        // lidar -> cam params
        lidarCamExtrin << lidar2camExtrinsic(0, 0), lidar2camExtrinsic(0, 1), lidar2camExtrinsic(0, 2), lidar2camExtrinsic(0, 3),
                          lidar2camExtrinsic(1, 0), lidar2camExtrinsic(1, 1), lidar2camExtrinsic(1, 2), lidar2camExtrinsic(1, 3),
                          lidar2camExtrinsic(2, 0), lidar2camExtrinsic(2, 1), lidar2camExtrinsic(2, 2), lidar2camExtrinsic(2, 3), 0.0, 0.0, 0.0, 1.0;
        Eigen::Matrix4d t;
        t = lidarCamExtrin.inverse();
        transformA.matrix() <<  t(0, 0), t(0, 1), t(0, 2), t(0, 3),
                                t(1, 0), t(1, 1), t(1, 2), t(1, 3),
                                t(2, 0), t(2, 1), t(2, 2), t(2, 3), 
                                t(3, 0), t(3, 1), t(3, 2), t(3, 3);


        // 消息同步
        message_filters::Subscriber<sensor_msgs::Image> subImageLeft(nh, "camera/color/image_raw", 1);
        message_filters::Subscriber<sensor_msgs::Image> subImageRight(nh, "camera/aligned_depth_to_color/image_raw", 1);
        message_filters::Subscriber<lio_sam::cloud_info> subCloudInfo(nh, "lio_sam/deskew/cloud_info", 1);
        typedef message_filters::sync_policies::ApproximateTime<lio_sam::cloud_info, sensor_msgs::Image, sensor_msgs::Image> mySyncPolicy;
        message_filters::Synchronizer<mySyncPolicy> sync(mySyncPolicy(10), subCloudInfo, subImageLeft, subImageRight);
        sync.registerCallback(boost::bind(&MapSemantic::cloudInfoImagehandler, this, _1, _2, _3));

        // std::cout << "debug file: " <<__FILE__ << " lin: " << __LINE__ << std::endl;

        pubSemanticCloud    = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/semantic/pointcloud", 1);
        pubSemanticImg      = nh.advertise<sensor_msgs::Image>("lio_sam/semantic/img", 1);
        pubCloudInfo        = nh.advertise<lio_sam::cloud_info>("lio_sam/semantic/cloud_info", 1);

        ROS_INFO("\033[1;32m----> Semantic Segmentation Started.\033[0m");
        // 注意！由于作用域原因，必须放在订阅函数之后（不能放在主函数中，否则无法订阅到话题）！
        ros::spin();

    }


    // const sensor_msgs::ImageConstPtr& imgRight
    void cloudInfoImagehandler(const lio_sam::cloud_infoConstPtr msgIn, const sensor_msgs::ImageConstPtr& imgColor, const sensor_msgs::ImageConstPtr& imgDepth)
    {
        // std::cout << "debug file: " << __FILE__ << " " << "line: " << __LINE__ << std::endl;
        detections.clear();
        imgAndMask.clear();
        cloudInfo = *msgIn;

        // process left image and detectection(segmantation)
        cv_bridge::CvImagePtr cvPtrColor = cv_bridge::toCvCopy(imgColor, sensor_msgs::image_encodings::TYPE_8UC3);
        cv_bridge::CvImagePtr cvPtrDepth = cv_bridge::toCvCopy(imgDepth, sensor_msgs::image_encodings::TYPE_32FC1);
        cv::Mat imageColor = cvPtrColor->image;
        cv::Mat imageDepth = cvPtrDepth->image;
        cv::Mat blob;
        cv::Vec4d params;
        preProcess(imageColor, blob, params);
        process(blob, net, detections);
        imgAndMask = post_process(imageColor, detections, class_name, params);
        sensor_msgs::ImagePtr msgImg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgAndMask[0]).toImageMsg();
        pubSemanticImg.publish(msgImg);

        // process lidar point
        cloudHeader = msgIn->header; // new cloud header
        cv::Mat mask = imgAndMask[1];
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZI>), cloudOut(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(cloudInfo.cloud_deskewed, *cloudIn);

        getLidarSemantic(mask, imageDepth, cloudIn, cloudOut);      
    
        cloudInfo.cloud_semantic = publishCloud(pubSemanticCloud, cloudOut, cloudHeader.stamp, lidarFrame);
        pubCloudInfo.publish(cloudInfo);
    }
    
    void cloudInfoHandler(const lio_sam::cloud_infoConstPtr& msgIn)
    {
        cloudInfo = *msgIn;
        std::cout << "debug file:" << __FILE__ << " " << "line: " << __LINE__ << std::endl;
        std::cout << "cloudInfo time: " << msgIn->header.stamp.sec << std::endl;
        
        std::lock_guard<std::mutex> lock1(lidarLock);
        cloudHeader = msgIn->header; // new cloud header
        cv::Mat mask = imgAndMask[1];
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudIn(new pcl::PointCloud<pcl::PointXYZI>), cloudOut(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(msgIn->cloud_deskewed, *cloudIn);
        // getLidarSemantic(mask, cloudIn, cloudOut);
    
        cloudInfo.cloud_semantic = publishCloud(pubSemanticCloud, cloudOut, cloudHeader.stamp, lidarFrame);
        pubCloudInfo.publish(cloudInfo);
    }

    void imgHandler(const sensor_msgs::ImageConstPtr& img)
    {
        std::lock_guard<std::mutex> lock2(imgLock);
        std::cout << "debug file:" << __FILE__ << " " << "line: " << __LINE__ << std::endl;
        std::cout << "img time: " << img->header.stamp.sec << std::endl;
        detections.clear();
        imgAndMask.clear();
        // sensor_msg::Image 转OpenCV
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::TYPE_8UC3);
        cv::Mat image = cv_ptr->image, blob;
        cv::Vec4d params;
        preProcess(image, blob, params);
        process(blob, net, detections);
        imgAndMask = post_process(image, detections, class_name, params);
        sensor_msgs::ImagePtr msgImg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", imgAndMask[0]).toImageMsg();
        pubSemanticImg.publish(msgImg);
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


        int cloudSize = cloudIn->points.size();
        for (int i = 0; i < cloudSize; ++i)
        {
            pcl::PointXYZI pointIn(cloudIn->points[i]);
            pcl::PointXYZI point;
            point = pcl::transformPoint(pointIn,transformA);

            if (point.z < 0.0) continue;

            double xC = point.x / point.z;
            double yC = point.y / point.z;
            double zC = point.z;

            cv::Point pt;
            pt.x = camIntrinsic(0, 0) * xC + camIntrinsic(0, 2);
            pt.y = camIntrinsic(1, 1) * yC + camIntrinsic(1, 2);

            if (pt.x < 0 || pt.x > 639 || pt.y < 0 || pt.y > 479) continue;
            int cls = img.at<uchar>(int(pt.y), int(pt.x));
            float depth = imgDepth.at<float>(int(pt.y), int(pt.x)) / 1000.0;
            // 根据lidar点云深度与相机点云深度过滤一部分深度不匹配的点云
            if (cls != 255 && abs(depth - point.z) < 0.5) {
                std::cout << "debug file:" << __FILE__ << " line: " << __LINE__ << std::endl;
                std::cout << "image depth: " << depth << " lidar depth: " << point.z << " diff: " << abs(depth - point.z) << std::endl; 
                pointIn.intensity = cls / 80.0;
                cloudOut->points.push_back(pointIn);
            }
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    MapSemantic MS;

    // ROS_INFO("\033[1;32m----> Semantic Segmentation Started.\033[0m");

    // ros::spin();

    return 0;
}