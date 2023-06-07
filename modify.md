1. mapSemantic.cpp      
订阅imageProjection.cpp的去畸变后的点云，以及来自相机的x图像，进行雷达语义提取，并发布cloudInfo, 带语义信息的lidar cloudpoint， 语义分割(目标检查后)的图像      
`lio_sam/semantic/cloudInfo`       
`lio_sam/semantic/img`     
`lio_sam/semantic/cloudPoint`
2. cloud_info.msg       
增加`sensor_msg::PointCloud2 cloud_semantic`        
3. featureExtraction.cpp       
将原来的订阅来自`imgageProjection.cpp`的`cloudInfo`修改为来自`mapsemantic.cpp`的`lio_sam/semantic/cloudInfo`
4. mapoptimization.cpp      
在`publishFrames()`函数中，添加发布坐标变换到全局坐标系下的语义点云。