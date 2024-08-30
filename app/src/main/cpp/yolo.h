// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef YOLO_H
#define YOLO_H

#include <opencv2/core/core.hpp>

#include <net.h>

struct Object {
    cv::Rect_<float> rect;
    int label;
    float prob;
    cv::Mat mask;
    std::vector<float> mask_feat;
};
struct GridAndStride {
    int grid0;
    int grid1;
    int stride;
};

class Yolo {
public:
    Yolo();

    /**
     * Yolo当前状态
     * <br>---------------<br>
     * 0 - 暂停 <br>
     * 1 - 分类 <br>
     * 2 - 分割 <br>
     * 3 - 检测 <br>
     * 4 - 绘制 <br>
     * */
    int j_state = 0;

    int load(AAssetManager *mgr,
             const char *model_type,
             int target_size,
             const float *mean_values,
             const float *norm_values,
             bool use_classify,
             bool use_segmentation,
             bool use_detect,
             bool use_gpu = false);

    void initNativeCallback(JavaVM *vm, jlong nativeObjAddr, jobject pJobject);

    /**
     * 分类
     * */
    int classify(const cv::Mat &rgb);

    /**
     * 分割
     * */
    int segmentation(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold = 0.4f,
                     float nms_threshold = 0.5f);

    /**
     * 检测
     * */
    int detect(const cv::Mat &rgb,
               std::vector<Object> &objects,
               float prob_threshold = 0.4f,
               float nms_threshold = 0.5f);

    /**
     * 绘制
     * */
    int draw(cv::Mat &rgb, const std::vector<Object> &objects);

    int draw_mask(cv::Mat &rgb, const std::vector<Object> &objects);

private:
    //分类
    ncnn::Net yolo_c;
    //分割
    ncnn::Net yolo_s;
    //检测
    ncnn::Net yolo_d;
    int target_size;
    float mean_values[3];
    float norm_values[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    /**
     * 全局引用
     * */
    JavaVM *javaVM;
    //Java传过来的Mat对象内存地址
    jlong j_mat_addr;
    //回调类
    jobject j_callback;
};

#endif
