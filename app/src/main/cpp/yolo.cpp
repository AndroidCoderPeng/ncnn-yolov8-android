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

#include "yolo.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>

#include "cpu.h"

static float fast_exp(float x) {
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

static float sigmoid(float x) {
    return 1.0f / (1.0f + fast_exp(-x));
}

static float intersection_area(const Object &a, const Object &b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object> &face_objects, int left, int right) {
    int i = left;
    int j = right;
    float p = face_objects[(left + right) / 2].prob;

    while (i <= j) {
        while (face_objects[i].prob > p)
            i++;

        while (face_objects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(face_objects[i], face_objects[j]);

            i++;
            j--;
        }
    }

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(face_objects, left, j);
        }
        //         #pragma omp section
        {
            if (i < right) qsort_descent_inplace(face_objects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object> &face_objects) {
    if (face_objects.empty())
        return;

    qsort_descent_inplace(face_objects, 0, face_objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object> &face_objects, std::vector<int> &picked,
                              float nms_threshold) {
    picked.clear();

    const int n = face_objects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++) {
        areas[i] = face_objects[i].rect.width * face_objects[i].rect.height;
    }

    for (int i = 0; i < n; i++) {
        const Object &a = face_objects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++) {
            const Object &b = face_objects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static void
generate_grids_and_stride(const int target_w, const int target_h, std::vector<int> &strides,
                          std::vector<GridAndStride> &grid_strides) {
    for (int stride: strides) {
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++) {
            for (int g0 = 0; g0 < num_grid_w; g0++) {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat &pred,
                               float prob_threshold, std::vector<Object> &objects, int num_class) {
    const int num_points = grid_strides.size();
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++) {
        const float *scores = pred.row(i) + 4 * reg_max_1;

        // find label with max score
        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++) {
            float confidence = scores[k];
            if (confidence > score) {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);
        if (box_prob >= prob_threshold) {
            ncnn::Mat bbox_pred(reg_max_1, 4, (void *) pred.row(i));
            {
                ncnn::Layer *softmax = ncnn::create_layer("Softmax");

                ncnn::ParamDict pd;
                pd.set(0, 1); // axis
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);

                softmax->forward_inplace(bbox_pred, opt);

                softmax->destroy_pipeline(opt);

                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++) {
                float dis = 0.f;
                const float *dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++) {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

/***模型分割*************/
static void matmul(const std::vector<ncnn::Mat> &bottom_blobs, ncnn::Mat &top_blob) {
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("MatMul");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// axis

    op->load_param(pd);

    op->create_pipeline(opt);
    std::vector<ncnn::Mat> top_blobs(1);
    op->forward(bottom_blobs, top_blobs, opt);
    top_blob = top_blobs[0];

    op->destroy_pipeline(opt);

    delete op;
}

static void sigmoid(ncnn::Mat &bottom) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Sigmoid");

    op->create_pipeline(opt);

    // forward

    op->forward_inplace(bottom, opt);
    op->destroy_pipeline(opt);

    delete op;
}

static void reshape(const ncnn::Mat &in, ncnn::Mat &out, int c, int h, int w, int d) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Reshape");

    // set param
    ncnn::ParamDict pd;

    pd.set(0, w);// start
    pd.set(1, h);// end
    if (d > 0)
        pd.set(11, d);//axes
    pd.set(2, c);//axes
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void slice(const ncnn::Mat &in, ncnn::Mat &out, int start, int end, int axis) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Crop");

    // set param
    ncnn::ParamDict pd;

    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(axis);
    ncnn::Mat ends = ncnn::Mat(1);
    ends.fill(end);
    ncnn::Mat starts = ncnn::Mat(1);
    starts.fill(start);
    pd.set(9, starts);// start
    pd.set(10, ends);// end
    pd.set(11, axes);//axes

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void interp(const ncnn::Mat &in, const float &scale, const int &out_w, const int &out_h,
                   ncnn::Mat &out) {
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer *op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// resize_type
    pd.set(1, scale);// height_scale
    pd.set(2, scale);// width_scale
    pd.set(3, out_h);// height
    pd.set(4, out_w);// width

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}

static void decode_mask(const ncnn::Mat &mask_feat, const int &img_w, const int &img_h,
                        const ncnn::Mat &mask_proto, const ncnn::Mat &in_pad, const int &wpad,
                        const int &hpad, ncnn::Mat &mask_pred_result) {
    ncnn::Mat masks;
    matmul(std::vector<ncnn::Mat>{mask_feat, mask_proto}, masks);
    sigmoid(masks);
    reshape(masks, masks, masks.h, in_pad.h / 4, in_pad.w / 4, 0);
    slice(masks, mask_pred_result, (wpad / 2) / 4, (in_pad.w - wpad / 2) / 4, 2);
    slice(mask_pred_result, mask_pred_result, (hpad / 2) / 4, (in_pad.h - hpad / 2) / 4, 1);
    interp(mask_pred_result, 4.0, img_w, img_h, mask_pred_result);
}

/***模型分割*************/

Yolo::Yolo() {
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int
Yolo::load(AAssetManager *mgr, const char *model_type, int _target_size, const float *_mean_values,
           const float *_norm_values, bool use_classify, bool use_segmentation, bool use_detect,
           bool use_gpu) {
    if (use_classify) {
        yolo_c.clear();
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_c.clear()");
    }
    if (use_segmentation) {
        yolo_s.clear();
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_s.clear()");
    }
    if (use_detect) {
        yolo_d.clear();
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_d.clear()");
    }

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo_c.opt = ncnn::Option();
    yolo_s.opt = ncnn::Option();
    yolo_d.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo_c.opt.use_vulkan_compute = use_gpu;
    yolo_s.opt.use_vulkan_compute = use_gpu;
    yolo_d.opt.use_vulkan_compute = use_gpu;
#endif
    yolo_c.opt.num_threads = ncnn::get_big_cpu_count();
    yolo_c.opt.blob_allocator = &blob_pool_allocator;
    yolo_c.opt.workspace_allocator = &workspace_pool_allocator;

    yolo_s.opt.num_threads = ncnn::get_big_cpu_count();
    yolo_s.opt.blob_allocator = &blob_pool_allocator;
    yolo_s.opt.workspace_allocator = &workspace_pool_allocator;

    yolo_d.opt.num_threads = ncnn::get_big_cpu_count();
    yolo_d.opt.blob_allocator = &blob_pool_allocator;
    yolo_d.opt.workspace_allocator = &workspace_pool_allocator;

    char param_path[256];
    char model_path[256];
    //拼接模型名（路径）
    sprintf(param_path, "%s.param", model_type);
    sprintf(model_path, "%s.bin", model_type);

    if (use_classify) {
        yolo_c.load_param(mgr, param_path);
        yolo_c.load_model(mgr, model_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_c load param %s", param_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_c load model %s", model_path);
    }
    if (use_segmentation) {
        yolo_s.load_param(mgr, param_path);
        yolo_s.load_model(mgr, model_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_s load param %s", param_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_s load model %s", model_path);
    }
    if (use_detect) {
        yolo_d.load_param(mgr, param_path);
        yolo_d.load_model(mgr, model_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_d load param %s", param_path);
        __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "yolo_d load model %s", model_path);
    }

    target_size = _target_size;
    mean_values[0] = _mean_values[0];
    mean_values[1] = _mean_values[1];
    mean_values[2] = _mean_values[2];
    norm_values[0] = _norm_values[0];
    norm_values[1] = _norm_values[1];
    norm_values[2] = _norm_values[2];

    return 0;
}

void Yolo::initNativeCallback(JavaVM *vm, jlong nativeObjAddr, jobject pJobject) {
    javaVM = vm;

    /**
     * JNIEnv不支持跨线程调用
     * */
    JNIEnv *env;
    vm->AttachCurrentThread(&env, nullptr);
    j_mat_addr = nativeObjAddr;

    j_callback = env->NewGlobalRef(pJobject);
}

int Yolo::classify(const cv::Mat &rgb) {
    if (j_state == 1) {
        static const float scale_values[3] = {0.017f, 0.017f, 0.017f};

        int width = rgb.cols;
        int height = rgb.rows;

        //把opencv Mat转为 ncnn Mat
        ncnn::Mat in = ncnn::Mat::from_pixels(rgb.data, ncnn::Mat::PIXEL_RGB, width, height);

        std::vector<float> cls_scores;
        {
            in.substract_mean_normalize(mean_values, scale_values);
            ncnn::Extractor ex = yolo_c.create_extractor();
            ex.input("images", in);

            ncnn::Mat out;
            ex.extract("output", out);

            int output_size = out.w;
            float float_buffer[output_size];
            for (int j = 0; j < out.w; j++) {
                float_buffer[j] = out[j];
            }

            /**
             * 回调给Java/Kotlin层
             * */
            JNIEnv *env;
            javaVM->AttachCurrentThread(&env, nullptr);
            jclass callback_clazz = env->GetObjectClass(j_callback);
            jmethodID j_method_id = env->GetMethodID(callback_clazz, "onClassify", "([F)V");

            jfloatArray j_output_Data = env->NewFloatArray(output_size);
            env->SetFloatArrayRegion(j_output_Data, 0, output_size, float_buffer);

            env->CallVoidMethod(j_callback, j_method_id, j_output_Data);
        }
    }
    return 0;
}

int Yolo::segmentation(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                       float nms_threshold) {
    if (j_state == 2) {
        int width = rgb.cols;
        int height = rgb.rows;

        // pad to multiple of 32
        int w = width;
        int h = height;
        float scale;
        if (w > h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB, width,
                                                     height, w, h);

        // pad to target_size rectangle
        int w_pad = (w + 31) / 32 * 32 - w;
        int h_pad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(in, in_pad, h_pad / 2, h_pad - h_pad / 2, w_pad / 2,
                               w_pad - w_pad / 2,
                               ncnn::BORDER_CONSTANT, 0.f);

        in_pad.substract_mean_normalize(0, norm_values);

        JNIEnv *env;
        javaVM->AttachCurrentThread(&env, nullptr);
        jclass callback_clazz = env->GetObjectClass(j_callback);
        jmethodID j_method_id = env->GetMethodID(
                callback_clazz, "onSegmentation", "(Ljava/util/ArrayList;Ljava/util/ArrayList;)V"
        );

        //获取ArrayList类
        jclass list_clazz = env->FindClass("java/util/ArrayList");
        jmethodID arraylist_init = env->GetMethodID(list_clazz, "<init>", "()V");
        jmethodID arraylist_add = env->GetMethodID(list_clazz, "add", "(Ljava/lang/Object;)Z");
        //初始化ArrayList对象
        jobject segment_array_obj = env->NewObject(list_clazz, arraylist_init);
        jobject detect_array_obj = env->NewObject(list_clazz, arraylist_init);

        //分割
        {
            ncnn::Extractor ex = yolo_s.create_extractor();
            ex.input("images", in_pad);

            ncnn::Mat out;
            ex.extract("output", out);

            ncnn::Mat mask_proto;
            ex.extract("seg", mask_proto);

            std::vector<int> strides = {8, 16, 32};
            std::vector<GridAndStride> grid_strides;
            generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

            std::vector<Object> proposals;
            std::vector<Object> objects8;
            generate_proposals(grid_strides, out, prob_threshold, objects8, 6);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, nms_threshold);

            int count = picked.size();

            ncnn::Mat mask_feat = ncnn::Mat(32, count, sizeof(float));
            for (int i = 0; i < count; i++) {
                float *mask_feat_ptr = mask_feat.row(i);
                std::memcpy(mask_feat_ptr, proposals[picked[i]].mask_feat.data(),
                            sizeof(float) * proposals[picked[i]].mask_feat.size());
            }

            ncnn::Mat mask_pred_result;
            decode_mask(mask_feat, width, height, mask_proto, in_pad, w_pad, h_pad,
                        mask_pred_result);

            objects.resize(count);
            for (int i = 0; i < count; i++) {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].rect.x - (w_pad / 2)) / scale;
                float y0 = (objects[i].rect.y - (h_pad / 2)) / scale;
                float x1 = (objects[i].rect.x + objects[i].rect.width - (w_pad / 2)) / scale;
                float y1 = (objects[i].rect.y + objects[i].rect.height - (h_pad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;

                objects[i].mask = cv::Mat::zeros(height, width, CV_32FC1);
                cv::Mat mask = cv::Mat(height, width, CV_32FC1,
                                       (float *) mask_pred_result.channel(i));
                mask(objects[i].rect).copyTo(objects[i].mask(objects[i].rect));
            }

            for (const auto &item: objects) {
                auto rect = item.rect;

                float array[6];
                array[0] = rect.x;
                array[1] = rect.y;
                array[2] = rect.width;
                array[3] = rect.height;
                array[4] = (float) item.label;
                array[5] = item.prob * 100;

                jfloatArray result_array = env->NewFloatArray(6);
                env->SetFloatArrayRegion(result_array, 0, 6, array);

                //add
                env->CallBooleanMethod(segment_array_obj, arraylist_add, result_array);
            }
        }

        //检测
        {
            ncnn::Extractor ex = yolo_d.create_extractor();

            ex.input("images", in_pad);

            std::vector<Object> proposals;

            ncnn::Mat out;
            ex.extract("output", out);

            std::vector<int> strides = {8, 16, 32}; // might have stride=64
            std::vector<GridAndStride> grid_strides;
            generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
            generate_proposals(grid_strides, out, 0.4f, proposals, 43);

            // sort all proposals by score from highest to lowest
            qsort_descent_inplace(proposals);

            // apply nms with nms_threshold
            std::vector<int> picked;
            nms_sorted_bboxes(proposals, picked, 0.5f);

            int count = picked.size();

            objects.resize(count);
            for (int i = 0; i < count; i++) {
                objects[i] = proposals[picked[i]];

                // adjust offset to original unpadded
                float x0 = (objects[i].rect.x - (w_pad / 2)) / scale;
                float y0 = (objects[i].rect.y - (h_pad / 2)) / scale;
                float x1 = (objects[i].rect.x + objects[i].rect.width - (w_pad / 2)) / scale;
                float y1 = (objects[i].rect.y + objects[i].rect.height - (h_pad / 2)) / scale;

                // clip
                x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
                y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
                x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
                y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

                objects[i].rect.x = x0;
                objects[i].rect.y = y0;
                objects[i].rect.width = x1 - x0;
                objects[i].rect.height = y1 - y0;
            }

            // sort objects by area
            struct {
                bool operator()(const Object &a, const Object &b) const {
                    return a.rect.area() > b.rect.area();
                }
            } objects_area_greater;
            std::sort(objects.begin(), objects.end(), objects_area_greater);

            for (const auto &item: objects) {
                auto rect = item.rect;

                float array[6];
                array[0] = rect.x;
                array[1] = rect.y;
                array[2] = rect.width;
                array[3] = rect.height;
                array[4] = (float) item.label;
                array[5] = item.prob * 100;

                jfloatArray result_array = env->NewFloatArray(6);
                env->SetFloatArrayRegion(result_array, 0, 6, array);

                //add
                env->CallBooleanMethod(detect_array_obj, arraylist_add, result_array);
            }
        }

        //回调
        env->CallVoidMethod(j_callback, j_method_id, segment_array_obj, detect_array_obj);
    }
    return 0;
}

int Yolo::detect(const cv::Mat &rgb, std::vector<Object> &objects, float prob_threshold,
                 float nms_threshold) {
    if (j_state == 3) {
        int width = rgb.cols;
        int height = rgb.rows;

        // pad to multiple of 32
        int w = width;
        int h = height;
        float scale = 1.f;
        if (w > h) {
            scale = (float) target_size / w;
            w = target_size;
            h = h * scale;
        } else {
            scale = (float) target_size / h;
            h = target_size;
            w = w * scale;
        }

        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
                rgb.data, ncnn::Mat::PIXEL_RGB, width, height, w, h
        );

        // pad to target_size rectangle
        int w_pad = (w + 31) / 32 * 32 - w;
        int h_pad = (h + 31) / 32 * 32 - h;
        ncnn::Mat in_pad;
        ncnn::copy_make_border(
                in, in_pad, h_pad / 2, h_pad - h_pad / 2, w_pad / 2,
                w_pad - w_pad / 2,
                ncnn::BORDER_CONSTANT, 0.f
        );

        in_pad.substract_mean_normalize(0, norm_values);

        ncnn::Extractor ex = yolo_d.create_extractor();

        ex.input("images", in_pad);

        std::vector<Object> proposals;

        ncnn::Mat out;
        ex.extract("output", out);

        std::vector<int> strides = {8, 16, 32}; // might have stride=64
        std::vector<GridAndStride> grid_strides;
        generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
        generate_proposals(grid_strides, out, prob_threshold, proposals, 43);

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++) {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].rect.x - (w_pad / 2)) / scale;
            float y0 = (objects[i].rect.y - (h_pad / 2)) / scale;
            float x1 = (objects[i].rect.x + objects[i].rect.width - (w_pad / 2)) / scale;
            float y1 = (objects[i].rect.y + objects[i].rect.height - (h_pad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }

        // sort objects by area
        struct {
            bool operator()(const Object &a, const Object &b) const {
                return a.rect.area() > b.rect.area();
            }
        } objects_area_greater;
        std::sort(objects.begin(), objects.end(), objects_area_greater);

        /**
         * 回调给Java/Kotlin层
         * */
        JNIEnv *env;
        javaVM->AttachCurrentThread(&env, nullptr);
        jclass callback_clazz = env->GetObjectClass(j_callback);
        /**
         * I: 整数类型（int）
         * J: 长整数类型（long）
         * D: 双精度浮点数类型（double）
         * F: 单精度浮点数类型（float）
         * Z: 布尔类型（boolean）
         * C: 字符类型（char）
         * B: 字节类型（byte）
         * S: 短整数类型（short）
         * <br>-----------------------------------------------<br>
         * Ljava/lang/Object;: 表示 Object 类型的引用
         * Ljava/lang/String;: 表示 String 类型的引用
         * L包名/类名;: 表示特定包名和类名的引用
         * <br>-----------------------------------------------<br>
         * 例如：
         * int add(int a, int b): (II)I
         *
         * String concat(String str1, String str2): (Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;
         * <br>-----------------------------------------------<br>
         * [Ljava/lang/String;: 表示 String 类型的一维数组
         * */
        jmethodID j_method_id = env->GetMethodID(
                callback_clazz, "onDetect", "(Ljava/util/ArrayList;)V"
        );

        //获取ArrayList类
        jclass list_clazz = env->FindClass("java/util/ArrayList");
        jmethodID arraylist_init = env->GetMethodID(list_clazz, "<init>", "()V");
        jmethodID arraylist_add = env->GetMethodID(list_clazz, "add", "(Ljava/lang/Object;)Z");
        //初始化ArrayList对象
        jobject arraylist_obj = env->NewObject(list_clazz, arraylist_init);

        for (const auto &item: objects) {
            auto rect = item.rect;

            float array[6];
            array[0] = rect.x;
            array[1] = rect.y;
            array[2] = rect.x + rect.width;
            array[3] = rect.y + rect.height;
            array[4] = (float) item.label;
            array[5] = item.prob * 100;

            jfloatArray result_array = env->NewFloatArray(6);
            env->SetFloatArrayRegion(result_array, 0, 6, array);

            //add
            env->CallBooleanMethod(arraylist_obj, arraylist_add, result_array);
        }
        //回调
        env->CallVoidMethod(j_callback, j_method_id, arraylist_obj);

        /**
         * Mat数据。
         * <br>-----------------------------------------------<br>
         * 通过内存地址赋值。Java层传入Mat对象内存地址，再通过C++给此地址赋值，Java即可得到内存地址的Mat矩阵数据
         * */
        auto *res = (cv::Mat *) j_mat_addr;
        res->create(rgb.rows, rgb.cols, rgb.type());
        memcpy(res->data, rgb.data, rgb.rows * rgb.step);
    }
    return 0;
}

int Yolo::draw(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const char *class_names[] = {
            "tripod", "tee", "person",
            "shut-off valve", "hazard signs", "pressure tester",
            "pressure gauge", "reflective clothing", "respirator masks",
            "throat foil", "round-headed water gun", "safety signs",
            "helmet", "security identification", "safety ropes",
            "intercom", "pointed water gun", "switch",
            "alarm device", "joint", "construction street signs",
            "gas detectors", "hoses", "hose_rectangle",
            "flow-meter", "fire hydrant box", "fire extinguisher",
            "lighting equipment", "flame-out protection", "exposed wires",
            "circuit diagram", "cordon", "regulator",
            "length adjuster", "stickers", "across wires",
            "road cones", "hose", "filter",
            "distribution box", "long-shank valves", "valve", "ducts"
    };

    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };

    int color_index = 0;

    for (const auto &obj: objects) {
        const unsigned char *color = colors[color_index % 19];

        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1,
                                              &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::Size size = cv::Size(label_size.width, label_size.height + baseLine);
        cv::Rect rc = cv::Rect(cv::Point(x, y), size);
        cv::rectangle(rgb, rc, cc, -1);

        cv::Scalar text_scalar = (color[0] + color[1] + color[2] >= 381)
                                 ? cv::Scalar(0, 0, 0)
                                 : cv::Scalar(255, 255, 255);


        cv::putText(rgb, text,
                    cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.5,
                    text_scalar, 1
        );
    }
    return 0;
}

int Yolo::draw_mask(cv::Mat &rgb, const std::vector<Object> &objects) {
    static const unsigned char colors[19][3] = {
            {54,  67,  244},
            {99,  30,  233},
            {176, 39,  156},
            {183, 58,  103},
            {181, 81,  63},
            {243, 150, 33},
            {244, 169, 3},
            {212, 188, 0},
            {136, 150, 0},
            {80,  175, 76},
            {74,  195, 139},
            {57,  220, 205},
            {59,  235, 255},
            {7,   193, 255},
            {0,   152, 255},
            {34,  87,  255},
            {72,  85,  121},
            {158, 158, 158},
            {139, 125, 96}
    };

    int color_index = 0;

    for (const auto &obj: objects) {
        const unsigned char *color = colors[color_index % 19];

        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);
        for (int y = 0; y < rgb.rows; y++) {
            uchar *image_ptr = rgb.ptr(y);
            const auto *mask_ptr = obj.mask.ptr<float>(y);
            for (int x = 0; x < rgb.cols; x++) {
                if (mask_ptr[x] >= 0.5) {
                    image_ptr[0] = cv::saturate_cast<uchar>(image_ptr[0] * 0.5 + color[2] * 0.5);
                    image_ptr[1] = cv::saturate_cast<uchar>(image_ptr[1] * 0.5 + color[1] * 0.5);
                    image_ptr[2] = cv::saturate_cast<uchar>(image_ptr[2] * 0.5 + color[0] * 0.5);
                }
                image_ptr += 3;
            }
        }
        cv::rectangle(rgb, obj.rect, cc, 2);
    }
    return 0;
}