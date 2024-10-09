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

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

#include <platform.h>
#include <benchmark.h>

#include "yolo.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON

static int draw_unsupported(cv::Mat &rgb) {
    const char text[] = "unsupported";

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1.0, 1, &baseLine);

    int y = (rgb.rows - label_size.height) / 2;
    int x = (rgb.cols - label_size.width) / 2;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y),
                                cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 0));

    return 0;
}

static int draw_fps(cv::Mat &rgb) {
    // resolve moving average
    double avg_fps = 0.f;
    {
        static double t0 = 0.f;
        static double fps_history[10] = {0.f};

        double t1 = ncnn::get_current_time();
        if (t0 == 0.f) {
            t0 = t1;
            return 0;
        }

        double fps = 1000.f / (t1 - t0);
        t0 = t1;

        for (int i = 9; i >= 1; i--) {
            fps_history[i] = fps_history[i - 1];
        }
        fps_history[0] = fps;

        if (fps_history[9] == 0.f) {
            return 0;
        }

        for (double i: fps_history) {
            avg_fps += i;
        }
        avg_fps /= 10.f;
    }

    char text[32];
    sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb,
                  cv::Rect(cv::Point(x, y),
                           cv::Size(label_size.width, label_size.height + baseLine)),
                  cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}

static Yolo *yolo_ptr = nullptr;
static ncnn::Mutex lock;
static JavaVM *jvm_ptr = nullptr;

class MyNdkCamera : public NdkCameraWindow {
public:
    void on_image_render(cv::Mat &rgb) const override;
};

void MyNdkCamera::on_image_render(cv::Mat &rgb) const {
    // nanodet
    {
        ncnn::MutexLockGuard g(lock);

        if (yolo_ptr) {
            //分类
            yolo_ptr->classify(rgb);

            std::vector<Object> objects;

            //分割
            yolo_ptr->segmentation(rgb, objects);

            //检测
            yolo_ptr->detect(rgb, objects);

            //绘制
            yolo_ptr->draw(rgb, objects);
        } else {
            draw_unsupported(rgb);
        }
    }

    //回执画面FPS值
    draw_fps(rgb);
}

static MyNdkCamera *camera_ptr = nullptr;

//分割、分类、检测
const char *model_types[] = {"best-sim-opt-fp16", "model.ncnn", "yolov8s-detect-sim-opt-fp16"};
const int target_sizes[] = {320, 320, 320};
const float mean_values[][3] = {
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f}
};
const float norm_values[][3] = {
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
        {1 / 255.f, 1 / 255.f, 1 / 255.f}
};

extern "C" {
JNIEXPORT jint JNI_OnLoad(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    jvm_ptr = vm;

    camera_ptr = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM *vm, void *reserved) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        ncnn::MutexLockGuard g(lock);

        delete yolo_ptr;
        yolo_ptr = nullptr;
    }

    delete camera_ptr;
    camera_ptr = nullptr;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_loadModel(JNIEnv *env, jobject thiz,
                                                 jobject assetManager, jint model_id,
                                                 jboolean use_gpu, jboolean use_classify,
                                                 jboolean use_segmentation, jboolean use_detect) {
    if (model_id < 0 || model_id > 2) {
        return JNI_FALSE;
    }

    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    const char *model_type = model_types[(int) model_id];
    int target_size = target_sizes[(int) model_id];

    // reload
    {
        ncnn::MutexLockGuard g(lock);

        if (use_gpu && ncnn::get_gpu_count() == 0) {
            // no gpu
            delete yolo_ptr;
            yolo_ptr = nullptr;
        } else {
            if (!yolo_ptr)
                yolo_ptr = new Yolo;
            int state;
            if (use_classify) {
                state = 1;
            }
            if (use_segmentation) {
                state = 2;
            }
            if (use_detect) {
                state = 3;
            }
            yolo_ptr->j_state = state;
            yolo_ptr->load(
                    mgr,
                    model_type,
                    target_size,
                    mean_values[(int) model_id],
                    norm_values[(int) model_id],
                    use_classify,
                    use_segmentation,
                    use_detect,
                    use_gpu
            );
        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_loadMultiModel(JNIEnv *env, jobject thiz,
                                                      jobject assetManager,
                                                      jintArray ids, jboolean use_gpu) {
    AAssetManager *mgr = AAssetManager_fromJava(env, assetManager);

    jint *intArray = env->GetIntArrayElements(ids, nullptr);
    jsize len = env->GetArrayLength(ids);
    for (int i = 0; i < len; i++) {
        int *id = intArray + i;
        const char *model_type = model_types[*id];
        int target_size = target_sizes[*id];

        {
            ncnn::MutexLockGuard g(lock);

            if (use_gpu && ncnn::get_gpu_count() == 0) {
                // no gpu
                delete yolo_ptr;
                yolo_ptr = nullptr;
            } else {
                if (!yolo_ptr)
                    yolo_ptr = new Yolo;
                yolo_ptr->j_state = 2;
                if (*id == 0) {
                    yolo_ptr->load(
                            mgr,
                            model_type,
                            target_size,
                            mean_values[*id],
                            norm_values[*id],
                            false,
                            true,
                            false,
                            use_gpu
                    );
                } else {
                    yolo_ptr->load(
                            mgr,
                            model_type,
                            target_size,
                            mean_values[*id],
                            norm_values[*id],
                            false,
                            false,
                            true,
                            use_gpu
                    );
                }
            }
        }
    }

    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_openCamera(JNIEnv *env, jobject thiz, jint facing) {
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    if (camera_ptr == nullptr) {
        return JNI_FALSE;
    }
    camera_ptr->open((int) facing);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_closeCamera(JNIEnv *env, jobject thiz) {
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    if (camera_ptr == nullptr) {
        return JNI_FALSE;
    }
    camera_ptr->close();
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_setOutputWindow(JNIEnv *env, jobject thiz,
                                                       jobject surface, jlong nativeObjAddr,
                                                       jobject native_callback) {
    ANativeWindow *win = ANativeWindow_fromSurface(env, surface);

    if (camera_ptr == nullptr) {
        return JNI_FALSE;
    }
    camera_ptr->set_window(win);

    if (yolo_ptr == nullptr) {
        return JNI_FALSE;
    }
    yolo_ptr->initNativeCallback(jvm_ptr, nativeObjAddr, native_callback);
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_onPause(JNIEnv *env, jobject thiz) {
    if (yolo_ptr == nullptr) {
        return JNI_FALSE;
    }
    yolo_ptr->j_state = 0;
    return JNI_TRUE;
}

JNIEXPORT jboolean JNICALL
Java_com_pengxh_ncnn_yolov8_Yolov8ncnn_onRestart(JNIEnv *env, jobject thiz) {
    if (yolo_ptr == nullptr) {
        return JNI_FALSE;
    }
    yolo_ptr->j_state = 3;
    return JNI_TRUE;
}
}