package com.pengxh.ncnn.yolov8

import android.content.res.AssetManager
import android.view.Surface

class Yolov8ncnn {
    companion object {
        init {
            System.loadLibrary("yolov8ncnn")
        }
    }

    /**
     * @param mgr 手机内存资源管理器
     * @param modelId 模型ID
     * @param useGpu 是否使用GPU
     * @param useClassify 是否使用分类模型
     * @param useSegmentation 是否使用分割模型
     * @param useDetect 是否使用检测模型
     * */
    external fun loadModel(
        mgr: AssetManager, modelId: Int, useGpu: Boolean, useClassify: Boolean,
        useSegmentation: Boolean, useDetect: Boolean
    ): Boolean

    /**
     * @param mgr 手机内存资源管理器
     * @param ids 多模型ID数组
     * @param useGpu 是否使用GPU
     * */
    external fun loadMultiModel(mgr: AssetManager, ids: IntArray, useGpu: Boolean): Boolean

    /**
     * @param facing 相机 0-前置镜头，1-后置镜头
     * */
    external fun openCamera(facing: Int): Boolean

    external fun closeCamera(): Boolean

    external fun setOutputWindow(
        surface: Surface, nativeObjAddr: Long, callBack: INativeCallback
    ): Boolean

    external fun onPause(): Boolean

    external fun onRestart(): Boolean
}