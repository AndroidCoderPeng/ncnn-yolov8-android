package com.pengxh.ncnn.yolov8;

import android.content.res.AssetManager;
import android.view.Surface;

public class Yolov8ncnn {
    public native boolean loadModel(AssetManager mgr, int model_id, boolean use_gpu, boolean use_classify, boolean use_segmentation, boolean use_detect);

    public native boolean loadMultiModel(AssetManager mgr, int[] ids, boolean use_gpu);

    public native boolean openCamera(int facing);

    public native boolean closeCamera();

    public native boolean setOutputWindow(Surface surface, long nativeObjAddr, INativeCallback nativeCallback);

    static {
        System.loadLibrary("yolov8ncnn");
    }
}
