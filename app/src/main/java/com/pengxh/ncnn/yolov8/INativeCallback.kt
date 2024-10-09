package com.pengxh.ncnn.yolov8;

import java.util.ArrayList;

public interface INativeCallback {
    /**
     * 分类
     */
    void onClassify(float[] possibles);

    /**
     * 分割
     */
    void onSegmentation(ArrayList<float[]> segmentationOutput, ArrayList<float[]> detectOutput);

    /**
     * 检测
     */
    void onDetect(ArrayList<float[]> output);
}
