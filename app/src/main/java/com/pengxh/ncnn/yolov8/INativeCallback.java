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
    void onPartition(ArrayList<String> output);

    /**
     * 检测
     */
    void onDetect(ArrayList<String> output);
}
