package com.pengxh.ncnn.yolov8

interface INativeCallback {
    /**
     * 分类
     */
    fun onClassify(possibles: FloatArray)

    /**
     * 分割
     */
    fun onSegmentation(
        segmentationOutput: ArrayList<FloatArray>, detectOutput: ArrayList<FloatArray>
    )

    /**
     * 检测。只需要返回类别
     */
    fun onDetect(output: ArrayList<String>)
}