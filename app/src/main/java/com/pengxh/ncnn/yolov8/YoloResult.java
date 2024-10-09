package com.pengxh.ncnn.yolov8;

public class YoloResult {
    private float[] position;
    private int type;
    public float[] getPosition() {
        return position;
    }

    public void setPosition(float[] position) {
        this.position = position;
    }

    public int getType() {
        return type;
    }

    public void setType(int type) {
        this.type = type;
    }
}
