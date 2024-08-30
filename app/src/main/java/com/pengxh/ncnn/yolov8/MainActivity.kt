package com.pengxh.ncnn.yolov8

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.WindowManager
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.pengxh.kt.lite.base.KotlinBaseActivity
import com.pengxh.kt.lite.extensions.dp2px
import com.pengxh.kt.lite.extensions.toJson
import com.pengxh.ncnn.yolov8.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.core.Mat


class MainActivity : KotlinBaseActivity<ActivityMainBinding>(), SurfaceHolder.Callback,
    INativeCallback {

    private val kTag = "MainActivity"
    private val yolov8ncnn by lazy { Yolov8ncnn() }
    private val mat by lazy { Mat() }
    private var facing = 1

    override fun initEvent() {

    }

    override fun initOnCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        OpenCVLoader.initLocal()



        binding.surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        binding.surfaceView.holder.addCallback(this)

        yolov8ncnn.loadModel(assets, 2, false, false, false, true)
//        yolov8ncnn.loadMultiModel(assets, intArrayOf(0, 2), false)
    }

    override fun initViewBinding(): ActivityMainBinding {
        return ActivityMainBinding.inflate(layoutInflater)
    }

    override fun observeRequestState() {

    }

    override fun setupTopBarLayout() {

    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        yolov8ncnn.setOutputWindow(holder.surface, mat.nativeObjAddr, this)
    }

    override fun onClassify(possibles: FloatArray) {
        Log.d(kTag, possibles.contentToString())
    }

    override fun onSegmentation(
        segmentationOutput: ArrayList<FloatArray>, detectOutput: ArrayList<FloatArray>
    ) {
        //转成泛型集合
        val segmentationResults = ArrayList<YoloResult>()
        segmentationOutput.forEach {
            val yolo = YoloResult()

            val array = FloatArray(4)
            array[0] = it[0].dp2px(this)
            array[1] = it[1].dp2px(this)
            array[2] = it[2].dp2px(this)
            array[3] = it[3].dp2px(this)
            yolo.position = array

            yolo.type = it[4].toInt()

            yolo.prob = "${"%.2f".format(it[5])}%"
            segmentationResults.add(yolo)
        }

        val detectResults = ArrayList<YoloResult>()
        detectOutput.forEach {
            val yolo = YoloResult()

            val array = FloatArray(4)
            array[0] = it[0].dp2px(this)
            array[1] = it[1].dp2px(this)
            array[2] = it[2].dp2px(this)
            array[3] = it[3].dp2px(this)
            yolo.position = array

            yolo.type = it[4].toInt()

            yolo.prob = "${"%.2f".format(it[5])}%"
            detectResults.add(yolo)
        }
        binding.detectView.updateTargetPosition(segmentationResults, detectResults)
    }

    override fun onDetect(output: ArrayList<FloatArray>) {
        //转成泛型集合
        val results = ArrayList<YoloResult>()
        output.forEach {
            /**
             * 前四位是目标Rect，第五位是目标名对应的角标，第六位是可信度
             *
             * [135.88397,120.17752,68.061325,204.02115,28.0,43.642334]
             * */
            val yolo = YoloResult()

            val array = FloatArray(4)
            array[0] = it[0].dp2px(this)
            array[1] = it[1].dp2px(this)
            array[2] = it[2].dp2px(this)
            array[3] = it[3].dp2px(this)
            yolo.position = array

            yolo.type = it[4].toInt()

            //保留两位有效小数
            yolo.prob = "${"%.2f".format(it[5])}%"
            results.add(yolo)
        }
        Log.d(kTag, results.toJson())
        binding.detectView.updateTargetPosition(results)

//        if (mat.width() > 0 || mat.height() > 0) {
//            val bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(mat, bitmap, true)
//            bitmap.saveImage("${createImageFileDir()}/${System.currentTimeMillis()}.png")
//        } else {
//            Log.d(kTag, "width: ${mat.width()}, height: ${mat.height()}")
//        }
    }

    override fun surfaceCreated(holder: SurfaceHolder) {}

    override fun surfaceDestroyed(holder: SurfaceHolder) {}

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_DENIED
        ) {
            ActivityCompat.requestPermissions(
                this, arrayOf(Manifest.permission.CAMERA), 100
            )
        }
        yolov8ncnn.openCamera(facing)
    }

    override fun onPause() {
        super.onPause()
        yolov8ncnn.closeCamera()
    }
}