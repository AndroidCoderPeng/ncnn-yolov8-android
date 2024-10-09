package com.pengxh.ncnn.yolov8

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.PixelFormat
import android.os.Bundle
import android.os.Handler
import android.os.Message
import android.view.SurfaceHolder
import android.view.WindowManager
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.pengxh.kt.lite.base.KotlinBaseActivity
import com.pengxh.kt.lite.extensions.dp2px
import com.pengxh.kt.lite.utils.WeakReferenceHandler
import com.pengxh.kt.lite.widget.dialog.AlertControlDialog
import com.pengxh.ncnn.yolov8.databinding.ActivityMainBinding
import org.opencv.core.Mat
import org.opencv.osgi.OpenCVNativeLoader


class MainActivity : KotlinBaseActivity<ActivityMainBinding>(), SurfaceHolder.Callback,
    INativeCallback, Handler.Callback {

    private val kTag = "MainActivity"
    private val yolov8ncnn by lazy { Yolov8ncnn() }
    private val mat by lazy { Mat() }
    private val weakReferenceHandler by lazy { WeakReferenceHandler(this) }

    override fun initEvent() {

    }

    override fun initOnCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        val openCVNativeLoader = OpenCVNativeLoader()
        openCVNativeLoader.init()

        yolov8ncnn.loadModel(
            assets, 2, useGpu = false,
            useClassify = false, useSegmentation = false, useDetect = true
        )

        binding.surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        binding.surfaceView.holder.addCallback(this)
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

    }

    override fun onSegmentation(
        segmentationOutput: ArrayList<FloatArray>, detectOutput: ArrayList<FloatArray>
    ) {
        //转成泛型集合
        val segmentationResults = ArrayList<YoloResult>()
        segmentationOutput.forEach {
            segmentationResults.add(it.convert2YoloResult(this))
        }

        val detectResults = ArrayList<YoloResult>()
        detectOutput.forEach {
            detectResults.add(it.convert2YoloResult(this))
        }

        binding.detectView.updateTargetPosition(segmentationResults, detectResults)

//        if (mat.width() > 0 || mat.height() > 0) {
//            val bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
//            Utils.matToBitmap(mat, bitmap, true)
//            bitmap.saveImage("${createImageFileDir()}/${System.currentTimeMillis()}.png")
//        } else {
//            Log.d(kTag, "width: ${mat.width()}, height: ${mat.height()}")
//        }
    }

    override fun onDetect(output: ArrayList<String>) {
        if (output.isEmpty()) {
            return
        }

        //暂停算法
        yolov8ncnn.onPause()
        weakReferenceHandler.sendEmptyMessageDelayed(2024082901, 1000)
    }

    override fun handleMessage(msg: Message): Boolean {
        when (msg.what) {
            2024082901 -> {
                AlertControlDialog.Builder()
                    .setContext(this)
                    .setTitle("温馨提示")
                    .setMessage("识别到目标场景，是否开始识别目标？")
                    .setNegativeButton("重新识别")
                    .setPositiveButton("开始检查")
                    .setOnDialogButtonClickListener(object :
                        AlertControlDialog.OnDialogButtonClickListener {
                        override fun onConfirmClick() {
                            //调用多模型
                            yolov8ncnn.loadMultiModel(assets, intArrayOf(0, 2), false)
                        }

                        override fun onCancelClick() {
                            yolov8ncnn.onRestart()
                        }
                    }).build().show()
            }
        }
        return true
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
        yolov8ncnn.openCamera(1)
    }

    override fun onPause() {
        super.onPause()
        yolov8ncnn.closeCamera()
    }

    private fun FloatArray.convert2YoloResult(activity: AppCompatActivity): YoloResult {
        /**
         * 前四位是目标左上角和右下角坐标，第五位是目标名对应的角标
         *
         * [135.88397,120.17752,68.061325,204.02115,28.0,43.642334]
         * */
        val yolo = YoloResult()

        val array = FloatArray(4)
        array[0] = this[0].dp2px(activity)
        array[1] = this[1].dp2px(activity)
        array[2] = this[2].dp2px(activity)
        array[3] = this[3].dp2px(activity)
        yolo.position = array

        yolo.type = this[4].toInt()
        return yolo
    }
}