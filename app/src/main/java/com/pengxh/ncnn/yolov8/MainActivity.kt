package com.pengxh.ncnn.yolov8

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.os.Bundle
import android.util.Log
import android.view.SurfaceHolder
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.pengxh.kt.lite.base.KotlinBaseActivity
import com.pengxh.kt.lite.extensions.createImageFileDir
import com.pengxh.kt.lite.extensions.saveImage
import com.pengxh.ncnn.yolov8.databinding.ActivityMainBinding
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat


class MainActivity : KotlinBaseActivity<ActivityMainBinding>(), SurfaceHolder.Callback,
    INativeCallback {

    private val kTag = "MainActivity"
    private val yolov8ncnn by lazy { Yolov8ncnn() }
    private val mat by lazy { Mat() }
    private var facing = 1
    private var currentModel = 0
    private var currentProcessor = 0

    override fun initEvent() {
        binding.processorSpinner.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    arg0: AdapterView<*>?, arg1: View, position: Int, id: Long
                ) {
                    if (position != currentProcessor) {
                        currentProcessor = position
                        reloadModel()
                    }
                }

                override fun onNothingSelected(arg0: AdapterView<*>?) {}
            }
    }

    private fun reloadModel() {
        val result = yolov8ncnn.loadModel(assets, currentModel, currentProcessor)
        if (!result) {
            Log.d(kTag, "reload: yolov8ncnn loadModel failed")
        }
    }

    override fun initOnCreate(savedInstanceState: Bundle?) {
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        OpenCVLoader.initLocal()

        binding.surfaceView.holder.setFormat(PixelFormat.RGBA_8888)
        binding.surfaceView.holder.addCallback(this)

        reloadModel()
    }

    override fun initViewBinding(): ActivityMainBinding {
        return ActivityMainBinding.inflate(layoutInflater)
    }

    override fun observeRequestState() {

    }

    override fun setupTopBarLayout() {

    }

    override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
        yolov8ncnn.setOutputWindow(holder.surface, DetectResult(), mat.nativeObjAddr, this)
    }

    override fun onDetect(output: ArrayList<DetectResult>) {
        binding.detectView.updateTargetPosition(output)

        if (mat.width() > 0 || mat.height() > 0) {
            val bitmap = Bitmap.createBitmap(mat.width(), mat.height(), Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(mat, bitmap, true)
            bitmap.saveImage("${createImageFileDir()}/${System.currentTimeMillis()}.png")
        } else {
            Log.d(kTag, "width: ${mat.width()}, height: ${mat.height()}")
        }
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