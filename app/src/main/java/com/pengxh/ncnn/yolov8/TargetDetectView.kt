package com.pengxh.ncnn.yolov8

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.text.TextPaint
import android.util.AttributeSet
import android.view.View
import com.pengxh.kt.lite.extensions.dp2px
import com.pengxh.kt.lite.extensions.sp2px

class TargetDetectView constructor(context: Context, attrs: AttributeSet) : View(context, attrs) {

    private val kTag = "DetectView"
    private val textPaint by lazy { TextPaint() }
    private val backgroundPaint by lazy { Paint() }
    private val borderPaint by lazy { Paint() }
    private val textRect by lazy { Rect() }
    private val rect by lazy { Rect() }
    private var detectResults: MutableList<YoloResult> = ArrayList()
    private var segmentationResults: MutableList<YoloResult> = ArrayList()
    private var textHeight = 0

    init {
        textPaint.color = Color.WHITE
        textPaint.isAntiAlias = true
        textPaint.textAlign = Paint.Align.CENTER
        textPaint.textSize = 16f.sp2px(context)
        val fontMetrics = textPaint.fontMetrics
        textHeight = (fontMetrics.bottom - fontMetrics.top).toInt()

        backgroundPaint.style = Paint.Style.FILL
        backgroundPaint.isAntiAlias = true
        backgroundPaint.color = Color.BLUE

        borderPaint.style = Paint.Style.STROKE
        borderPaint.strokeWidth = 2f.dp2px(context) //设置线宽
        borderPaint.isAntiAlias = true
        borderPaint.color = Color.BLUE
    }

    fun updateTargetPosition(
        segmentationResults: MutableList<YoloResult>, detectResults: MutableList<YoloResult>
    ) {
        this.segmentationResults = segmentationResults
        this.detectResults = detectResults
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        detectResults.forEach {
            drawTarget(canvas, it, LocaleConstant.TARGET_NAMES_ARRAY[it.type])
        }

        segmentationResults.forEach {
            drawTarget(canvas, it, LocaleConstant.SEGMENTATION_ARRAY[it.type])
        }
    }

    private fun drawTarget(canvas: Canvas, it: YoloResult, label: String) {
        val textLength = textPaint.measureText(label)
        val textRectWidth = textLength * 1.1 //文字背景宽度
        val textRectHeight = textHeight * 1.1 //文字背景高度

        textRect.set(
            it.position[0].toInt(),
            it.position[1].toInt(),
            (it.position[0] + textRectWidth).toInt(),
            (it.position[1] - textRectHeight).toInt()
        )
        canvas.drawRect(textRect, backgroundPaint)

        //画文字
        canvas.drawText(
            label,
            (it.position[0] + textRectWidth / 2).toFloat(),
            (it.position[1] - textRectHeight / 4).toFloat(),
            textPaint
        )

        //画框
        rect.set(
            it.position[0].toInt(),
            it.position[1].toInt(),
            it.position[2].toInt(),
            it.position[3].toInt()
        )
        canvas.drawRect(rect, borderPaint)
    }
}