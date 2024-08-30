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
    private val classNames = arrayListOf(
        "三脚架", "三通", "人", "切断阀", "危险告知牌",
        "压力测试仪", "压力表", "反光衣", "呼吸面罩", "喉箍",
        "圆头水枪", "安全告知牌", "安全帽", "安全标识", "安全绳",
        "对讲机", "尖头水枪", "开关", "报警装置", "接头",
        "施工路牌", "气体检测仪", "水带", "水带_矩形", "流量计",
        "消火栓箱", "灭火器", "照明设备", "熄火保护", "电线暴露",
        "电路图", "警戒线", "调压器", "调长器", "贴纸",
        "跨电线", "路锥", "软管", "过滤器", "配电箱",
        "长柄阀门", "阀门", "风管"
    )
    private val segmentationArray = arrayOf("弯折", "断裂", "烧焦", "磨损", "铁锈", "龟裂")
    private var results: MutableList<YoloResult> = ArrayList()
    private var segmentationResults: MutableList<YoloResult> = ArrayList()
    private var textHeight = 0

    init {
        textPaint.color = Color.WHITE
        textPaint.isAntiAlias = true
        textPaint.textAlign = Paint.Align.CENTER
        textPaint.textSize = 16f.sp2px(context)
        val fontMetrics = textPaint.fontMetrics
        textHeight = (fontMetrics.bottom - fontMetrics.top).toInt()

        backgroundPaint.color = Color.BLUE
        backgroundPaint.style = Paint.Style.FILL
        backgroundPaint.isAntiAlias = true

        borderPaint.color = Color.BLUE
        borderPaint.style = Paint.Style.STROKE
        borderPaint.strokeJoin = Paint.Join.ROUND
        borderPaint.strokeWidth = 3f.dp2px(context) //设置线宽
        borderPaint.isAntiAlias = true
    }

    fun updateTargetPosition(results: MutableList<YoloResult>) {
        this.results = results
        invalidate()
    }

    fun updateTargetPosition(
        segmentationResults: MutableList<YoloResult>, detectResults: MutableList<YoloResult>
    ) {
        this.segmentationResults = segmentationResults
        this.results = detectResults
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        results.forEach {
            drawTarget(canvas, it, classNames[it.type])
        }

        segmentationResults.forEach {
            drawTarget(canvas, it, segmentationArray[it.type])
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

        //画文字。数值是文字左右边距，可酌情调整
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