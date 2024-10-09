# ncnn-yolov8-android

腾讯高性能神经网络前向计算框架——ncnn联合yolov8模型、OpenCV框架交叉编译移植到Android平台。

### 1、课题背景

本课题原本采用Android端采集实时画面帧，然后通过网络将画面帧传递到媒体服务器，服务器再用Python+Yolov8对画面帧做检测和识别，最后将结果返回给Android端去绘制目标检测结果。这样做最大的问题就是延时，经过局域网、4/5G/WiFi网络测试，延时大概1-2s，此方案并不是最优解。为了优化（解决）此痛点，就必须将目标检测和识别移植到Android端，否则这个延时不可能会降下来。

### 2、解决方案

如题，采用 ncnn + yolov8 + opencv 三个框架实现这一目标

### 3、集成三个框架移植到Android端

#### 3.1、Android Studio 新建C++项目

注意，必须是C++项目，否则交叉编译环境会把人搞疯，不要走弯路了，步骤如下：
![微信截图_20240603170904.png](imags/微信截图_20240603170904.png)

![微信截图_20240603170934.png](imags/微信截图_20240603170934.png)

![微信截图_20240603170944.png](imags/微信截图_20240603170944.png)

图二根据自己熟悉的语言选择即可，不一定非得Kotlin(Java也是可以的)。至于其他的选项，建议按我的来，不然出问题了，会把人搞疯。

项目建完之后，按照Android Studio Giraffe版本的Android Studio会自动给你下载8.x版本的Gradle，高版本的Gradle的在Android
Studio
Giraffe上面不咋好使，咱改成本地的Gradle仓库路径（如果本地没有Gradle的，[点击下载](https://services.gradle.org/distributions/gradle-6.7.1-all.zip)
），如下图：

![微信截图_20240603171618.png](imags/微信截图_20240603171618.png)

然后修改项目根目录下的 [settings.gradle](settings.gradle) 如下：

```gradle
rootProject.name = "Test"
include ':app'
```

注意，rootProject.name修改为自己的项目名。其余代码全部删除，用不到。

再修改项目根目录下的 [build.gradle](build.gradle) 如下：

```gradle
// Top-level build file where you can add configuration options common to all sub-projects/modules.
buildscript {
    ext.kotlin_version = '1.6.10'
    repositories {
        maven { url 'http://maven.aliyun.com/nexus/content/groups/public/' }
        mavenCentral()
        google()
    }
    dependencies {
        classpath 'com.android.tools.build:gradle:4.2.2'
        classpath "org.jetbrains.kotlin:kotlin-gradle-plugin:$kotlin_version"
    }
}

allprojects {
    repositories {
        //阿里云镜像
        maven { url 'http://maven.aliyun.com/nexus/content/groups/public/' }
        //依赖库
        maven { url 'https://jitpack.io' }
        mavenCentral()
        google()
    }
}

tasks.register('clean', Delete) {
    delete rootProject.buildDir
}
```

这个不用改，直接复制进去

再然后修改 app 目录下面的 [build.gradle](app/build.gradle) 如下：

```gradle
apply plugin: 'com.android.application'
apply plugin: 'kotlin-android'

android {
    compileSdkVersion 33

    defaultConfig {
        applicationId "com.casic.test"
        minSdkVersion 24
        targetSdkVersion 33
        versionCode 1000
        versionName "1.0.0.0"
        ndkVersion "26.1.10909125"
        ndk {
            abiFilters "arm64-v8a", "armeabi-v7a", "x86_64", "x86"
        }
        testInstrumentationRunner "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            minifyEnabled false
            proguardFiles getDefaultProguardFile('proguard-android-optimize.txt'), 'proguard-rules.pro'
        }
    }

    compileOptions {
        sourceCompatibility JavaVersion.VERSION_1_8
        targetCompatibility JavaVersion.VERSION_1_8
    }

    kotlinOptions {
        jvmTarget = '1.8'
    }

    externalNativeBuild {
        cmake {
            path file('src/main/cpp/CMakeLists.txt')
            version '3.22.1'
        }
    }

    buildFeatures {
        viewBinding true
    }
}

dependencies {
    implementation 'androidx.core:core-ktx:1.9.0'
    implementation 'androidx.appcompat:appcompat:1.6.1'
    implementation 'com.google.android.material:material:1.6.0'
    testImplementation 'junit:junit:4.13.2'
    androidTestImplementation 'androidx.test.ext:junit:1.1.5'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.5.1'
    //基础依赖库
    implementation 'com.github.AndroidCoderPeng:Kotlin-lite-lib:1.0.10'
}
```

对着自己项目的修改此文件，dependencies里面的内容可以复制过去

然后点击OK，最后在主界面点击”Try Again“即可。到此，C++项目的的环境已经配置完毕，接下来修改Android的几个基本配置。

首先修改 [AndroidManifest.xml](app/src/main/AndroidManifest.xml) 如下：

```xml
<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android" package="com.casic.test">

    <uses-permission android:name="android.permission.CAMERA" />
    <uses-feature 
        android:name="android.hardware.camera" 
        android:required="false" />

    <application 
        android:allowBackup="true" 
        android:icon="@mipmap/ic_launcher"
        android:label="@string/app_name" 
        android:roundIcon="@mipmap/ic_launcher_round"
        android:supportsRtl="true" 
        android:theme="@style/Theme.Test">
        <activity android:name=".MainActivity" android:exported="true">
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />

                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>
    </application>
</manifest>
```

对着自己项目的修改此文件。

然后修改 [MainActivity.kt](app/src/main/java/com/pengxh/ncnn/yolov8/MainActivity.kt) 如下：

```kotlin
class MainActivity : KotlinBaseActivity<ActivityMainBinding>() {
    override fun initEvent() {

    }

    override fun initOnCreate(savedInstanceState: Bundle?) {

    }

    override fun initViewBinding(): ActivityMainBinding {
        return ActivityMainBinding.inflate(layoutInflater)
    }

    override fun observeRequestState() {

    }

    override fun setupTopBarLayout() {

    }
}
```

把原来自带的那部分代码全删了，用不到。

最后修改 [gradle.properties](gradle.properties) 如下：

```properties
org.gradle.jvmargs=-Xmx2048m -Dfile.encoding=UTF-8
android.useAndroidX=true
kotlin.code.style=official
android.nonTransitiveRClass=true
android.enableJetifier=true
```

直接复制即可。然后点击”Sync Now“

最后点击顶部三角形，如果能运行，就表示已经完全完成C++项目配置。

#### 3.2、集成腾讯神经网络框架-ncnn

先下载[腾讯ncnn开源库](https://github.com/Tencent/ncnn)最新的框架，如图：

![微信截图_20240603180404.png](imags/微信截图_20240603180404.png)

然后解压，复制到项目的cpp目录下，**不要改任何文件以及代码**，如下图：
![微信截图_20240603180840.png](imags/微信截图_20240603180840.png)

修改新建项目时侯生成的 [CMakeLists.txt](app/src/main/cpp/CMakeLists.txt)，如下：

```cmake
project("test")

cmake_minimum_required(VERSION 3.22.1)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240410-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)
```

project根据实际情况修改即可，然后点击”Sync Now“

最后点击顶部三角形，如果能运行，就表示已经完成ncnn框架配置。

#### 3.3、集成opencv-mobile框架

同理，先去[opencv-mobile](https://github.com/nihui/opencv-mobile/releases/tag/v27)
下载最新框架，如图（最新版本是：4.10.0）：

![微信截图_20240603181800.png](imags/微信截图_20240603181800.png)

然后解压，复制到项目的cpp目录下，**不要改任何文件以及代码**，如下图（**最新版本是：4.10.0**）：

![微信截图_20240603182024.png](imags/微信截图_20240603182024.png)

修改上一步的 [CMakeLists.txt](app/src/main/cpp/CMakeLists.txt)，添加如下代码：

```cmake
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.10.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)
```

在之前修改过的cmake文件里面加如上两行，然后点击”Sync Now“。最后点击顶部三角形，如果能运行，就表示已经完成opencv-mobile框架配置。

#### 3.4、集成OpenCV框架

这里的OpenCV和3.3里面的opencv-mobile是有区别的，opencv-mobile是专门针对移动端做了优化。此处引入OpenCV的目的是为了后面的画面预览的数据矩阵。同样去[OpenCV](https://opencv.org/releases)
官网下载最新的Android端SDK，如下图：

![微信截图_20240604083058.png](imags/微信截图_20240604083058.png)

然后解压在桌面备用，按如下步骤操作：

![QQ截图20240604083401.png](imags/QQ截图20240604083401.png)

![微信截图_20240604083549.png](imags/微信截图_20240604083549.png)

导入进去之后会报错，别慌，按下面步骤修改即可：

1. 首先修改app目录下面的[build.gradle](app/build.gradle)，在dependencies括号里面添加一行代码，如下图：

```gradle
implementation project(':sdk')
```

这里的”sdk“就是刚刚导入进来的OpenCV依赖库的名字，如果按照我的步骤来没有改过名字的应该就是这个，如果自己改过名字的，这里填写你改过的依赖库名字。

2. 然后再修改sdk里面的[build.gradle](sdk/build.gradle)
   ，将里面的compileSdkVersion、minSdkVersion和targetSdkVersion三个字段改为了主项目一致，并删掉其中的如下代码：

```gradle
    publishing {
        singleVariant('release') {
            withSourcesJar()
            withJavadocJar()
        }
    }
```

3. 最后在主界面点击”Try Again“即可完成OpenCV的集成，最后效果如下（**最新版本是：4.10.0**）：

![微信截图_20240604084455.png](imags/微信截图_20240604084455.png)

到此，三大框架集成完毕。

### 4、导入自研yolov8的模型

在 app 的 main 目录下新建 assets
文件夹（一定要这个名字，别自己另辟蹊径），将Python导出的yolov8模型（后缀是 *.bin 和 *
.param，如果不是这俩后缀的自行查找解决方案）复制进去即可。暂时先不用管，备用。

### 5、JNI配置

#### 5.1、什么是JNI？

JNI（Java Native Interface），是方便Java/Kotlin调用C/C++等Native代码封装的一层接口。简单来说就是Java/Kotlin与C/C++沟通的桥梁。

#### 5.2、新建 [Yolov8ncnn.java](ap/src/main/java/com/pengxh/ncnn/yolov8/Yolov8ncnn.java)

在 **app/src/main/java/自己的包名** 目录下新建 Yolov8ncnn.kt（Yolov8ncnn.java 也是可以的），代码如下：

```kotlin
class Yolov8ncnn {
  companion object {
    init {
      System.loadLibrary("yolov8ncnn")
    }
  }

  /**
   * @param mgr 手机内存资源管理器
   * @param modelId 模型ID
   * @param useGpu 是否使用GPU
   * @param useClassify 是否使用分类模型
   * @param useSegmentation 是否使用分割模型
   * @param useDetect 是否使用检测模型
   * */
  external fun loadModel(
    mgr: AssetManager, modelId: Int, useGpu: Boolean, useClassify: Boolean,
    useSegmentation: Boolean, useDetect: Boolean
  ): Boolean

  /**
   * @param mgr 手机内存资源管理器
   * @param ids 多模型ID数组
   * @param useGpu 是否使用GPU
   * */
  external fun loadMultiModel(mgr: AssetManager, ids: IntArray, useGpu: Boolean): Boolean

  /**
   * @param facing 相机 0-前置镜头，1-后置镜头
   * */
  external fun openCamera(facing: Int): Boolean

  external fun closeCamera(): Boolean

  external fun setOutputWindow(
    surface: Surface, nativeObjAddr: Long, callBack: INativeCallback
  ): Boolean

  external fun onPause(): Boolean

  external fun onRestart(): Boolean
}
```

companion object（伴生对象，类似Java里面的static关键字）包裹的代码意思是C/C++代码编译之后动态链接库的名字（此时还没有，因为还没有添加C/C++代码）。另外四个方法和普通的Kotlin方法的区别在于全部都有external关键字修饰（Java里面是native），表明这几个方法需要调用C/C++代码，也就是前文提到的”桥梁“。此时代码会报错，是因为还没有在C/C++里面实现它们，先不用管。

#### 5.3、将本项目的 [ndkcamera.cpp](app/src/main/cpp/ndkcamera.cpp) 和 [ndkcamera.h](app/src/main/cpp/ndkcamera.h) 复制到自己项目

这是底层相机相关的代码逻辑，包括相机打开、关闭、预览、数据回调等，通用代码，无需修改。此时会爆一堆错误提示，别慌，先不用管。

#### 5.4、将本项目的 [yolo.cpp](app/src/main/cpp/yolo.cpp) 和 [yolo.h](app/src/main/cpp/yolo.h) 复制到自己项目

这两代码文件主要的功能是对相机预览产生的nv21数据进行处理，包括nv21转换、nv21转Mat矩阵、图像裁剪、灰度处理、调用模型检测目标、显示检测结果、回调等一些列操作，底层逻辑就在此实现。此时依旧会爆一堆错误提示，别慌，先不用管。

#### 5.5、将本项目的 [yolov8ncnn.cpp](app/src/main/cpp/yolov8ncnn.cpp) 复制到自己项目

此代码文件主要包括相机初始化，参数初始化。整体来说就是各种初始化以及状态管理。

#### 5.6、修改 app/src/main/cpp 目录下的 [CMakeLists.txt](app/src/main/cpp/CMakeLists.txt)

添加如下两行代码：

```cmake
add_library(yolov8ncnn SHARED yolov8ncnn.cpp yolo.cpp ndkcamera.cpp)

target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
```

最终的cmake代码如下图：
```cmake
project(yolov8ncnn)

cmake_minimum_required(VERSION 3.10)

set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv-mobile-4.10.0-android/sdk/native/jni)
find_package(OpenCV REQUIRED core imgproc)

set(ncnn_DIR ${CMAKE_SOURCE_DIR}/ncnn-20240102-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

add_library(yolov8ncnn SHARED yolov8ncnn.cpp yolo.cpp ndkcamera.cpp)

target_link_libraries(yolov8ncnn ncnn ${OpenCV_LIBS} camera2ndk mediandk)
```

#### 5.7、Java/C/C++代码调整

复制过去的yolov8ncnn.cpp文件，有四个函数一定是没有高亮的，如下图：
![微信截图_20240707213433.png](imags/微信截图_20240707213433.png)

此时需要将此函数根据情况修改为自己项目包名_函数名的方式，”.“用”_
“代替，比如：com.casic.test.Yolov8ncnn应改为Java_com_casic_test_Yolov8ncnn，改了之后就会发现，这四个函数已经高亮了，说明桥接代码已经生效。

* Java
  ![微信截图_20240707214218.png](imags/微信截图_20240707214218.png)

* Cpp
  ![微信截图_20240707214345.png](imags/微信截图_20240707214345.png)

可以看到Java和C++代码左侧已经出现相对应的代码标识。另外还有两个文件，就是setOutputWindow方法里面的DetectResult和INativeCallback，这俩都属于回调部分的代码，一个是数据模型类，一个是回调接口，直接复制即可。

如果没有出现以上效果的，先点击”Sync Now“，再”Build-Clean Project“，再”Build-Rebuild
Project“，再”Build-Refresh Linked C++ Projects“，最后关闭工程重新启动Android Studio，此时应该就没问题了。

### 6、JNI编码

此模块需要有能看懂的C/C++代码的的能力，以及非常熟练的使用Kotlin/Java的能力。

#### 6.1、修改yolov8ncnn.cpp

* 定义全局指针变量

```cpp
 static JavaVM *javaVM = nullptr;
```

* 修改JNI_OnLoad方法，在相机初始化之前添加一行如下代码：

```cpp
 javaVM = vm;
```

* 修改Java_com_casic_test_Yolov8ncnn_loadModel方法（注意自己的包名）

将model_types、target_sizes、mean_values、norm_values改为如下代码：

```cpp
//分割、分类、检测
const char *model_types[] = {"best-sim-opt-fp16", "model.ncnn", "yolov8s-detect-sim-opt-fp16"};
const int target_sizes[] = {320, 320, 320};
const float mean_values[][3] = {
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f},
        {103.53f, 116.28f, 123.675f}
};
const float norm_values[][3] = {
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
        {1 / 255.f, 1 / 255.f, 1 / 255.f},
        {1 / 255.f, 1 / 255.f, 1 / 255.f}
};
```

其中model_types里面的值是你yolov8模型去掉后缀后剩下的部分，一定要注意，否则会报错，找不到模型。

* 修改Java_com_casic_test_Yolov8ncnn_setOutputWindow方法（同样注意包名），在return前面加一行代码：

```cpp
g_yolo->initNativeCallback(javaVM, nativeObjAddr, native_callback);
```

以上这些，我在代码里面已经加好，注意下就可以了。有个值得注意的地方，在此文件的on_image_render函数，里面的注释我也写清楚了，可以根据需求选择draw和draw_fps，如果不需要，可以都注释掉，不影响后面的逻辑。

#### 6.2、修改Yolo.h

* 添加全局变量

```cpp
    JavaVM *javaVM;
    //Java传过来的Mat对象内存地址
    jlong j_mat_addr;
    //回调类
    jobject j_callback;
```

* 添加Java/C++初始化函数

```cpp
void initNativeCallback(JavaVM *vm, jlong nativeObjAddr, jobject pJobject);
```

#### 6.3、修改Yolo.cpp

* 修改generate_proposals函数

根据自己模型能够识别的目标种类修改此函数的num_class字段，比如，此处我已改为如下：

```cpp
const int num_class = 43;
```

* 实现自己在Yolo.h里面定义的setNativeCallback函数

```cpp
void Yolo::initNativeCallback(JavaVM *vm, jobject input, jlong nativeObjAddr, jobject pJobject) {
    javaVM = vm;

    /**
     * JNIEnv不支持跨线程调用
     * */
    JNIEnv *env;
    vm->AttachCurrentThread(&env, nullptr);
    j_mat_addr = nativeObjAddr;

    j_callback = env->NewGlobalRef(pJobject);
}
```

有个注意点，JNIEnv不支持跨线程调用，一定要注意，否则会报错，之前在Yolo.h定义的全局变量也需要在此处初始化。
以上这些，我在代码里面已经加好，如果要加自己的逻辑，知道在此处改就行了。

* 修改detect函数（划重点！划重点！划重点！）

其实在以上步骤完成时候就已经能把自定义的模型在Android端跑起来了（运行了一下，没效果？？？那是自然，因为 [MainActivity.kt](app/src/main/java/com/pengxh/ncnn/yolov8/MainActivity.kt)
还没有实现逻辑），已经可以检测目标了。

但是有缺陷，第一就是检测的结果只能在C++层面使用，Java/Kotlin层无法用，所以修改此函数的目的就是把检测结果回传到应用层，让应用层去做具体业务逻辑处理。第二个就是，C++底层只能渲染英文字符，中文的显示不出来或者显示乱码，当然也不是没有解决思路，需要交叉编译freetype2这个库，有兴趣的可以自行实现。

**知识点预热**

1. 基本类型签名

|  Java   |   JNI    | 签名  |
|:-------:|:--------:|:---:|
|  byte   |  jbyte   |  B  |
|  char   |  jchar   |  C  |
| double  | jdouble  |  D  |
|  float  |  jfloat  |  F  |
|   int   |   jint   |  I  |
|  short  |  jshort  |  S  |
|  long   |  jlong   |  J  |
| boolean | jboolean |  Z  |
|  void   |   void   |  V  |

2. 引用数据类型的转换

|   Java    |      JNI      |          签名           |
|:---------:|:-------------:|:---------------------:|
|   所有对象    |    jobject    |    L+classname +;     |
|   Class   |    jclass     |   Ljava/lang/Class;   |
|  String   |    jstring    |  Ljava/lang/String;   |
| Throwable |  jthrowable   | Ljava/lang/Throwable; |
| Object[]  | jobjectArray  |    [L+classname +;    |
|  byte[]   |  jbyteArray   |          [B           |
|  char[]   |  jcharArray   |          [C           |
| double[]  | jdoubleArray  |          [D           |
|  float[]  |  jfloatArray  |          [F           |
|   int[]   |   jintArray   |          [I           |
|  short[]  |  jshortArray  |          [S           |
|  long[]   |  jlongArray   |          [J           |
| boolean[] | jbooleanArray |          [Z           |

预热完毕，举几个例子：

```cpp
函数：int add(int a, int b)
签名：(II)I
说明：入参两个整型，返回值为整型

函数：String concat(String str1, String str2)
签名：(Ljava/lang/String;Ljava/lang/String;)Ljava/lang/String;
说明：入参两个String类型，返回值为String类型

[Ljava/lang/String;
表示 String 类型的一维数组
```

回归正题，回传结果给上层，理清步骤就很简单，用过Java反射的这里理解起来应该不难

1. 获取Java代码的回调接口，得到回调接口的jclass

```cpp
    JNIEnv *env;
    javaVM->AttachCurrentThread(&env, nullptr);
    jclass callback_clazz = env->GetObjectClass(j_callback);
```

注意：JNIEnv不支持跨线程，所以必须通过之前定义的全局指针变量 javaVM 得到当前线程的JNIEnv。

2. 根据或取到的jclass获取接口回调方法名，得到jmethodID

```cpp
jmethodID j_method_id = env->GetMethodID(callback_clazz, "onDetect", "(Ljava/util/ArrayList;)V");
```

3. 给回调入参的jobject设置值。此处只举个复杂点例子，基本类型的很简单就不展示了，具体返回值要看自己的逻辑

```cpp
          for (const auto &item: objects) {
                auto rect = item.rect;

                float array[5];
                array[0] = rect.x;
                array[1] = rect.y;
                array[2] = rect.x + rect.width;
                array[3] = rect.y + rect.height;
                array[4] = (float) item.label;

                jfloatArray result_array = env->NewFloatArray(5);
                env->SetFloatArrayRegion(result_array, 0, 5, array);

                //add
                env->CallBooleanMethod(segment_array_obj, arraylist_add, result_array);
          }
```

上面的代码意思是给float[]赋值，从签名”[F“可以看出来，然后将float[]数组通过反射添加进ArrayList

5. 发起回调

回调就很简单了，看清参数含义就行。意思就是在什么类里面调用什么方法，填入什么值

```cpp
env->CallVoidMethod(j_callback, j_method_id, arraylist_obj);
```

至此，JNI产生的目标检测结果已经回调到上层，上层可以接下来就可以用回调结果处理相应的业务逻辑。但是这里只能传常见的数据类型，还有一种数据无法回传上去，那就是图像的Mat矩阵，这个到后面会介绍。

### 7、Kotlin编码

#### 7.1、搭建界面布局

修改 app/src/main/res/layout/[activity_main.xml](app/src/main/res/layout/activity_main.xml) 如下：

![微信截图_20240707215753.png](imags/微信截图_20240707215753.png)

#### 7.2、初始化Yolov8ncnn和矩阵Mat对象，懒汉模式

```kotlin
private val yolov8ncnn by lazy { Yolov8ncnn() }
private val mat by lazy { Mat() }
```

#### 7.3、在initOnCreate中加载模型以及初始化OpenCV和SurfaceView

```kotlin
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
```

#### 7.4、在onResume里面打开相机

打开之前需要给应用授予相机的权限，否则会报错

```kotlin
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
```

#### 7.5、实现SurfaceHolder.Callback回调

只需要实现surfaceChanged方法，surfaceCreated和surfaceDestroyed不必管

```kotlin
override fun surfaceChanged(holder: SurfaceHolder, format: Int, width: Int, height: Int) {
  yolov8ncnn.setOutputWindow(holder.surface, mat.nativeObjAddr, this)
}
```

#### 7.6、实现INativeCallback回调

```kotlin
override fun onDetect(output: ArrayList<FloatArray>) {
  if (output.isEmpty()) {
    return
  }

  //暂停算法
  yolov8ncnn.onPause()
  weakReferenceHandler.sendEmptyMessageDelayed(2024082901, 1000)
}
```

调用单模型（检测）获取目标所对应的场景，然后再根据场景切换为多模型（分割+检测）

```kotlin
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
```

处理多模型结果

```kotlin
    override fun onSegmentation(segmentationOutput: ArrayList<FloatArray>, detectOutput: ArrayList<FloatArray>) {
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
```

此时detectView会报错，因为这是个自定义控件，可先注释掉，后面再说。

这里还有个隐藏的细节，那就是mat，哪来的值？

setOutputWindow里面有个入参，mat.nativeObjAddr，这个是Java/Kotlin层通过JNI往C++传入内存地址（可以理解为指针），然后在在 [yolo.cpp](app/src/main/cpp/yolo.cpp)
里面给此指针赋值，那么，这样就实现了Mat矩阵数据回传的效果。 在 int Yolo::detect(const cv::Mat &rgb,
std::vector<Object> &objects, float prob_threshold,
float nms_threshold) 的 return 前面加上如下代码：

```cpp
    auto *res = (cv::Mat *) j_mat_addr;
    res->create(rgb.rows, rgb.cols, rgb.type());
    memcpy(res->data, rgb.data, rgb.rows * rgb.step);
```

#### 7.7、在onPause里面关闭相机

```kotlin
override fun onPause() {
    super.onPause()
    yolov8ncnn.closeCamera()
}
```

自此，ncnn + yolov8 + opencv 这三个框架已完成在Android端的移植。将报错的地方先注释掉

```kotlin
// binding.detectView.updateTargetPosition(results)
```

* 结果回调：

![微信截图_20240604141117.png](imags/微信截图_20240604141117.png)

* Mat矩阵转PNG结果：
  ![微信截图_20240604141334.png](imags/微信截图_20240604141334.png)

#### 7.8、实现自定义控件

如果没有特殊要求，直接复制过去即可，但是需要将里面的 classNames 改为自己模型对应的类别，虽然不会报错，但是会显示成错误的类别，注意下就行了。
然后修改app/src/main/res/layout/[activity_main.xml](app/src/main/res/layout/activity_main.xml)
里面的内容如下：
![微信截图_20240707220511.png](imags/微信截图_20240707220511.png)

红框里面改成自己包名，然后编译运行即可。这样既方便了后续逻辑处理，也规避了C++不方便渲染中文的尴尬，效果如下：

![20240604142258.png](imags/20240604142258.png)

### 以上就是完整的ncnn + yolov8 + opencv 移植到Android端的详细步骤，希望能帮到需要此需求的人！最后感谢各个开源者的辛勤付出（排名不分先后）：

* [ncnn](https://github.com/Tencent/ncnn)
  ：腾讯开源的一个为手机端极致优化的高性能神经网络前向计算框架，无第三方依赖，跨平台，ncnn主要基于C++和caffe。
* [yolov8](https://github.com/ultralytics/ultralytics): YOLO(You Only Look
  Once）是一种流行的物体检测和图像分割模型，由华盛顿大学的约瑟夫-雷德蒙（Joseph Redmon）和阿里-法哈迪（Ali
  Farhadi）开发，YOLO 于 2015 年推出。
* [opencv-mobile](https://github.com/nihui/opencv-mobile): The minimal opencv for Android, iOS, ARM
  Linux, Windows, Linux, MacOS, WebAssembly
* [OpenCV](https://github.com/opencv/opencv): OpenCV（open source computer vision
  library）是一个基于BSD许可（开源）发行的跨平台计算机视觉库，可以运行在Linux、Windows、Android和Mac
  OS操作系统上。