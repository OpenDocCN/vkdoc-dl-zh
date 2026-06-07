# 4. 应用滤镜概述

在 Clojure 语言中，你可以直接将一组变换应用于视频流的 `Mat` 对象，而无需使用任何额外的样板代码。我强烈建议你即使只看一下 origami 示例中最基础的部分，这些示例可在 README 中找到：

```
https://github.com/hellonico/origami/blob/master/README.md#support-for-opencv-412-is-in
```

清单 4-1 展示了如何在一个管道中加载图片、将其转换为灰度图并应用 `canny` 函数。这甚至可能让你想尝试 OpenCV 的 Clojure 版本。

```clojure
(require
'[opencv4.utils :as u]
'[opencv4.core :refer :all])
(->
(imread "doc/cat_in_bowl.jpeg")
(cvt-color! COLOR_RGB2GRAY)
(canny! 300.0 100.0 3 true)
(bitwise-not!)
(u/resize-by 0.5)
(imwrite "doc/canny-cat.jpg"))
```

清单 4-1
读取、转灰度、Canny 边缘检测、调整大小并保存

不过，这是一本关于 Java 的书，所以让我们看看如何在 Java 中应用相同的概念。

这里我们引入了滤镜管道的概念，其中每个滤镜对 `Mat` 对象执行一项操作。

以下是滤镜可以执行的一些操作示例：

- 将 `Mat` 对象转换为灰度图
- 应用 Canny 效果
- 寻找边缘
- 铅笔素描
- Instagram 滤镜，如棕褐色或复古风格
- 进行背景减除
- 使用 Haar 目标检测或颜色检测来检测猫或人脸
- 运行神经网络并识别物体

以下是我们将引入的两种 Java 类型，用于实现这些概念：

- `Filter` 接口，仅包含一个函数 `apply(Mat in)`，并返回一个 `Mat` 对象，就像函数式编程中一样。
- `Pipeline` 类，它本身也是一个 `Filter`，接受一个类列表或已实例化的滤镜列表。当调用 `apply` 时，它会逐个应用这些类。

清单 4-2 展示了（简单的）`Filter` 接口。

```java
import org.opencv.core.Mat;
public interface Filter {
    public Mat apply(Mat in);
}
```

清单 4-2
`Filter` 接口

清单 4-3 展示了实现 `Pipeline` 类的示例，你只需通过逐个调用不同的滤镜来组合它们。

```java
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import org.opencv.core.Mat;
public class Pipeline implements Filter {
    List<Filter> filters;
    public Pipeline(Class<?>... __filters) {
        List<Class<?>> _filters = (List<Class<?>>) Arrays.asList(__filters);
        this.filters = _filters.stream().map(i -> {
            try {
                return (Filter) Class.forName(i.getName()).newInstance();
            } catch (Exception e) {
                return null;
            }
        }).collect(Collectors.toList());
    }
    public Pipeline(Filter... __filters) {
        this.filters = (List<Filter>) Arrays.asList(__filters);
    }
    @Override
    public Mat apply(Mat in) {
        Mat dst = in.clone();
        for (Filter f : filters) {
            dst = f.apply(dst);
        }
        return dst;
    }
}
```

清单 4-3
多个滤镜

请注意，这里直接在类中使用了已弃用版本的 `newInstance` 函数。这可能不会调用你真正想要的构造函数，但对于本书中的示例来说，它已经足够好了。

当然，到目前为止，`Filter` 和 `Pipeline` 还没有做太多事情，所以让我们在接下来的部分中回顾一些基本示例。

### 应用基础滤镜

在本节中，我们将介绍几个基础滤镜的示例。

#### 灰度滤镜

滤镜最明显的用途是将 `Mat` 对象从彩色转换为灰度。在 OpenCV 中，这是通过使用 `Imgproc` 类中的 `cvtColor` 函数完成的。

省略包定义，我们将几页前的网络摄像头代码与标准的 `cvtColor` 包装器结合在一个类中，该类实现了前面介绍的 `Filter` 接口（清单 4-4）。

```java
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;
import org.opencv.videoio.VideoCapture;
import origami.ImShow;
import origami.Origami;
```

清单 4-4
流上的灰度滤镜

```java
public class WebcamWithFilters {
    public static void main(final String[] args) {
        Origami.init();
        final VideoCapture cap = new VideoCapture(0);
        final ImShow ims = new ImShow("Camera", 800, 600);
        final Mat buffer = new Mat();
        Filter gray = new Gray();
        while (cap.read(buffer)) {
            ims.showImage(gray.apply(buffer));
        }
        cap.release();
    }
}
class Gray implements Filter {
    public Mat apply(final Mat img) {
        final Mat mat1 = new Mat();
        Imgproc.cvtColor(img, mat1, Imgproc.COLOR_RGB2GRAY);
        return mat1;
    }
}
```

你可以直接在树莓派上运行此代码，如果你站在网络摄像头前，你会得到类似图 4-1 中我得到的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig1_HTML.jpg](img/490964_1_En_4_Fig1_HTML.jpg)

图 4-1

不仅我的头发，整张图片都变成了灰色

#### 边缘保留滤镜

以同样的方式，我们可以为 OpenCV 的 `Photo` 类中的 `EdgePreserving` 函数实现一个包装器。这在许多不同的应用中用于平滑和去除图片中不需要的线条。例如，清单 4-5 实际上只是对 `edgePreservingFilter` 函数的一个基本调用。

```java
import org.opencv.photo.Photo;
class EdgePreserving implements Filter {
    public int flags = Photo.RECURS_FILTER;
    // int flags = NORMCONV_FILTER;
    public float sigma_s = 60;
    public float sigma_r = 0.4f;
    public Mat apply(Mat in) {
        Mat dst = new Mat();
        Photo.edgePreservingFilter(in, dst, flags, sigma_s, sigma_r);
        return dst;
    }
}
```

清单 4-5
重新实现为 `Filter` 的 `EdgePreservering` 类

你可以通过修改 `WebcamWithFilters` 的 main 方法来使用这个新滤镜，如清单 4-6 所示。

```java
Filter filter = new EdgePreserving();
while (cap.read(buffer)) {
    ims.showImage(filter.apply(buffer));
}
```

清单 4-6
修改后的主函数以使用边缘保留滤镜

现在让我们继续执行新代码。同样，如果你正对着网络摄像头，你应该会得到类似于图 4-2 中我的样子。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig2_HTML.jpg](img/490964_1_En_4_Fig2_HTML.jpg)

图 4-2

边缘保留滤镜

#### Canny 边缘检测

OpenCV 世界中另一个有用的滤镜是应用 Canny 效果，这是一种快速有效的方法，用于在 `Mat` 对象中查找轮廓和形状。将 `Canny` 作为滤镜的快速实现如清单 4-7 所示。

```java
class Canny implements Filter {
    public boolean inverted = true;
    public int threshold1 = 100;
    public int threshold2 = 200;
    @Override
    public Mat apply(Mat in) {
        Mat dst = new Mat();
        Imgproc.Canny(in, dst, threshold1, threshold2);
        if (inverted) {
            Core.bitwise_not(dst, dst, new Mat());
        }
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2RGB);
        return dst;
    }
}
```

清单 4-7
`Filter` 中的 OpenCV Canny 边缘检测

图 4-3 显示了将 Canny 滤镜应用于主循环的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig3_HTML.jpg](img/490964_1_En_4_Fig3_HTML.jpg)

图 4-3

网络摄像头流上的 Canny 滤镜

#### 调试（再谈）

我们再次来谈谈调试的注意事项。如果在 `WebcamWithFilters` 的主捕获循环中添加断点，你将能够访问过滤器的所有不同字段。如图 4-4 所示，让我们将反转布尔值改为 `false`。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig4_HTML.jpg](img/490964_1_En_4_Fig4_HTML.jpg)

图 4-4：实时更新过滤器参数

然后，让我们移除断点并正常重新启动代码执行。图 4-5 展示了直接更改过滤器值如何立即改变正在运行的代码以及屏幕上显示的 `Mat` 对象的颜色。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig5_HTML.jpg](img/490964_1_En_4_Fig5_HTML.jpg)

图 4-5：非反转的 Canny 过滤器

在实现你自己的过滤器时，最好将最有影响力的变量作为类的字段保留，这样你就不会被大量数值淹没，但仍能访问到重要的变量。

#### 组合过滤器

你可能在前面的章节中已经意识到，你会想要了解每个过滤器的性能。

性能实际上取决于从视频文件或直接从摄像头设备读取时，每秒能处理的帧数。

接下来我们将执行以下操作：

- 直接在图像上显示帧率
- 使用本章前面定义的 `Pipeline` 类，将灰度过滤器与帧率过滤器组合起来

我们已经有了灰度过滤器的代码，因此我们将直接进入显示每秒帧数（FPS）的代码。

我一直以为可以通过 OpenCV 的 `VideoCapture` 属性集来访问帧率值。不幸的是，我们几乎总是只能使用一个硬编码的值，而不是屏幕上实际显示的值。

因此，清单 4-8 中的 FPS 实现是一个小的变通方法，它基于自过滤器生命周期开始以来已显示的帧数进行简单的算术运算。

最后，它使用 `putText` 将文本直接应用到帧上。对于简单的用例来说，这已经足够好了。

```java
class FPS implements Filter {
    long start = System.currentTimeMillis();
    int count = 0;
    Point org = new Point(50, 50);
    int fontFace = Imgproc.FONT_HERSHEY_PLAIN;
    double fontScale = 4.0;
    Scalar color = new Scalar(0, 0, 0);
    int thickness = 3;
    public Mat apply(Mat in) {
        count++;
        String text = "FPS: " + count / (1 + ((System.currentTimeMillis() - start) / 1000));
        Imgproc.putText(in, text, org, fontFace, fontScale, color, thickness);
        return in;
    }
}
```

清单 4-8：FPS 过滤器

现在让我们回到将流上显示的 FPS 与灰度过滤器组合起来的问题。

你会很高兴地知道，在 `WebcamWithFilters` 的 `main()` 函数中，唯一需要更新的行是实例化过滤器的那一行，如下所示：

```java
Filter filter = new Pipeline(Gray.class, FPS.class);
```

当你再次从树莓派上运行示例时，你将得到类似于图 4-6 所示的结果。将两个过滤器一起应用，我在树莓派上通常能得到大约每秒 15 帧。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig6_HTML.jpg](img/490964_1_En_4_Fig6_HTML.jpg)

图 4-6：灰度过滤器与 FPS 的组合

### 应用类似 Instagram 的滤镜

严肃的工作到此为止。让我们稍作休息，用一些类似 Instagram 的滤镜来玩一玩。

#### 颜色映射

让我们从使用 `ImgProc` 中的 OpenCV `colormap` 函数开始这个有趣的章节。我们将 `colormap` 的参数移到构造函数中，如清单 4-9 所示，这样我们就可以通过调试屏幕来更新它。

```java
class Color implements Filter {
    int colormap = 0;
    public Color(int colormap) {
        this.colormap = colormap;
    }
    public Color() {
        this.colormap = Imgproc.COLORMAP_INFERNO;
    }
    public Mat apply(Mat img) {
        Mat threshed = new Mat();
        Imgproc.applyColorMap(img, threshed, colormap);
        return threshed;
    }
}
```

清单 4-9：颜色映射

要实例化过滤器，我们需要传入想要使用的颜色映射，因此这直接在构造函数中完成。这里我们需要的是实例化后的过滤器，而不仅仅是它的类，因此我们使用 `Pipeline` 类的第二个构造函数，将实例化后的 `Filter` 对象传递给构造函数。

```java
Filter filter = new Pipeline(new Color(Imgproc.COLORMAP_INFERNO), new FPS());
```

当你执行此代码时，你会得到类似于图 4-7 的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig7_HTML.jpg](img/490964_1_En_4_Fig7_HTML.jpg)

图 4-7：地狱火

#### 阈值

阈值是另一个有趣的滤镜，通过应用 `Imgproc` 的 `threshold` 函数实现。它对 `Mat` 对象的每个数组元素应用一个固定级别的阈值。

阈值滤镜最初的目的是分割图片中的元素，例如通过移除不需要的元素来去除图片的噪点。它通常不用于 Instagram 滤镜，但效果看起来不错，并且可以给你带来一些创意灵感。

清单 4-10 展示了如何实现阈值滤镜。

```java
class Thresh implements Filter{
    int sensitivity = 100;
    int maxVal = 255;
    public Thresh() {
    }
    public Thresh(int _sensitivity) {
        this.sensitivity = _sensitivity;
    }
    public Mat apply(Mat img) {
        Mat threshed = new Mat();
        Imgproc.threshold(img, threshed, sensitivity, maxVal, Imgproc.THRESH_BINARY);
        return threshed;
    }
}
```

清单 4-10：应用阈值

图 4-8 显示了结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig8_HTML.jpg](img/490964_1_En_4_Fig8_HTML.jpg)

图 4-8：应用阈值产生燃烧效果

#### 棕褐色

让我们再次使用古老的（双关语）棕褐色效果，如清单 4-11 所示。

```
class Sepia implements Filter {
public Mat apply(Mat source) {
Mat kernel = new Mat(3, 3, CvType.CV_32F);
kernel.put(0, 0,
0.272, 0.534, 0.131,
0.349, 0.686, 0.168,
0.393, 0.769, 0.189);
Mat destination = new Mat();
Core.transform(source, destination, kernel);
return destination;
}
}
```

清单 4-11：棕褐色

当与视频流一起使用时，棕褐色效果会给你带来如图 4-9 所示的输出。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig9_HTML.jpg](img/490964_1_En_4_Fig9_HTML.jpg)

图 4-9：棕褐色效果

#### 卡通效果

这种卡通效果的简单实现会提取原始图像中定义特征的重要线条，在应用平滑和模糊效果后，对每个像素值进行阈值处理。然后将这些操作的结果进行组合，如代码清单 4-12 所示。

```
class Cartoon implements Filter {
public int d = 17;
public int sigmaColor = d;
public int sigmaSpace = 7;
public int ksize = 7;
public double maxValue = 255;
public int blockSize = 19;
public int C = 2;
public Mat apply(Mat inputFrame) {
Mat gray = new Mat();
Mat co = new Mat();
Mat m = new Mat();
Mat mOutputFrame = new Mat();
Imgproc.cvtColor(inputFrame, gray, Imgproc.COLOR_BGR2GRAY);
Imgproc.bilateralFilter(gray, co, d, sigmaColor, sigmaSpace);
Mat blurred = new Mat();
Imgproc.blur(co, blurred, new Size(ksize, ksize));
Imgproc.adaptiveThreshold(blurred, blurred, maxValue, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY,
blockSize, C);
Imgproc.cvtColor(blurred, m, Imgproc.COLOR_GRAY2BGR);
Core.bitwise_and(inputFrame, m, mOutputFrame);
return mOutputFrame;
}
}
```

代码清单 4-12：卡通滤镜

将卡通滤镜应用于视频流，会得到如图 4-10 所示的效果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig10_HTML.jpg](img/490964_1_En_4_Fig10_HTML.jpg)

图 4-10：卡通效果

#### 铅笔效果

我非常喜欢铅笔效果，它是通过调用 OpenCV 核心 `Photo` 类中的 `pencilSketch` 方法实现的。不幸的是，在树莓派上实时应用这个效果速度太慢了。不过，它几乎不需要任何实现工作就能产生相当漂亮的效果。请参见代码清单 4-13。

```
class PencilSketch implements Filter {
float sigma_s = 60;
float sigma_r = 0.07f;
float shade_factor = 0.05f;
boolean gray = false;
@Override
public Mat apply(Mat in) {
Mat dst = new Mat();
Mat dst2 = new Mat();
pencilSketch(in, dst, dst2, sigma_s, sigma_r, shade_factor);
return gray ? dst : dst2;
}
}
```

代码清单 4-13：铅笔效果

应用此效果后，会得到如图 4-11 所示的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig11_HTML.jpg](img/490964_1_En_4_Fig11_HTML.jpg)

图 4-11：铅笔素描

哇哦。我们已经有了不少可以随时使用和娱乐的效果。在 origami 仓库中还有其他一些效果可用，当然你也可以贡献自己的效果，但现在，让我们继续学习更严肃的目标检测。

### 执行目标检测

*目标检测* 是指使用不同的编程算法在图片中查找物体的概念。这是一项长期以来由人类完成的任务，对于没有大脑的计算机来说相当困难。但随着技术的进步，这种情况最近已经发生了变化。

在本章的这部分内容中，我们将回顾不同的计算机视觉技术，以识别图像中的物体，而无需任何关于其内容的先验信息。具体来说，我们将回顾以下内容：

*   使用简单的轮廓绘制滤镜

*   通过颜色检测物体

*   使用 Haar 分类器

*   使用模板匹配

*   使用像 Yolo 这样的神经网络

这些示例的难度大致是递增的，因此最好按照列表顺序尝试。

#### 移除背景

移除背景是一种可以用来移除场景中不必要杂物的技术。你试图寻找的物体可能不是静止的，并且很可能在一组图片或视频流中移动。为了有效地移除杂物，算法需要能够区分两个 `Mat` 对象，并使用某种短期记忆来区分移动的物体（前景）和背景中的标准场景物体。

在 OpenCV 中，有两个易于使用的 `BackgroundSubtractor` 类可供使用。它们的介绍和完整解释可以在以下网站上找到：

```
https://docs.opencv.org/master/d1/dc5/tutorial_background_subtraction.html
```

基本上，你向背景减法器提供越来越多的帧，它就能检测出哪些是前景中移动的物体，哪些不是。

代码清单 4-14 非常容易理解；只需注意不要将 `BackgroundSubtractor` 类中的 `apply` 函数与我们 `Filter` 接口中的 `apply` 函数混淆。

```
class BackgroundSubtractor implements Filter {
boolean useMOG2 = true;
BackgroundSubtractor backSub;
double learningRate = 1.0;
boolean showMask = true;
public BackgroundSubtractor() {
if (useMOG2) {
backSub = Video.createBackgroundSubtractorMOG2();
} else {
backSub = Video.createBackgroundSubtractorKNN();
}
}
@Override
public Mat apply(Mat in) {
Mat mask = new Mat();
backSub.apply(in, mask);
Mat result = new Mat();
if (showMask) {
Imgproc.cvtColor(mask, result, Imgproc.COLOR_GRAY2RGB);
return result;
} else {
in.copyTo(result, mask);
return result;
}
}
}
```

代码清单 4-14：`BackgroundSubtractor` 类

通过调用构造函数加载滤镜，你应该会得到类似于图 4-12 的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig12_HTML.jpg](img/490964_1_En_4_Fig12_HTML.jpg)

图 4-12：移除背景

使用 KNN 背景减法器

一旦这个滤镜运行起来，尝试切换到基于 KNN 的 `BackgroundSubtractor`，看看速度（观察帧率）和结果准确性的差异。

#### 通过轮廓检测

OpenCV 第二个最基本的功能是能够在图像中找到轮廓。轮廓滤镜使用了 `Imgproc` 类中的 `findContours` 函数。

`findContours` 通常在你先执行以下步骤时效果更好：

*   将输入的 `Mat` 对象转换为灰度图

*   应用 Canny 滤镜

这两个步骤已添加到代码清单 4-15 中；然后我们在一个使用 `zeros` 函数创建的黑色 `Mat` 对象上绘制轮廓。

```
class Contours implements Filter {
private int threshold = 100;
public Mat apply(Mat srcImage) {
Mat cannyOutput = new Mat();
Mat srcGray = new Mat();
Imgproc.cvtColor(srcImage, srcGray, Imgproc.COLOR_BGR2GRAY);
Imgproc.Canny(srcGray, cannyOutput, threshold, threshold * 2);
List contours = new ArrayList();
Mat hierarchy = new Mat();
Imgproc.findContours(cannyOutput, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
Mat drawing = Mat.zeros(cannyOutput.size(), CvType.CV_8UC3);
for (int i = 0; i < contours.size(); i++) {
Scalar color = new Scalar(256, 150, 0);
Imgproc.drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, new Point());
}
return drawing;
}
}
```

代码清单 4-15：检测轮廓

为轮廓使用管道

细心的读者会注意到，由于前面的代码使用了 `Pipeline` 类，实际上写成下面这样会更优雅：

```
Pipeline(new Canny(), new Gray(), new Contours())
```

其中 `Contours` 滤镜只负责提取轮廓。试试看！

将轮廓滤镜应用于 Marcel 的视频会得到一种艺术效果（图 4-13）。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig13_HTML.jpg](img/490964_1_En_4_Fig13_HTML.jpg)

图 4-13：使用 OpenCV 的轮廓检测功能

移除第一个 Canny 滤镜并比较

这里有一个练习给你：尝试移除转换为灰度图和应用 Canny 滤镜这两个步骤，然后将结果与原始结果进行比较。

#### 通过颜色检测

在 OpenCV 中，一张图片或一个 `Mat` 对象通常采用红/绿/蓝（RGB）色彩空间（实际上在 OpenCV 中准确来说是蓝/绿/红）。如果你理解为每个像素的每个通道都被赋予了一个值，这就很容易理解了。要查看这些通道的可能值，你可以查阅以下网站：

```
https://www.rapidtables.com/web/color/RGB_Color.html
```

这种色彩空间的问题在于，亮度和对比度的信息与颜色本身的信息混杂在一起。

当在 `Mat` 对象中寻找特定颜色时，我们会切换到一个名为 HSV（色相、饱和度、明度）的色彩空间。在这个色彩空间中，颜色直接对应色相值。

色相值通常在 0 到 360 之间，类似于一个圆柱体上的度数。OpenCV 的方案略有不同，其范围被除以了 2（这样在内存中占用更少空间）。表 4-1 列出了色相值的范围。

**表 4-1** OpenCV 中的色相值

| 颜色 | 色相范围 |
| --- | --- |
| 红色 | 0 到 30 *以及* 150 到 180 |
| 绿色 | 30 到 90 |
| 蓝色 | 90 到 150 |

清单 4-16 转换了色彩空间，并使用 `inRange` 检查目标颜色范围内的色相值，并在示例末尾再次使用 `findContours` 添加了一些魔法来正确绘制形状。

```
class ColorDetector implements Filter {
Scalar minColor, maxColor;
public ColorDetector(Scalar minColor, Scalar maxColor) {
this.minColor = minColor;
this.maxColor = maxColor;
}
@Override
public Mat apply(Mat input) {
Mat array255 = new Mat(input.height(), input.width(), CvType.CV_8UC1);
array255.setTo(new Scalar(255));
Mat distance = new Mat(input.height(), input.width(), CvType.CV_8UC1);
List lhsv = new ArrayList(3);
Mat circles = new Mat();
Mat hsv_image = new Mat();
Mat thresholded = new Mat();
Mat thresholded2 = new Mat();
Imgproc.cvtColor(input, hsv_image, Imgproc.COLOR_BGR2HSV);
Core.inRange(hsv_image, minColor, maxColor, thresholded);
Imgproc.erode(thresholded, thresholded, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
Imgproc.dilate(thresholded, thresholded, Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(8, 8)));
Core.split(hsv_image, lhsv);
Mat S = lhsv.get(1);
Mat V = lhsv.get(2);
Core.subtract(array255, S, S);
Core.subtract(array255, V, V);
S.convertTo(S, CvType.CV_32F);
V.convertTo(V, CvType.CV_32F);
Core.magnitude(S, V, distance);
Core.inRange(distance, new Scalar(0.0), new Scalar(200.0), thresholded2);
Core.bitwise_and(thresholded, thresholded2, thresholded);
Imgproc.GaussianBlur(thresholded, thresholded, new Size(9, 9), 0, 0);
List contours = new ArrayList();
Imgproc.HoughCircles(thresholded, circles, Imgproc.CV_HOUGH_GRADIENT, 2, thresholded.height() / 8, 200, 100, 0, 0);
Imgproc.findContours(thresholded, contours, thresholded2, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
Imgproc.drawContours(input, contours, -2, new Scalar(10, 0, 0), 4);
return input;
}
}
class RedDetector extends ColorDetector {
public RedDetector() {
super(new Scalar(0, 100, 100), new Scalar(10, 255, 255));
}
}
```

清单 4-16：检测红色

将此滤镜应用于玫瑰视频的结果如图 4-14 所示。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig14_HTML.jpg](img/490964_1_En_4_Fig14_HTML.jpg)

**图 4-14** 检测红玫瑰

**实现一个检测蓝色的滤镜**

查看表 4-1 中的色相值，你会发现实现一个搜索蓝色的滤镜并不困难。这留作练习供你完成。

#### 通过 Haar 检测

正如你在第 1 章中所见，你可以使用基于 Haar 的分类器来识别 `Mat` 对象中的物体和/或人物。代码与你之前看到的几乎相同，只是额外强调了我们要寻找的形状的数量和大小。

具体来说，以下代码展示了如何使用两个尺寸作为参数来指定我们寻找物体的最小尺寸和最大尺寸。

```
classifier.detectMultiScale(input, faces, 1.1, 2, -1, new Size(100, 100), new Size(500, 500));
```

因此，清单 4-17 附带了一个额外的 main 示例函数，展示了如何使用不同的 XML 文件作为 Haar 分类器检测的参数。

### 基于 Haar 分类器的检测

```java
public class DetectWithHaar {
    public static void main(String[] args) {
        Origami.init();
        VideoCapture cap = new VideoCapture(0);
        Mat buffer = new Mat();
        ImShow ims = new ImShow("Camera", 800, 600);
        Filter filter = new Pipeline(new Haar("haarcascades/haarcascade_frontalface_default.xml"), new FPS());
        while (cap.grab()) {
            cap.retrieve(buffer);
            ims.showImage(filter.apply(buffer));
        }
        cap.release();
    }
}

class Haar implements Filter {
    private CascadeClassifier classifier;
    Scalar white = new Scalar(255, 255, 255);

    public Haar(String path) {
        classifier = new CascadeClassifier(path);
    }

    public Mat apply(Mat input) {
        MatOfRect faces = new MatOfRect();
        classifier.detectMultiScale(input, faces, 1.1, 2, -1, new Size(100, 100), new Size(500, 500));
        for (Rect rect : faces.toArray()) {
            Imgproc.putText(input, "Face", new Point(rect.x, rect.y - 5), 3, 5, white);
            Imgproc.rectangle(input, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height),
                    white, 5);
        }
        return input;
    }
}
```

清单 4-17 基于 Haar 分类器的检测

如果你家里有猫，或者正在使用示例中的猫视频，当你将其应用于网络摄像头流时，应该会得到如图 4-15 所示的结果。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig15_HTML.jpg](img/490964_1_En_4_Fig15_HTML.jpg)

**图 4-15** 寻找猫

### 使用其他 Haar 定义

示例中还有其他用于 Haar 级联的 XML 文件。请随意使用其中一个来检测人物、眼睛或笑脸作为练习。

#### 检测上的透明叠加层

在前面的示例中绘制矩形时，您可能想知道是否可以在检测到的形状上绘制矩形以外的其他内容。

清单 4-18 展示了如何通过加载一个将叠加在检测到的形状位置上的遮罩来实现这一点。这基本上就是您在智能手机应用中一直使用的功能。

请注意，在 `drawTransparency` 函数中，关于透明度层有一个技巧。叠加层的遮罩是使用 `IMREAD_UNCHANGED` 作为加载标志加载的；您必须使用此标志，否则透明度层会丢失。

一旦您获得了透明度层，就可以在复制叠加层时将其用作遮罩，从而复制您想要的 `Mat` 对象的精确像素。

```java
class FunWithHaar implements Filter {
    CascadeClassifier classifier;
    Mat mask;
    Scalar white = new Scalar(255, 255, 255);

    public FunWithHaar(String path) {
        classifier = new CascadeClassifier(path);
        mask = Imgcodecs.imread("masquerade_mask.png", Imgcodecs.IMREAD_UNCHANGED);
    }

    void drawTransparency(Mat frame, Mat transp, int xPos, int yPos) {
        List<Mat> layers = new ArrayList<>();
        Core.split(transp, layers);
        Mat mask = layers.remove(3);
        Core.merge(layers, transp);
        Mat submat = frame.submat(yPos, yPos + transp.rows(), xPos, xPos + transp.cols());
        transp.copyTo(submat, mask);
    }

    public Mat apply(Mat input) {
        MatOfRect faces = new MatOfRect();
        classifier.detectMultiScale(input, faces);
        Mat maskResized = new Mat();
        for (Rect rect : faces.toArray()) {
            Imgproc.resize(mask, maskResized, new Size(rect.width, rect.height));
            int adjusty = (int) (rect.y - rect.width * 0.2);
            try {
                drawTransparency(input, maskResized, rect.x, adjusty);
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
        return input;
    }
}
```

清单 4-18 添加叠加层

根据所使用的透明 `Mat` 对象，您可能需要调整位置，否则您应该会得到类似图 4-16 所示的结果。最后，您可以为您的视频流增添一些威尼斯狂欢节的感觉！

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig16_HTML.jpg](img/490964_1_En_4_Fig16_HTML.jpg)

图 4-16 添加一个神秘面具作为叠加层

使用蝙蝠侠面具

我实际上尝试过，但没能找到一个合适的蝙蝠侠面具用作视频流的叠加层。也许您可以发送给我一个包含适当 `Mat` 叠加层的代码来帮助我！

#### 通过模板匹配进行检测

使用 OpenCV 进行模板匹配非常简单。它简单到可能应该更早地出现在检测方法的顺序中。模板匹配意味着在一个 `Mat` 中寻找另一个 `Mat`。OpenCV 有一个名为 `matchTemplate` 的超强函数可以做到这一点。

清单 4-19 主要围绕使用 `matchTemplate`。请注意在从 `matchTemplate` 返回的结果上使用 `Core.minMaxLoc`。它用于定位最佳得分的索引，并且在运行神经网络时会再次使用。

```java
class Template implements Filter {
    Mat template;

    public Template(String path) {
        this.template = Imgcodecs.imread(path);
    }

    @Override
    public Mat apply(Mat in) {
        Mat outputImage = new Mat();
        Imgproc.matchTemplate(in, template, outputImage, Imgproc.TM_CCOEFF);
        MinMaxLocResult mmr = Core.minMaxLoc(outputImage);
        Point matchLoc = mmr.maxLoc;
        Imgproc.rectangle(in, matchLoc, new Point(matchLoc.x + template.cols(), matchLoc.y + template.rows()),
                new Scalar(255, 255, 255), 3);
        return in;
    }
}
```

清单 4-19 模式匹配

现在，让我们找一个装有 ReSpeaker 的盒子，就像图 4-17 中所示的那样，因为我们在下一章中会用到这个扬声器，而我现在找不到它了。让我们用 OpenCV 来帮我们找到它。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig17_HTML.jpg](img/490964_1_En_4_Fig17_HTML.jpg)

图 4-17 模板

通过 OpenCV 的模板匹配进行检测出奇地快速和准确，如图 4-18 所示。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig18_HTML.jpg](img/490964_1_En_4_Fig18_HTML.jpg)

图 4-18 找到扬声器的盒子

在图 4-18 中很难看到帧率，但在 Raspberry Pi 4 上，它实际上大约在每秒 10 到 15 帧之间。

#### 通过 Yolo 进行检测

这是本章介绍的最后一个检测方法。假设我们想要应用一个训练好的神经网络来识别流中的物体。在经过一些计算能力有限的硬件测试后，我使用 Yolo/Darknet 以及在 Coco 数据集上训练的可免费获取的 Darknet 网络获得了相当快速的结果。

在随机输入上使用神经网络的优势在于，大多数训练好的网络都相当稳健，并且能给出良好的结果，在接近实时的流上准确率达到 80% 到 90%。

训练是使用神经网络最困难的部分。在本书中，我们将仅限于在 Raspberry Pi 上运行检测代码，而不是训练。您可以在 Darknet/Yolo 网站上找到如何组织图片以重新训练网络的步骤。

在 OpenCV 中使用 Darknet 实现物体检测的步骤如下：

1.  从配置文件和权重文件中加载网络。

2.  找到该网络的输出层/节点，因为结果将在这里产生。输出层是指那些没有连接到更多输出层的层。

3.  将 `Mat` 对象转换为网络所需的 blob。Blob 是一个图像或一组图像，经过调整以匹配网络在大小、通道顺序等方面期望的格式。

4.  然后我们运行网络，这意味着我们将 blob 输入网络，并检索标记为输出层的层的值。

5.  对于结果中的每一行，我们实际上会获得每个预期可识别特征的置信度值。在 Coco 中，网络被训练为能够识别 80 种不同的可能物体，例如人、自行车、汽车等。

6.  然后我们再次使用 `MinMaxLocResult` 来获取最可能被识别物体的索引，如果该索引的值大于 0，我们就保留它。

7.  每个结果行中的前四个值实际上是描述检测到物体所在框的四个值，因此我们提取这四个值，并保留矩形及其标签的索引。

8.  在绘制所有框之前，我们通常还会使用 `NMSBoxes`，它会移除重叠的框。大多数情况下，重叠的框是对同一物体多次阳性检测的多个版本。

9.  最后，我们绘制剩余的矩形，并添加识别出的物体的标签。

清单 4-20 展示了作为过滤器实现的基类 `YoloDetector` 的完整代码。

```java
class YoloDetector implements Filter {
    final static Size sz = new Size(416, 416);
    List outBlobNames;
    Net net;
    List layers;
    List labels;

    List getOutputsNames(Net net) {
        List layersNames = net.getLayerNames();
        return net.getUnconnectedOutLayers().toList().stream().map(i -> i - 1).map(layersNames::get)
                .collect(Collectors.toList());
    }

    public YoloDetector(String modelWeights, String modelConfiguration) {
        net = Dnn.readNetFromDarknet(modelConfiguration, modelWeights);
        layers = getOutputsNames(net);
        try {
            labels = Files.readAllLines(Paths.get(LABEL_FILE));
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    @Override
    public Mat apply(Mat in) {
        findShapes(in);
        return in;
    }

    final int IN_WIDTH = 416;
    final int IN_HEIGHT = 416;
    final double IN_SCALE_FACTOR = 0.00392157;
    final int MAX_RESULTS = 20;
    final boolean SWAP_RGB = true;
    final String LABEL_FILE = "yolov3/coco.names";

    void findShapes(Mat frame) {
        Mat blob = Dnn.blobFromImage(frame, IN_SCALE_FACTOR, new Size(IN_WIDTH, IN_HEIGHT), new Scalar(0, 0, 0),
                SWAP_RGB);
        net.setInput(blob);
        List outputs = new ArrayList();
        for (int i = 0; i < layers.size(); i++) {
            outputs.add(new Mat());
        }
        net.forward(outputs, layers);
        postProcess(frame, outputs);
    }

    private void postProcess(Mat frame, List outs) {
        List tmpLocations = new ArrayList();
        List tmpClasses = new ArrayList();
        List tmpConfidences = new ArrayList();
        int w = frame.width();
        int h = frame.height();
        for (Mat out : outs) {
            final float[] data = new float[(int) out.total()];
            out.get(0, 0, data);
            int k = 0;
            for (int j = 0; j < out.height(); j++) {
                Mat result = out.row(j);
                Core.MinMaxLocResult mm = Core.minMaxLoc(result);
                if (mm.maxVal > 0) {
                    float center_x = data[k + 0] * w;
                    float center_y = data[k + 1] * h;
                    float width = data[k + 2] * w;
                    float height = data[k + 3] * h;
                    float left = center_x - width / 2;
                    float top = center_y - height / 2;
                    tmpClasses.add((int) mm.maxLoc.x);
                    tmpConfidences.add((float) mm.maxVal);
                    tmpLocations.add(new Rect((int) left, (int) top, (int) width, (int) height));
                }
                k += out.width();
            }
        }
        annotateFrame(frame, tmpLocations, tmpClasses, tmpConfidences);
    }

    private void annotateFrame(Mat frame, List tmpLocations, List tmpClasses,
                               List tmpConfidences) {
        MatOfRect locMat = new MatOfRect();
        MatOfFloat confidenceMat = new MatOfFloat();
        MatOfInt indexMat = new MatOfInt();
        locMat.fromList(tmpLocations);
        confidenceMat.fromList(tmpConfidences);
        Dnn.NMSBoxes(locMat, confidenceMat, 0.1f, 0.1f, indexMat);
        for (int i = 0; i < indexMat.total() && i < MAX_RESULTS; ++i) {
            int idx = (int) indexMat.get(i, 0)[0];
            int labelId = tmpClasses.get(idx);
            Rect box = tmpLocations.get(idx);
            String label = labels.get(labelId);
            annotateOne(frame, box, label);
        }
    }

    private void annotateOne(Mat frame, Rect box, String label) {
        Imgproc.rectangle(frame, box, new Scalar(0, 0, 0), 2);
        Imgproc.putText(frame, label, new Point(box.x, box.y), Imgproc.FONT_HERSHEY_PLAIN, 4.0, new Scalar(0, 0, 0), 3);
    }
}
```

*清单 4-20* 基于神经网络的检测

现在，你可以使用不同的可用网络运行自己的一组目标检测和实验。清单 4-21 展示了如何加载每个主要的基于 Yolo 的网络。

```java
class Yolov2 extends YoloDetector {
    public Yolov2() {
        super("yolov2/yolov2.weights", "yolov2/yolov2.cfg");
    }
}

class TinyYolov2 extends YoloDetector {
    public TinyYolov2() {
        super("yolov2-tiny/yolov2-tiny.weights", "yolov2-tiny/yolov2-tiny.cfg");
    }
}

class Yolov3 extends YoloDetector {
    public Yolov3() {
        super("yolov3/yolov3.weights", "yolov3/yolov3.cfg");
    }
}

class TinyYolov3 extends YoloDetector {
    public TinyYolov3() {
        super("yolov3-tiny/yolov3-tiny.weights", "yolov3-tiny/yolov3-tiny.cfg");
    }
}
```

*清单 4-21* 不同 Yolo 网络的 Java 类和构造函数

在树莓派上运行时，Yolo v3 可以检测里斯本繁忙街道上的汽车和行人（图 4-19），以及更多的猫（图 4-20）。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig20_HTML.jpg](img/490964_1_En_4_Fig20_HTML.jpg)

*图 4-20* Yolo v3 检测猫

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig19_HTML.jpg](img/490964_1_En_4_Fig19_HTML.jpg)

*图 4-19* Yolo v3 检测汽车和行人

如你所见，标准 Yolo v3 的帧率实际上非常低。

当使用 Yolo v3 Tiny 进行此实验时，你实际上可以达到接近每秒 5 到 6 帧，这仍然略低于实时要求，但结果仍然具有非常好的准确性。参见图 4-21。

![../images/490964_1_En_4_Chapter/490964_1_En_4_Fig21_HTML.jpg](img/490964_1_En_4_Fig21_HTML.jpg)

*图 4-21* TinyYoloV3 检测猫

> *你知道我的方法。它是建立在对琐事的观察之上的。*
> 
> ——阿瑟·柯南·道尔，《博斯科姆比溪谷秘案》（1891）

你已经读到了本章的最后几行，这是一段漫长的旅程，你了解了在树莓派上使用 Java 和 OpenCV 进行目标检测的大部分概念。

具体来说，你学习了以下内容：

- 如何设置树莓派以进行实时目标检测编程
- 如何使用过滤器和管道执行图像和实时视频处理
- 如何为 `Mat` 实现一些基本过滤器，直接用于来自外部设备和基于文件的视频的实时视频流
- 如何通过类似 Instagram 的过滤器增加一些趣味
- 如何使用过滤器和管道实现多种目标检测技术
- 如何运行一个在 Coco 数据集上训练、可在树莓派上实时使用的神经网络

在下一章中，你将了解 Rhasspy（一个语音识别系统），以连接本章介绍的概念，并将其应用于家庭、办公室或猫舍的自动化。