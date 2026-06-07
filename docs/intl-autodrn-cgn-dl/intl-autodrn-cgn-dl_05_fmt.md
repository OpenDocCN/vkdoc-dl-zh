# 5. 向我们的仿真添加传感器

在第四章中，我们创建了我们的第一个 Gazebo 漫游车仿真。我们手动驾驶漫游车在空旷的环境中移动。我们使用 URDF 文件来定义我们的漫游车外观，并使用启动文件来定义 Gazebo 环境中的漫游车控制。现在我们想要添加传感器，以便漫游车能够“看到”障碍物。不幸的是，没有一点帮助，URDF 文件会变得非常大且难以维护。这时就出现了 Xacro XML 语言扩展（Xacro 代表 XML 宏）。它通过简单的代码块（称为插件）帮助添加标准传感器。插件支持如激光雷达、雷达、摄像头等传感器。我们的激光雷达传感器为漫游车开发了第一距离测量能力。我们将开发 Python 脚本，通过差速驱动插件直接控制漫游车的轮子。最后，我们将通过 Python 脚本进行键盘控制（遥控）漫游车的实验。

## 目标

以下是为成功完成本章所需达成的目标：

+   学习 XML 宏编程语言（Xacro）

+   使用 Xacro 语言重新制作漫游车模型，以实现简单性和可扩展性

+   为传感器和电机开发 Xacro 编程语言例程

+   在 RViz 和 Gazebo 仿真器中测试漫游车

+   控制漫游车

## XML 宏编程语言

Xacro 编程语言是一种宏语言，用于开发可维护和模块化的 XML 文件。我们将简要概述 Xacro，以优化 URDF 机器人和 Gazebo 仿真文件。

宏是一个简单的将“名称”替换为“值”的过程。最直接的宏是属性替换。属性通常是一个在文件中使用的常量。用“名称”占位符替换常量允许程序员在单个位置定义属性，从而更容易维护。在该单个位置更改值会更改文件中所有位置的命名常量。格式如下：

| **推荐样式** `<xacro:property name="propertyName" value="propertyValue"/>` |
| --- |
| **等效块** |
| `<xacro:property>` `<name="propertyName">` `<value="propertyValue" />` `</xacro:property>` |

属性名称在 XML 文件中找到，并将属性值替换进去。值可以是简单的数字或字符串。以下示例展示了如何声明和使用属性：

属性通过替换美元花括号（`${}`）内的名称来替换几何表达式。我们将使用属性块来定义漫游车底盘的尺寸。如果我们想要改变我们漫游车的大小，我们只需更改属性块。

更复杂的替换允许一个属性名称有多个值。以下是一个使用属性块为几何表达式（笛卡尔坐标（x，y，z）和方向（翻滚，俯仰，偏航）值）放置值的示例：

我们在 `front_left_origin` 属性块中定义 `xyz` 值为“0.3 0 0”和 `rpy = "0 0 0"`。如果我们得到一辆带有新底盘的新漫游器，我们可以通过更改 `front_left_origin` 属性块来更新整个系统。我们不需要追踪所有文件中“`front_left_origin`”的每个实例以进行所需更改。

我们可以使用 Xacro 进行简单的数学表达式，用于传感器处理或用于漫游器组件尺寸。仅支持基本的算术和变量替换。例如：

Xacro 的 `${}` 也是一个 Python 算术库的扩展。库中定义的常量（`pi`）和函数（`radians`，度到弧度的转换）都是可访问的。

Xacro 有条件块（`if..unless`），类似于编程语言（`if..else`），如 Python。Xacro 条件块的语法格式如下：

```py
">

">

```

条件块必须始终返回一个布尔值；即，`true`（1）或 `false`（0）。任何其他返回值都会抛出异常，无效的返回值。在以下五个语句中，第一行将 `var` 定义为 `useit`。第二行检查 `var` 是否等于 `useit` 并返回 `true`。第三行查看 `var` 的子字符串并返回 `true`。第四行定义了一个名为 `allowed` 的值数组。第五行检查 `"1"` 是否在数组中允许。在确定值时，我们使用了双引号“ ”，而在使用字符串常量时使用了单引号‘ ‘。

## 更多 XML 示例

为了展示 XML 和 Xacro 的强大功能，我们将创建一些示例宏。此 XML 示例声明了一个关节和链接对。动态关节名为 `caster_front_left_joint`，其轴值为 `xyz="0 0 1"`。链接组件 `caster_front_left` 定义了坐标 (`xyz="0 1 0"`)、方向 `{`rpy="0 0 0"`}、颜色（名称="yellow"）和质量常数（0.1）。此外，我们还定义了惯性（转动惯量）值。

```py

0.1

```

为了说明 Xacro 的功能，下一个宏创建了一个接收两个宏作为参数的宏，“first”和“second”！这两个宏在别处定义。这两个参数按顺序插入创建了一个更大的宏“reorder”。这些多个块参数按指定的线性顺序执行。这些执行的参数可能将左/右、前/后轮作为单个控制宏中的宏传递，以生成四个单独的控制宏（左/前，右/前，左/后，右/后）。

宏可以包含其他宏，称为嵌套。外部宏首先展开，然后是内部宏。完整的描述超出了本文的范围。要在主宏中包含嵌套宏，请执行以下操作：

```py
This macro searches the project directories for a filename called `ai_rover_remastered_plugins.xacro` and inserts that file into the current file. The `ai_rover_remastered_plugins.xacro` file stores the filenames of the plug-ins for the rover, such as sensor plugins (LiDAR, radar, etc.) and the two-wheeled differential drive (DDC). Built into the DDC are simple keyboard commands to control a differential drive.The Rover RevisitedWe will review and remaster the design of our two-wheeled differential-drive rover system by transitioning from standard XML to a Xacro URDF description file. The differential-drive system for the rover is the most common type of drive system for a robot and navigates by independently controlling the velocities of each of the wheels. Since the rover only utilizes two wheels and a non-moving static caster, we should consider refactoring in Xacro. The significant advantage of Xacro is that it is easier to maintain, implement, test, and expand Gazebo simulations. Xacro’s modular design and Python scripting mean we can quickly test the virtual rover’s routines. Xacro helps transition our designs from the Gazebo simulations to the physical rover’s existing software and hardware implementation.Recall that a differential-drive system navigates the environment by controlling each wheel’s velocity independently. The left- and right-front wheels control (or actuate) navigation, and their velocities determine the rover’s driving path. For additional information, please refer to [`https://en.wikipedia.org/wiki/Differential_wheeled_robot`](https://en.wikipedia.org/wiki/Differential_wheeled_robot).Modular Designed RoverIf we are going to maintain this increasingly complex software project, we should simplify its structure. At this moment, putting all the source code in one file seems to make life simple. But soon, we will be adding additional hardware and software that will make modifying the underlying changes challenging to track. So, while we are at the beginning of our design, let us simplify the packaging of our code for later. We will divide our original code into modules that have one (and exactly one) responsibility. Our original URDF code divides into the following Xacro modules:*   Dimensions `(dimensions.xacro`), which tracks the constants for the physical components of our rover

    *   Chassis (`chassisInertia.xacro`), which tracks the physics related to the body of our rover

    *   Wheels (`wheelsInertia.xacro`), which tracks the physics related to each wheel of our rover

    *   Caster (`casterInertia.xacro`), which tracks the physics related to the caster of our rover

    *   Laser (`laserDimensions`.`xacro`), which tracks the physics and geometry layout of the LiDAR’s housing box

    *   Camera (`cameraDimensions`.`xacro`), which tracks the physics and geometry layout of the camera’s housing box

    *   IMU (`IMUDimensions`.`xacro`), which tracks the physics and geometry layout of the inertial measurement unit’s (IMU) housing box

    After adding new sensors, we will need to modify the dimensions file and add our `<sensor>Inertia.xacro` file. Logically, our software project now looks like Figure 5-1.![](img/494112_1_En_5_Fig1_HTML.png)A block diagram of the Rover. At the top left a i underscore rover underscore remastered dot xacro is written. Dimensions are divided into chassis, wheels, caster, and more sensors. There are three horizontal dots between the block of the caster and more sensors.Figure 5-1Modular design of our roverTo begin our code refactoring, we need to create the rover sub-directory `ai_rover_remastered`. The terminal commands are as follows:

```

$ mkdir -p catkin_ws/src/ai_rover_remastered

$ cd catkin_ws/src/ai_rover_remastered

$ mkdir launch urdf config



Now we should have three sub-directories, `launch`, `urdf`, and `config`, in the `ai_rover_remastered` directory. Recall that having the `launch`, `urdf`, and `config` sub-directories prepare the `ai_rover_remastered` directory as an ROS package. **Note that URDF supports XML and Xacro.**

## dimensions.xacro

Next, create the `dimensions.xacro` file in the `catkin_ws/src/ai_rover_remastered/urdf` directory:

We have created our first Xacro URDF file for the rover. It defines the rover’s dimensions and serves as a framework for different rover models. The property block tags define numeric and string constants, such as the rover’s base length, `base_length`, which is statically defined as 0.16. Any Xacro files that include this file will have access to this global constant. Changing the value to 0.20 in `dimensions.xacro` will have a ripple effect across all files. Without Xacro, we would have to search ALL the files for "0.16," decide whether it referred to `base_length` or `base_width`, and manually change it—a very error-prone procedure. (Compare to Chapter 4’s URDF.)

## chassisInertia.xacro

Next, we create `chassisInertia.xacro` to define the movement of our rover. Notice that it "includes" the `dimensions.xacro` file. This `chassisInertia.xacro` file also includes our final `ai_rover_remastered.xacro` file. While this nesting of files seems complicated at first, separating the structure from the functionality allows us to change one without modifying the other.

## wheels.xacro

The following components are the two identical wheels. Because they are similar, we need to define the concept (class) once and instantiate (object) it twice by offsetting them by the appropriate amount specified in the generated `ai_rover_remastered.urdf` file.

There are many examples of the object-oriented programming (OOP) paradigm in this short script. The `${prefix}` macro creates separate left and right wheel links and joints connected to the base link. You should note the OOP instantiation (left and right wheel joints and links using the `${prefix}` macro) and aggregation (connecting the left and right wheels on either side of the `base_link`). The `wheel_joint_offset` determines how far horizontally from the base center is the wheel offset.

## casterInertia.xacro

Using macros, we can model the mass and moment of inertia for other components on the rover. For instance, we model the caster (`casterInertia.xacro`) as a "spherical" wheel that rotates in any direction, similar to the physical rover.

We can now create a caster wheel with any geometry required to model the physical caster wheel accurately. This source code creates a caster wheel link (`caster_wheel`) and a joint (`caster_wheel_joint`) connected to the base link.

## laserDimensions.xacro

The Gazebo Laser Range Finding Scanner Plug-in (LRFP) determines the shape and geometry of previously unexplored areas. The LRFP simulates a LiDAR sensor; in our case, the Hokuyo LiDAR system. A LiDAR sensor uses laser pulses to measure distances to objects within the environment, helping to determine their geometry. We can then generate a map, helping the rover to navigate and avoid obstacles. LiDAR systems are one of the primary sources of odometry in modern robotics. The following code (`laserDimensions.xacro`) places the `sensor_laser` with its correct geometric dimensions on top of the rover chassis to receive messages from the Xacro sensor file:

## cameraDimensions.xacro

The integrated rover camera captures and processes image data to sense and avoid obstacles. For simplicity, we make the camera dimensions the same as the LiDAR sensor and place it in front of and below the LiDAR sensor. The `cameraDimensions.xacro` defines a simple camera housing with a fixed `camera_link` in front of the chassis (`base_link`):

## IMUDimensions.xacro

In this section, we define the inertial measurement unit (IMU) plug-in. The IMU data captures the rover’s speed (straight and turning) and orientation (attitude) relative to the environment. The IMU heading and attitude data is processed so the rover can maneuver in an environment. The SLAM process needs the IMU and wheel encoder data to help accurately and precisely outline boundaries, walls, and obstacles (Chapter 7). The IMU housing dimensions will be small compared to those of the camera and LiDAR. To simplify the physics, the IMU location will be in the origin (xyz=0,0,0) of the rover. To avoid the LiDAR sweep, we reduce the IMU’s size. The `IMUDimensions.xacro` defines a simple IMU housing with a fixed `IMU_link` relative to the chassis (`base_link`):

## Gazebo Plug-ins

A plug-in is a section of source code compiled as a C++ library and inserted into any Gazebo simulation. Python is an "interpreted" language (slow), while C++ is compiled (fast). All plug-ins have direct access to all the functions of the physics engine of Gazebo.

Plug-ins are helpful because they:
*   let developers control and enhance Gazebo features;
*   are self-contained software routines for simulations; and
*   can be inserted and removed from a running system.

Previous versions of Gazebo utilized integrated controllers, which behaved in much the same way as modern Gazebo plug-ins. Consequently, no enhancements were possible with these controllers. Current Gazebo plug-ins are now far more flexible and allow users to program what functionality to include in their simulations.

You should only use a plugin when:
*   you want to alter a simulation programmatically, such as responding to simulation events; and
*   you want a fast interface to Gazebo without the overhead of the transport layer, such as an interface. An example of an interface would be to control the speed and direction of our rover.

### Plug-in Types

There are currently six types of Gazebo plug-ins, as follows:
1.  World (catacombs, etc.)
2.  Robot Model (rover)
3.  Sensor (LiDAR, IMU, camera, etc.)
4.  System (differential-drive controller, etc.)
5.  Visual (laser view blue field in Figure [5-8], etc.)
6.  GUI (rover controls)

Each plug-in is attached to a specific "object" in the Gazebo environment. For example, the robot model plug-in is attached to and controls the rover in Gazebo. Similarly, the world plug-in is attached to a catacombs environment, and each sensor has a sensor plug-in. The system plug-in is specified in the command line and loads the wheels and caster physics configuration in the differential-drive controller. The visual plug-in is automatically loaded and shows colors as defined in the different Xacro files; e.g., the "blue" wheels, "red" LiDAR box, etc. The GUI plug-in is also automatically loaded and connects the Gazebo controls to control objects (and sub-objects such as joints) in the environment (move, turn, rotate, etc.). These controls ARE NOT the same as simulation controls using Teleops.

### Differential-Drive Controller (DDC) Plug-in

The differential-drive controller (DDC) plug-in is a system plug-in that ties the physics engine to the rover. The DDC uses the `wheels.xacro` definitions to attach the individually defined parts of the DDC to the physics engine. This connection will be used by the Teleops keystrokes to realistically control the movement of the rover.

```xml
<plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <alwaysOn>true</alwaysOn>
    <updateRate>20</updateRate>
    <leftJoint>left_wheel_joint</leftJoint>
    <rightJoint>right_wheel_joint</rightJoint>
    <wheelSeparation>${wheel_separation}</wheelSeparation>
    <wheelDiameter>${wheel_radius * 2}</wheelDiameter>
    <robotNamespace>/ai_rover_remastered</robotNamespace>
    <commandTopic>cmd_vel</commandTopic>
    <odometryTopic>odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <robotBaseFrame>base_footprint</robotBaseFrame>
</plugin>
```


# 集成DDC插件与传感器

在本节中，我们将DDC插件集成到重新设计的漫游车模型中。我们这样做是为了设计、开发和测试漫游车的手动控制。简单的手动控制使漫游车前进或后退，并向左或向右转弯。由于车轮独立转动，它们的相对速度也可以不同。这些手动控制测试差速驱动，以便在添加自主导航控制时，控制逻辑是正确的。此外，差速驱动的初步测试强化了基本的ROS控制概念。

我们通过将内置的DDC插件集成到Xacro漫游车模型中来实现这一点。在当前目录中创建一个名为`ai_rover_remastered_plugins.xacro`的脚本文件，内容如下：

```xml
<!--
Now we are ready to add control to our robot. We will add a new plug-in to our Xacro file, and we will add a differential-drive plug-in to our robot. The new tag looks as follows:
-->
<plugin name="differential_drive_controller" filename="libgazebo_ros_diff_drive.so">
    <alwaysOn>false</alwaysOn>
    <legacyMode>false</legacyMode>
    <updateRate>20</updateRate>
    <leftJoint>left_wheel_joint</leftJoint>
    <rightJoint>right_wheel_joint</rightJoint>
    <wheelSeparation>${wheel_separation}</wheelSeparation>
    <wheelDiameter>${wheel_radius * 2}</wheelDiameter>
    <commandTopic>/ai_rover_remastered/base_controller/cmd_vel</commandTopic>
    <odometryTopic>/ai_rover_remastered/base_controller/odom</odometryTopic>
    <odometryFrame>odom</odometryFrame>
    <robotBaseFrame>base_footprint</robotBaseFrame>
</plugin>
```

让我们简要回顾一下这个文件。我们将检查的第一行是`filename`行。文件名是包含差速驱动控制器插件实现的`gazebo_ros`库名。我们将快速回顾此插件的以下定义标签：

*   插件名称是`differential_drive_controller`，位于库`libgazebo_ros_diff_drive.so`中。
    *   `<alwaysOn>`标签允许机器人接收速度命令；默认设置为`false`。
    *   `<legacyMode>`标签为`false`，不允许我们交换左右车轮。
    *   `<updateRate>`标签为20 Hz，是发送给控制器的信息频率。
    *   `<leftJoint>`标签是左关节的名称。
    *   `<rightJoint>`标签是右关节的名称。
    *   `<wheelSeparation>`标签是一个车轮中心到另一个车轮中心的距离，单位为米。通常默认为0.34米。
    *   `<wheelDiameter>`标签是每个车轮的直径。通常，每个车轮的直径相同，默认设置为0.15米。
    *   `<commandTopic>`标签用于接收来自用户、深度学习或认知AI控制架构的`geometry_msgs`或`Twist`消息命令。
    *   `<odometryTopic>`标签用于发布`nav_msgs`或`odometry`消息。
    *   `<odometryFrame>`标签默认为里程计坐标系。
    *   `<robotBaseFrame>`标签是用于计算里程计的漫游车坐标系，默认为`base_footprint`。

创建插件脚本时，`<plugin>`块和定义必须位于`<gazebo>`块内。

最后，该插件发布漫游车的里程计`<odom>`。里程计使用传感器数据（车轮编码器）来估计漫游车随时间的位置变化，并计算漫游车相对于其起始位置的当前位置。漫游车的每次移动都会触发一次传感器读数，用于更新内部地图位置。这种计算可能对由速度测量随时间积分引起的误差和不确定性来源极其敏感。

## 激光插件

激光插件是一个传感器插件。它是一个通用激光器，根据我们在`laserDimensions.xacro`中定义的实际激光规格进行了修改。

现在我们已经将DDC插件和模拟LiDAR传感器组件的几何结构添加到底盘中，我们必须将其连接到LRFP。LFRP提供了Hokuyo LiDAR系统的内部逻辑、行为和特性。以下代码访问LRFP的Gazebo插件库，并将LiDAR传感器设置为Hokuyo特性：

```xml
<plugin name="sensor_laser" filename="libgazebo_ros_laser.so">
    <topicName>/ai_rover_remastered/laser_scan/scan</topicName>
    <frameName>sensor_laser</frameName>
    <robotNamespace>/</robotNamespace>
    <bodyName>base_link</bodyName>
    <topicName>/scan</topicName>
    <frameName>laser_link</frameName>
    <xyz>0 0 0</xyz>
    <rpy>0 0 0</rpy>
    <updateRateHZ>20</updateRateHZ>
    <range>1440</range>
    <minRange>1</minRange>
    <minAngle>-3.14159</minAngle>
    <maxAngle>3.14159</maxAngle>
    <resolution>0.10</resolution>
    <maxRange>30.0</maxRange>
    <noise>0.01</noise>
    <noiseType>gaussian</noiseType>
    <mean>0.0</mean>
    <stddev>0.01</stddev>
</plugin>
```

LiDAR传感器系统将激光扫描数据发布到`/ai_rover_remastered/sensor_laser/scan`主题。TF坐标系订阅`sensor_laser`链接，该链接将LiDAR传感器模型与漫游车模型的其余部分集成。（如果我们还将LiDAR传感器系统从Hokuyo更改为其他系统，则必须更改`range, sample_rate, min_angle, max_angle, resolution,`和`signal_noise`参数以匹配新的LiDAR系统。）在集成链接和关节Xacro URDF以及LiDAR插件代码后，使用以下终端shell命令运行更新的漫游车模型：

```bash
$ roslaunch ai_rover_remastered ai_rover_remastered_gazebo.launch
```

运行后，您应该会看到类似于图5-2的显示。请注意，传感器具有无限范围，可以在一个像素厚的水平圆中看到360°，但在传感器附近有一个盲点。LiDAR无法检测到此盲点中的任何对象。

![img/494112_1_En_5_Fig2_HTML.jpg](img/494112_1_En_5_Fig2_HTML.jpg)
*图5-2 漫游车和LiDAR传感器：蓝色区域是传感器覆盖范围。灰色圆圈是LiDAR盲点*

在环境中放置一个立方体对象会在立方体后面显示一个灰色区域（图5-3）。这个灰色区域是另一个盲点。蓝色区域是LiDAR传感器扫描，显示没有遇到对象。同样，传感器具有无限范围。

![img/494112_1_En_5_Fig3_HTML.jpg](img/494112_1_En_5_Fig3_HTML.jpg)
*图5-3 漫游车和LiDAR传感器运行（Gazebo）*

LiDAR传感器在RViz环境中通过`sensor_laser/scan`主题发布数据（图5-4）。

![img/494112_1_En_5_Fig4_HTML.jpg](img/494112_1_En_5_Fig4_HTML.jpg)
*图5-4 LiDAR传感器扫描主题*

从漫游车的角度来看，LiDAR传感器将立方体形状的边界“看作”一条粗红线（图5-5）。LiDAR传感器发布激光已与对象相交的信息。

![img/494112_1_En_5_Fig5_HTML.jpg](img/494112_1_En_5_Fig5_HTML.jpg)
*图5-5 漫游车将立方体“看作”边界线（RViz）*

现在我们将为漫游车集成Gazebo相机插件。此插件将允许我们查看漫游车在探索环境时“看到”的内容。我们模拟的ROS相机为漫游车提供图像数据，用于对象识别、跟踪和操作任务。Noetic ROS目前支持单目和立体相机。为了简化本书，我们将在ROS/Gazebo/RViz设置中仅使用模拟单目相机，在GoPiGo3漫游车上仅使用物理单目相机。我们可以使用立体相机来创建环境的SLAM（同时定位与建图），但这只会增加本书的复杂性。我们只需要单目相机来定位和识别漫游车前方在环境中面对的对象。我们将仅使用LiDAR/SLAM配置来感知和避开障碍物，并定位环境中的异常，以便漫游车进一步探索和检查。

## 相机插件

相机插件是一个传感器插件，它将来自Gazebo的模拟相机图像连接起来，然后在RViz中显示它们。RViz GUI现在可以“显示”漫游车在Gazebo环境中“看到”的内容。

```xml
<plugin name="camera" filename="libgazebo_ros_camera.so">
    <alwaysOn>true</alwaysOn>
    <updateRateHZ>30.0</updateRateHZ>
    <cameraName>ai_rover_remastered/camera1</cameraName>
    <imageTopicName>image_raw</imageTopicName>
    <cameraInfoTopicName>camera_info</cameraInfoTopicName>
    <frameName>camera_link</frameName>
    <hackBaseline>0.07</hackBaseline>
    <distortionK1>0.0</distortionK1>
    <distortionK2>0.0</distortionK2>
    <distortionK3>0.0</distortionK3>
    <distortionT1>0.0</distortionT1>
    <distortionT2>0.0</distortionT2>
    <focalLength>1.3962634</focalLength>
    <horizontalFov>1.57</horizontalFov>
    <imageWidth>800</imageWidth>
    <imageHeight>800</imageHeight>
    <imageFormat>R8G8B8</imageFormat>
    <noiseMean>0.0</noiseMean>
    <noiseStddev>0.007</noiseStddev>
    <clip>300</clip>
    <noiseType>gaussian</noiseType>
    <robotNamespace>/</robotNamespace>
</plugin>
```


# Once we have inserted the Gazebo camera plug-in in the `ai_rover_remastered_plugins.xacro`, we should make certain that the camera plugin is indeed operational. This requires that we set the correct parameters in RViz to accept the incoming `image_raw` from the Gazebo simulation, which requires that we first start the Gazebo simulation launch file, followed by the RViz launch file. Please look to the Gazebo and RViz launch file sections of this chapter. To accept the `image_raw` from the Gazebo simulation, we must first go to the display options and select Add, and then select the camera option. Next, add `/ai_rover_remastered/camera1/image_raw` to the image topic menu for the camera under the Displays menu. The rover now “sees” a spherical object in Gazebo by “seeing” that same spherical object from the perspective of the rover in the RViz Camera window (Figure 5-6).

![](img/494112_1_En_5_Fig6_HTML.jpg)

Two screenshots of Gazebo and Rviz window. The left window has a spherical object at the bottom right and the same spherical object is at the center right in the right window.

**Figure 5-6** Rover’s camera “seeing” the sphere in Gazebo (left) and Rviz (right)

## IMU Plug-in

The IMU plug-in is a sensor plug-in. It connects the rover location to the global environment. Think of the rover as having a local coordinate system, while the environment has a global coordinate system. When the rover moves “1 space forward” in its local system, it moves from `<x,y>` to `<x+1, y>` in the global environment. The plug-in also maps the internal acceleration to the global environment to see it in Gazebo (purple arrow).

An IMU must aide the robot’s navigation tasks to allow genuine autonomy for the rover. The IMU (inertial measurement unit) sensor must measure and report the rover’s speed (accelerometer), direction, acceleration, specific force, angular rotation rate (gyroscope), and the magnetic field (magnetometer) surrounding the rover in all three directions (x, y, and z). We will need both the IMU and wheel encoder values to estimate the robot’s 6D pose and position in maps generated with Simultaneous Localization and Mapping (SLAM). The IMU can also combine input from several different sensor types to estimate output movement accurately.

We do this by integrating the IMU plug-in into the Xacro rover model. Create a script file called `ai_rover_remastered_plugins.xacro` in the current directory with the following content:

```xml
<plugin name="imu_plugin" filename="libgazebo_ros_imu.so">
    <alwaysOn>true</alwaysOn>
    <bodyName>IMU_link</bodyName>
    <topicName>imu</topicName>
    <serviceName>imu_service</serviceName>
    <gaussianNoise>0.0</gaussianNoise>
    <updateRate>20.0</updateRate>
</plugin>
```

Let us briefly review this file. The first line we will examine is the `filename` line. The filename is the `libgazebo_ros_imu.so` library and contains the plug-in implementation for the IMU sensor plug-in. We will quickly review the following defined tags for this plug-in:

*   The plug-in name is `imu_plugin` and is located in the library `libgazebo_ros_diff_drive.so`.
*   The `<alwaysOn>` tag allows the IMU to send data.
*   The `<bodyName>` tag is set to `IMU_link`, the IMU object link, a child link to `base_link`, which is the rover chassis.
*   The `<topicName>` tag is the message tag from IMU.
*   The `<serviceName>` tag is the message from the `IMU_service`.
*   The `<gaussianNoise>` tag is set to the value of zero, which means there is no gaussian noise to the LIDAR sensor simulations. This might need to be changed to closely reflect the explored environment.
*   The `<updateRate>` tag is the frequency of sensor updates for the LIDAR, which in the case of the source code is set to 20 hertz.

The `<plugin>` block and definitions must be in a `<gazebo>` block when creating a plug-in script.

## Visuals Plug-in

The visuals plug-in is (obviously) a visual plug-in. The material colors defined in the Xacro files are only used in RViz. The colors have to be redefined in our `visuals.xacro` plug-in to be displayed Gazebo.

```xml
<material name="Gazebo/Orange"/>
<material name="Gazebo/Blue"/>
<material name="Gazebo/Blue"/>
```

## Putting It All Together

We have now defined the individual components of the physical rover and their associated plug-ins for Gazebo. To complete the construction of our rover, we need to “glue” them together. First, we create the plug-in file (`ai_rover_remastered_plugins.xacro`) and then the rover model (`ai_rover_remastered.xacro`), which includes the plug-ins.

### ai_rover_remastered_plugins.xacro

Now that we have created our plug-ins for each rover component, we have to combine them in a single module and load them into Gazebo and RViz. To do this, we “include” the individual plug-in files into an `ai_rover_remastered_plugins.xacro` file.

### ai_rover_remastered.xacro

Finally, we bring all of these separate components together in the `ai_rover_remastered.xacro` file. This file “glues” or constructs the individual parts (chassis, wheels, caster, etc.) into a cohesive whole and complete rover.

Now that we have remastered the rover into modules using Xacro, we need to convert it to an URDF file and then verify the converted file has no errors. To do this, we run the following two terminal commands in the URDF directory:

```bash
$ rosrun xacro xacro ai_rover_remastered.xacro > rover.urdf
$ check_urdf  rover.urdf
```

Assuming everything is correct, we should have the following `check_urdf` output:

```
Robot name is: ai_rover_remastered
-----------Successfully Parsed XML ------------------
Root Link: base_footprint has 1 child(ren)
child(1): base_link
child(1): caster_wheel
child(2): left_wheel
child(3): right_wheel
```

If there are any errors, go back and review the scripts for typos.

## RViz Launch File

Recall that RViz is used to simulate the rover in isolation; i.e., we see the world “through the eyes” of the rover. To prepare the project for RViz, we convert the `ai_rover_remastered` directory into an ROS package just like we did in Chapter 4.

```bash
$ cd ~/catkin_ws/src/
$ catkin_create_pkg ai_rover_remastered
```

The `catkin_create_pkg` command creates two files in the `ai_rover_remastered` directory: `CMake_lists.txt` and `package.xml`. Finally, create the `ai_rover_remastered_rviz.launch` file in the launch directory:

This launch file is nearly identical to the RViz launch file in Chapter 4, the only difference being the script files with the Xacro extension. Similar to Chapter 4, Noetic ROS launches three nodes: `robot_state_publisher`, `joint_state_publisher,` and `rviz.` The first two guarantee the correct transformation between the link and joint frames published by ROS. The final ROS node launches the RViz program. Just as we did in Chapter 4, compile the code using the following:

```bash
$ cd catkin_ws/
$ catkin_make
$ source devel/setup.sh
$ roslaunch ai_rover_remastered ai_rover_remastered_RViz.launch
```


```markdown
The `roslaunch` command launches the RViz environment and GUI. To eliminate the errors found within RViz, we will need to alter the *Global Options* ➤ fixed_frame ➤ *map* to *Global Options* ➤ fixed_frame ➤ *base_link*. Therefore, just like in Chapter 4, we will have to add the RobotModel within the Displays tab (Figure 5-7).

![](img/494112_1_En_5_Fig7_HTML.jpg)

A screenshot of the dialogue box in the Rviz window with the heading create visualization. Below the heading, under the by display type tab, RobotModel is selected. Below it, there are boxes to write description and display name. Ok button is selected below the display name box.

Figure 5-7 Adding the Xacro RobotModel within Rviz

## Gazebo Launch File

Recall that Gazebo is used to test the rover in a robust environment. That means we see the rover and background from a third-person perspective. The rover, once verified to be correct in RViz, needs to be imported to Gazebo, and Gazebo needs two changes:

1.  Convert the Xacro files to URDF files. Gazebo "gets confused" reading valid Xacro files, but converting them to a single URDF removes the most warnings.

    1.  Copy the `ai_rover_gazebo.launch` file from Chapter 4 and call it `ai_rover_remastered_gazebo.launch`. Change any occurrence of `<ai_rover>` to `<ai_rover_remastered>`.

```bash
$ rosrun xacro xacro ai_rover_remastered.xacro > ai_rover_remastered.urdf
```

The `MYROBOT.world` file is the same one we used in Chapter 4.

The rover model spawned in Gazebo uses the `spawn_model` node of the `gazebo_ros` package. The rover model passes as an argument to the Gazebo instance. The following is the `roslaunch` terminal command that launches the rover model in Gazebo:

```bash
$ roslaunch ai_rover_remastered ai_rover_remastered_gazebo.launch
```

**Note** Before launching a Gazebo simulation, go to the home directory and kill all Gazebo and Gzserver processes:

```bash
$ killall gazebo
$ killall gzserver
```

These commands guarantee a clean slate for Gazebo simulations. If you get the error `[Err] [REST.cc:205] Error in REST request`, refer to [`https://automaticaddison.com/how-to-launch-gazebo-in-ubuntu/`](https://automaticaddison.com/how-to-launch-gazebo-in-ubuntu/).

![](img/494112_1_En_5_Fig8_HTML.jpg)

Two screenshots of the Rviz and Gazebo window. The left window has a horizontal square along with two circular wheels placed at the center. The right window has a 3 D image of the same object as in the left window.

Figure 5-8 Remastered rover displayed in Rviz and Gazebo

Once we execute the RViz and Gazebo launch commands, we will have the following RViz and Gazebo displays (Figure 5-8).

## Troubleshooting Xacro and Gazebo

This section reviews issues you might encounter while developing the Xacro model and converting it to URDF for Gazebo.

*   Gazebo has two visual components: the environment and the rover. The potential "physical" interaction of the rover and the environment could cause your program (or operating system) to crash. Verify that the spawning of the rover does not overlap any objects in the environment.

*   Run the simulation once, save the configuration using the *Global* ➤ *Options* menu, and then exit. Now the saved configuration will automatically be loaded every time you run Gazebo.

*   Make sure to terminate ALL Gazebo and Gzserver processes. A fresh start of these processes will ensure fewer problems.

## Teleop Node for Rover Control

*Teleop* means to control (**op**erate) from a distance (tele-). One of the ROS packages that functions to remotely control the rover is `teleop_twist_keyboard`. This function uses the keys `i/j/l/` to move the rover `up/left/right/down,` respectively. The `teleop_twist_keyboard` intercepts the keyboard commands and passes (publishes) the information via the `cmd_vel` topic to all subscribers. We bind the DDC plug-in to the `cmd_vel` topic by subscribing.

We will use two terminals to install the Teleops package. In the first terminal:

```bash
$ sudo apt-get install ros-noetic-teleop-twist-keyboard
$ roscore
```

In the second terminal:

```bash
$rosrun teleop_twist_keyboard teleop_twist_keyboard.py
```

Optionally, in a third terminal, you can view the published messages for all subscribers:

```bash
rostopic echo /cmd_vel
```

As usual, we prepare the groundwork for our Teleops control package by creating the Ubuntu directory structure and placing ourselves in the correct directory:

```bash
$ cd catkin_ws/src
$ catkin_create_pkg ai_rover_simple_control
$ cd ai_rover_simple_control
$ mkdir -p launch src
$ cd ~/catkin_ws
$ catkin_make
$ source devel/setup.sh
$ cd catkin_ws/src/ai_rover_simple_control/launch
```

Next, create the `ai_rover_teleops.launch` file that publishes the `cmd_vel` topic:

Now, make this launch file executable:

```bash
$ chmod +rwx ai_rover_teleops.launch
```

## Transform (TF) Graph Visualization

In the context of this project, a transform is a single-movement step in time. Each movement generates messages to each component to reflect the location change. Messages passed between nodes are difficult to visualize without a tool. We use the TF Graph tool to visualize (and test) the connections between the `teleops_twist_keyboard` package and the DDC plug-in. Using TF Graph, we see the real-time published messages, and we can use this information to guide our debugging.

First, we must get our rover running in Gazebo (Figure 5-9). Create three side-by-side terminals in the Terminator program. Launch the Gazebo program (not shown) in the left terminal (orange box):

```bash
$ roslaunch ai_rover_remastered ai_rover_remastered_gazebo.launch
```

In the middle terminal, launch the `teleops` launch file (blue box):

```bash
$ roslaunch ai_rover_remastered_simple_control ai_rover_remastered_teleops.launch
```

Be careful using the keyboard; any key press could now accidentally be interpreted as a `teleop` command! Click your mouse in the right terminal (green box) to make it active. In the right terminal run the following:

```bash
$ cd ~/catkin_ws/
$ ~/catkin_ws/ source devel/setup.sh
$ ~/catkin_ws/ rosrun tf2_tools view_frames.py
```


## After completing the previous set of commands

After completing the previous set of commands, your screen should look like Figure 5-9.

![](img/494112_1_En_5_Fig9_HTML.jpg)

A screenshot of a Gazebo window. Under the ROS program, there are three terminals labeled a, b, and c from left to right.

**Figure 5-9** Multiple terminals are running the `tf2_tools` ROS analysis program

Once we have determined that the ROS program is working, we can control the rover directly in Gazebo (Figure 5-10).

![](img/494112_1_En_5_Fig10_HTML.jpg)

A screenshot of multiple terminal windows and Gazebo. From left to right, two terminals are labeled a and b and c is the Gazebo. It highlights the real time factor labeled d at the bottom.

**Figure 5-10** Multiple terminal windows and Gazebo running

Please note that once these terminals are running, we need to be certain that the terminal responsible for executing the `teleops.py` program (blue box) is active by clicking on it with our pointer. To review: The launch files are the orange box, the active terminal found in blue box performs the `teleops.py` program, and the `TF View_Frames` is the green box.

The `rosrun tf view_frames` command produces a pop-up window with the TF Graph (Figure 5-11) showing the interrelation between the rover components. Each component has four properties: broadcaster, average rate, recent transform, and buffer length. The broadcaster is the package that published the data, and the average rate is the frequency of updates. The recent transform is the internal clock timestamp for the last update. And, finally, the buffer length is the amount of time that transpired to complete the previous update. Of these four properties, we will only be using broadcaster.

![](img/494112_1_En_5_Fig11_HTML.png)

A flowchart of the Rover U R D F and Gazebo with the heading view underscore frames result, recorded at time. It starts with Odom, base underscore footprint, base underscore link which is divided into I M U underscore link, camera underscore link, sensor underscore laser, caster underscore wheel, left underscore wheel, and right underscore wheel.

**Figure 5-11** Frames outline for both rover URDF and Gazebo

## Troubleshooting RViz Window Errors

In Figure 5-12, if you are missing the `RobotModel` field or `TF` field, you need to add it. Luckily, both solutions are very similar. Go to the Add button and select `RobotModel` or `TF`. Expand the `TF` visualization to see coordinate frames; these are our components defined in our Xacro files. Select the checkbox for the `RobotModel` and `TF` displays. These should now be shown in Gazebo and be reflected in the RViz GUI panel.

![](img/494112_1_En_5_Fig12_HTML.jpg)

A screenshot of the Rviz display panel. Under the displays tab, the options for global options, global status, grid, RobotModel and T F are marked.

**Figure 5-12** The correct settings for the RViz Displays panel

To reflect these changes in the RViz Views panel, change orbit view in the Views panel to odom: Current View ➤ Target Frame ➤ odom. See Figure 5-13.

![](img/494112_1_En_5_Fig13_HTML.jpg)

A screenshot of the Rviz views panel. Below the heading, there are options for type. Below it, a drop down menu is open in which the target frame, Odom, is selected under the headings current view and orbit, respectively.

**Figure 5-13** Odometry setting for the Target Frame option

## Controlling the Rover

We can control the rover in Gazebo with keyboard commands. Click on the Terminator terminal to interact with the `teleops_twist_keyboard` program (Figure 5-9, blue box). Press the “i” key, and the rover will continuously move forward in both RViz and Gazebo until you tell it to stop. Some simple case-sensitive keyboard commands: “i” (move forward), “j” (turn left), “k” (stop), “l” (turn right). Other commands may be defined as we need them. Figure 5-14 shows a snapshot of the rover in the RViz and Gazebo environments.

![](img/494112_1_En_5_Fig14_HTML.jpg)

A schematic of a snapshot of the Rover in the Rviz and Gazebo environment. The left has some vertical and horizontal lines labeled base underscore footprint and spherical object labeled Odom. The right has a 3 D horizontal square along with two circular wheels and a sphere below it.

**Figure 5-14** Rover in Rviz (left) and Gazebo (right)

Remember to save your work: File ➤ “Save Config As.” Save the configuration file in the following directory:

```
ai_rover_remastered_simple_control/config/rviz_odom.rviz
```

Reload the program with the new configuration file.

## Drifting Issues with the Rover

As you turned and moved the rover in the world, there may have been “drifting” problems. This drifting means the rover does not travel in a straight line. Usually, drifting occurs after the rover turns. If this happens, alter the mass of the rover by modifying the corresponding inertia values in the Xacro URDF file. Experiment with the weights of the different components until you are satisfied. These changes should control the drifting issues with the rover to a greater extent. Eventually, we will use deep learning to prevent drifting errors. For additional information, refer to [`https://www.youtube.com/watch?v=1bnEdQzf8Yw`](https://www.youtube.com/watch%253Fv%253D1bnEdQzf8Yw).

## Our First Python Controller

The DDC plug-in subscribes to the `cmd_vel` topic to receive velocity commands to control the rover. The DDC does not know where the data originates. In Chapter 4, keyboard commands (i/j/k/l) published messages to the `cmd_vel` topic. But we can send commands from a program! The `Twist` library function publishes command messages to the `cmd_vel` topic, too. But because it accepts parameters, command control and accuracy are better than the keyboard version.

The `Twist` function has two velocity attributes: linear (forward/backward) and angular (turning). The parameters set these attributes. For instance, in the following Python script, set the `msg.linear.x=0.1` initial value for `Twist()`. This parameter translates to “move the Rover forward 0.1m every second” and is equivalent to pressing the “i" key once on the keyboard.

```
ai_rover_remastered_simple_control/src/ai_rover_simple_twist_pub.py
```

Create the `ai_rover_remastered_simple_twist_pub.py` script and make it executable:

```python
#!/usr/bin/env python3

import rospy
import sys
from geometry_msgs.msg import Twist

def publish_velocity_commands():
    # Velocity publisher
    vel_pub = rospy.Publisher('/ai_rover_remastered/base_controller/cmd_vel', Twist, queue_size=10)
    rospy.init_node('ai_rover_simple_twist_pub', anonymous=True)
    msg = Twist()
    msg.linear.x = 0.1
    msg.linear.y = 0
    msg.linear.z = 0
    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = 0
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        vel_pub.publish(msg)
        rate.sleep()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        try:
            publish_velocity_commands()
```

Load the configuration file from the first simple control keyboard program before executing the Python control program. If you do not, then you may encounter difficulties. Using the Terminator program, launch the RViz and Gazebo simulators:

```bash
$ roslaunch ai_rover_remastered ai_rover_remastered_gazebo.launch
$ chmod +rwx src/ai_rover_remastered_simple_control/src/ai_rover_remastered_simple_twist_pub.py
$ rosrun ai_rover_remastered_simple_control ai_rover_remastered_simple_twist_pub.py
```


```py
The `ai_rover_remastered_simple_twist_pub.py` controller script publishes a `Twist` message directly to the `cmd_vel` topic to which the DDC subscribes. Experiment with the movement for the `Twist` function by changing any of the `msg.linear` or `msg.angular` fields. You should be able to observe the rover moving around in the RViz and Gazebo environments. We now have ALL the fundamental building blocks needed to construct an autonomous rover.

## Building Our Environment

After we have created the controller script for the rover, it moves without human input. We have now developed a primitive autonomous rover, and now we have to put it somewhere to explore.

We will simulate our environment as a simple maze. Go to the Gazebo GUI, select Edit (Open Building Editor), and create a simple maze with a few walls and at least one door. For more information, refer to [`http://gazebosim.org/tutorials?cat=build_world&tut=building_editor`](http://gazebosim.org/tutorials%253Fcat%253Dbuild_world%2526tut%253Dbuilding_editor). Our generated maze is shown in Figure 5-15; yours can be different. We will go over more details regarding maze generation in Chapter 6.

![A schematic of the maze in the Gazebo window. There is a circle at the bottom left and at the top the dimension vector icon is selected.](img/494112_1_En_5_Fig15_HTML.jpg)

**Figure 5-15** The rover (gray circle) is now exploring a maze

Save the maze as `ai_rover_remastered/worlds/catacomb.world` (and load it into your Gazebo environment).

## Summary

In this chapter, we have used Xacro to simplify the rover development by using modular design. This technique gave us the ability to tack on sensors such as the LiDAR and camera. We tested the rover in both the RViz and Gazebo environments using keyboard commands. Then we showed we could control the rover without the keyboard using the Teleops ROS node program. Finally, we created a simple maze to explore with our rover. In the next chapter, we will develop a rover with an embedded controller. We also reviewed and made the first use of SLAM libraries provided with ROS. However, future chapters will allow us to refine our skills with SLAM.

## Exercises

- **Exercise 5.1:** How would you save the constructed maps?
- **Exercise 5.2:** How would you integrate other sensors, such as depth cameras, into the rover?
- **Exercise 5.3:** Why did we create a digital twin before building a real rover? (Hint: Think expense and complexity.)
- **Exercise 5.4:** Let us assume we want to use another range sensor, such as a depth camera (RGB-D camera). How can you create a plug-in to accommodate this new sensor?
```