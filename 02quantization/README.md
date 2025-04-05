# 第4章 模型量化

![](images/Quantization.jpeg)

&emsp;&emsp;本章我们将在 4.1 节学习什么是量化？为什么量化？4.2 节学习不同数据类型如何表示，以及 4.3 量化的基本方法，不同数据类型之间是如何进行量化转换的。当我们学习了量化的基本原理之后，将在 4.4、4.5、4.6、4.7、4.8 节进一步学习不同的量化策略来优化量化效果。最后我们通过几个量化实践来巩固学到的知识。

- [第4章 模型量化](#第4章-模型量化)
  - [4.1 什么是量化？为什么量化？](#41-什么是量化为什么量化)
  - [4.2 数据类型（Data Type）](#42-数据类型data-type)
      - [4.2.1 整型（Integer）](#421-整型integer)
    - [4.2.2 定点数（Fixed Point Number）](#422-定点数fixed-point-number)
    - [4.2.3 浮点数（Floating Point Number）](#423-浮点数floating-point-number)
  - [4.3 量化基本方法](#43-量化基本方法)
    - [4.3.1 k-means 量化](#431-k-means-量化)
    - [4.3.2 线性量化](#432-线性量化)
      - [4.3.2.1 定义](#4321-定义)
      - [4.3.2.2 线性矩阵乘量化](#4322-线性矩阵乘量化)
      - [4.3.2.3 全连接层线性量化](#4323-全连接层线性量化)
      - [4.3.2.4 卷积层线性量化](#4324-卷积层线性量化)
  - [4.4 训练后量化 （Post-Training Quantization）](#44-训练后量化-post-training-quantization)
    - [4.4.1 量化粒度](#441-量化粒度)
    - [4.4.2 动态量化参数的计算 ( Cliping )](#442-动态量化参数的计算--cliping-)
        - [4.4.2.1 指数移动平均（EMA）](#4421-指数移动平均ema)
        - [4.4.2.2 Min-Max](#4422-min-max)
        - [4.4.2.3 KL 量化](#4423-kl-量化)
        - [4.4.2.4 均方误差（MSE）](#4424-均方误差mse)
    - [4.4.3 Rounding](#443-rounding)
  - [4.5 量化感知训练（Quantization-Aware Training）](#45-量化感知训练quantization-aware-training)
    - [4.5.1 前向传播](#451-前向传播)
    - [4.5.2 反向传播](#452-反向传播)
  - [4.6 混合精度量化](#46-混合精度量化)
  - [4.7 其他量化方法](#47-其他量化方法)
    - [4.7.1 INT4 和 FP4](#471-int4-和-fp4)
    - [4.7.2 二值和三值量化](#472-二值和三值量化)
        - [4.7.2.1 二值量化 (Binarization)](#4721-二值量化-binarization)
        - [4.7.2.2 三值量化 (Ternary Quantization)](#4722-三值量化-ternary-quantization)
  - [4.8 模型量化对象](#48-模型量化对象)
  - [4.9 实践](#49-实践)
  - [引用资料](#引用资料)


## 4.1 什么是量化？为什么量化？

&emsp;&emsp;上一章节我们学习了剪枝 (pruning)，剪枝操作能够减少网络的参数量，从而起到压缩模型的作用。而量化 (quantization) 是另一种能够压缩模型参数的方法。量化将神经网络的浮点算法转换为定点，修改网络中每个参数占用的比特数，从而减少模型参数占用的空间。

&emsp;&emsp;移动端的硬件资源有限，比如内存和算力。而量化可以减少模型的大小，从而减少内存和算力的占用。同时，量化可以提高模型的推理速度。下图为不同数据类型的加法和乘法操作的耗时对比。

![图4-1 不同数据类型的加法和乘法操作的耗时对比](images/different_datatypes_operations_speeds.jpg)

&emsp;&emsp;模型量化有以下几个好处：
- 减小模型大小：如 int8 量化可减少 75% 的模型大小，int8 量化模型大小一般为 32 位浮点模型大小的 1/4。
- 减少存储空间：在端侧存储空间不足时更具备意义。
- 减少内存占用：更小的模型当然就意味着不需要更多的内存空间。
- 减少设备功耗：内存耗用少了推理速度快了自然减少了设备功耗。
- 加快推理速度：浮点型可以访问四次 int8 整型，整型运算比浮点型运算更快；CPU 用 int8 计算的速度更快。
- 某些硬件加速器如 DSP/NPU 只支持 int8。比如有些微处理器属于 8 位的，低功耗运行浮点运算速度慢，需要进行 8bit 量化。
  
## 4.2 数据类型（Data Type）

&emsp;&emsp;模型量化过程可以分为两部分：将模型从 fp32 转换为 int8 ;使用 int8 进行推理。整个量化过程都和数据类型的转换息息相关。我们先在这个小节了解数据类型，如果不了解基础的数据类型，在后续的量化细节内容中通常会感到困惑。
#### 4.2.1 整型（Integer）
&emsp;&emsp;如下图所示，整型数据可以分为无符号整型（Unsigned Integer）和有符号整型（Signed Integer）。
- 无符号整型：数据范围为 0 到 $2^{n-1}$，n 为数据位数。
- 有符号整型：
  - 原码表示(Sign-Magnitude Representation)：其实现的原理是取二进制数的最高位（左起第一位）为符号位，约定符号位为0时表示正数，符号位为1时表示负数，其余二进制位则用于待表示数值的绝对值。数据范围为 $-2^{n-1}-1$ 到 $2^{n-1}-1$，n 为数据位数。
  - 补码表示（Two's Complement Representation）：为了弥补原码表示，有 +0 和 -0 两种表示的缺点，最高位除了具有符号表示的功能，也具有权重值。数据范围为 $-2^{n-1}$ 到 $2^{n-1}-1$，n 为数据位数。

![图4-2 整型](images/integer.png)

### 4.2.2 定点数（Fixed Point Number）

&emsp;&emsp;定点数的关键地方就在“定”和“点”这两个字上面，即在表示小数数据时，把小数点的位置已经约定好固定在某个位置。与之对应的是浮点数，其小数点的位置不是固定的。如下图所示，指定蓝色部分为符号位，绿色部分为整数位，橙色部分为小数位。

![图4-3 定点数](images/fixed-point.png)


### 4.2.3 浮点数（Floating Point Number）

&emsp;&emsp;如下图所示，浮点数的每个位数表示的数值和定点数是一样的，但是计算方式不再是单纯的相加。其中fraction表示小数部分，exponent表示指数部分。fraction和exponent的位数分别决定了数据的精度和表示范围大小。例如指数的bias = 127 = $2^{7}-1$，所以指数部分表示的范围为 -127 到 128。

&emsp;&emsp;fp32浮点数的表示公式如下：


$$
fp32 = (-1)^{sign}·(1+fraction)·2^{exponent-127}
$$

![图4-4 IEEE 754浮点数表示](images/fp32_1.png)


&emsp;&emsp;我们考虑一下如何表示0呢？如下图所示，我们规定 exponent 为 0 时，强制 fraction 部分不再加 1，指数部分强制为 1-bias = -126 。当fraction = 0，表示 0 。这种表示方式叫非正规浮点数。公式如下：

$$
fp32 = (-1)^{sign}·(fraction)·2^{1-127}
$$

![图4-5 fp32示例](images/fp32_2.png)

&emsp;&emsp;这两种表示方式的区别在于，当 expontent 不等于 0 时，采用正常的浮点表示方式。当 expontent = 0 时，采用非正规表示方式。
- 正常浮点可表示的最小正值为 fraction = 0，exponent = 1，结果为$2^{-126}$ 。
$$
(1+0)·2^{1-127} = 2^{-126}
$$
- 非正规浮点可表示的最小正值为 fraction = $2^{-23}$，结果为$2^{-149}$ 。
$$
2^{-23}·2^{1-127} = 2^{-149}
$$
- 非正规浮点可表示的最大值为 fraction 部分全为1 ，结果为$2^{-126}-2^{-149}$。
- 正常浮点数可表示的最大值为 fraction = $2^{-23}$，结果为$(1+1-2^{-23})·2^{127}$  。
- 正常浮点数的小数部分全为0，指数部分全为1时分别表示正无穷和负无穷。

&emsp;&emsp;我们可以扩展到其他数据类型：
- 当expontent位数为5，fraction位数为11时，为fp16。
  $$
  fp16 = (-1)^{sign}·(1+fraction)·2^{exponent-15}
  $$
- 当expontent位数为8，fraction位数为7时，为bf16。
$$
bf16 = (-1)^{sign}·(1+fraction)·2^{exponent-127}
$$
- 当expontent位数为4，fraction位数为3时，为fp8(E4M3)。
  $$
  fp8 = (-1)^{sign}·(1+fraction)·2^{exponent-7}
  $$
- 当expontent位数为5，fraction位数为2时，为fp8(E5M2)。
  $$
  fp8 = (-1)^{sign}·(1+fraction)·2^{exponent-15}
  $$

> 与 fp16 相比，bf16 扩大了可表示范围，从而降低了下溢和上溢风险。尽管由于有效位数较少而导致精度降低，但 bf16 通常不会显着影响模型性能。

&emsp;&emsp;我们需要注意的是，expontent 决定了表示范围，fraction决定了精度范围。fp8的两种表示方式fp8(E4M3)精度更高，fp8(E5M2)表示范围更大。

![图4-6 常见浮点数汇总](images/fp.png)

> **定点数和浮点数的比较:** 在计算机上用相同位数表示数据时，浮点数可以表示的数据范围比定点数大得多。在计算机上用相同位数表示数据时，浮点数的相对精度高于定点数。浮点数在计算时需要同时计算指数和尾数，并且需要对结果进行归一化。因此，浮点运算比定点运算涉及更多步骤，导致计算速度较慢。



## 4.3 量化基本方法

&emsp;&emsp;根据存储和计算时使用的数据类型不同，主要介绍以下两种量化方法：
- 基于 k-means 的量化（K-means-based Quantization）：存储方式为整型权重 + 浮点型的转换表），计算方式为浮点计算。
- 线性量化（Linear Quantization）：存储方式为整型权重存储，计算方式为整型计算。
 
![图4-7 量化方法](images/quantization.png)

### 4.3.1 k-means 量化

&emsp;&emsp;如下图所示，k-means 量化将weights聚类。每个权重的位置只需要存储聚类的索引值。将权重聚类成4类(0,1,2,3)，就可以实现2-bit的压缩。
存储占用从 `32bit x 16 = 512 bit = 64 B => 2bit x 16 + 32 bit x 4 = 32 bit + 128 bit = 160 bit = 20 B`

&emsp;&emsp;当weight更大时，压缩比例将会更大。

![图4-8 k-means量化流程](images/k-means.png)

- 推理时，我们读取转换表，根据索引值获取对应的值。
- 训练时，我们将gradient按照weights的聚类方式进行聚类相加，反向传播到转换表，更新转换表的值。

&emsp;&emsp;以下是将上一节的剪枝和k-means 量化结合起来的压缩流程。首先，剪枝将模型中的参数进行剪枝，循环进行微调和剪枝，得到最优的剪枝模型。然后，k-means 量化将剪枝后的参数进行聚类，将聚类的索引值存储在模型中，并构建相应的索引表，并使用哈夫曼编码进一步压缩。

![图4-9 压缩范式](images/compression.png)

### 4.3.2 线性量化
#### 4.3.2.1 定义

&emsp;&emsp;线性量化是将原始浮点数据和量化后的定点数据之间建立一个简单的线性变换关系，因为卷积、全连接等网络层本身只是简单的线性计算，因此线性量化中可以直接用量化后的数据进行直接计算。

&emsp;&emsp;我们用 $r$ 表示浮点实数，$q$ 表示量化后的定点整数。浮点和整型之间的换算公式为：

$$
r = S(q - Z)
$$ 

$$
q = round(r / S + Z)
$$ 
其中，$S$ 是量化放缩的尺度，表示实数和整数之间的比例关系，$Z$ 是偏移量，表示浮点数中的 0 经过量化后对应的数（量化偏移），根据偏移量$Z$是否为0，可以将浮点数的线性量化分为对称量化（$Z$=0）和非对称量化（$Z$≠0）。大多数情况下量化是选用无符号整数，比如INT8的值域为[0,255],这种情况下需要要用非对称量化。$S$和$Z$的计算方法为：
$$
S =  \frac{r_{max} - r_{min}}{q_{max} - q_{min}}
$$ 
$$
Z = round(q_{max}-\frac{r_{max}}{S})
$$ 

&emsp;&emsp;其中，$r_{min}$ 和 $r_{max}$分别表示浮点数中的最小值和最大值,$q_{min}$ 和 $q_{max}$分别表示定点数中的最小值和最大值。

![图4-10 线性量化](images/linear.png)

下面举一个例子来详细说明：

如下图所示，给定一个矩阵，可以通过上面的公式计算出Z和S。

![图4-11 计算S和Z](images/linear_1.png)

可进一步利用上面的公式计算出量化后的矩阵。

![图4-12 计算量化后的矩阵](images/linear_2.png)

> 上述方法又经常被称为零点量化(Zero-Point Quantization)。

&emsp;&emsp;此外，还有一种比较常见的线性量化方法:绝对最大（ absmax ）量化。

![图4-13 AbsMax量化](images/absmax.png)

&emsp;&emsp;我们用 $X$ 表示原始张量，$X_{quant}$ 表示量化后的张量。两者之间的换算公式为：


$$
X_{quant}=round(S \cdot X)
$$ 



其中，$S$ 是量化放缩的尺度，表示实数和整数之间的比例关系。$S$的计算方式为：
$$
S = \frac{2^{n-1}-1}{\max |X|}
$$ 
其中，$n$表示要量化的字节数，$\max |\cdot|$表示张量中的绝对值最大值。

&emsp;&emsp;反量化$X_{dequant} $公式表示为：
$$
X_{dequant} = round(X_{quant} / S )
$$ 

#### 4.3.2.2 线性矩阵乘量化

&emsp;&emsp;线性矩阵乘量化是将线性量化应用于矩阵乘法。

&emsp;&emsp;矩阵乘法可以用下式表示：

$$
Y = WX
$$

&emsp;&emsp;假设 $S_Y$ 和 $Z_Y$ 是矩阵输出 Y 对应的 scale 和 zero point，$S_W$、$Z_W$、$S_X$、$Z_X$ 同理，那么由上式可以推出：

$$
S_Y(q_Y-Z_Y) = S_W(q_W-Z_W)·S_X(q_S-Z_X)
$$

&emsp;&emsp;整理一下可以得到：
$$
q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X-Z_Wq_X-Z_Xq_W+Z_WZ_X) + Z_Y
$$

&emsp;&emsp;其中，$\frac{S_WS_X}{S_Y}$表示为$2^{-n}M_0$ 转化成为定点计算，便可以通过移位得到近似的缩放因子。所谓定点，指的是小数点的位置是固定的，即小数位数是固定的。$Z_Xq_W$ 和 $Z_WZ_X$ 以及 $Z_Y$ 我们可以提前计算出来。

&emsp;&emsp;我们能否让 $Z_W$ 的值为 0 呢？当 $Z_W$ 的值为 0 时，我们只需要将 $q_W$ 和 $q_X$ 存储起来，然后根据公式计算出 $q_Y$，最后将 $q_Y$ 存储起来。

&emsp;&emsp;当$Z_W$的值为0时，对应的量化方式为线性对称量化(symmetric quantization)。

![图4-14 对称线性量化](images/symmetric_linear.png)

&emsp;&emsp;计算公式变换如下：
$$
S =\frac{｜r｜_{max}}{2^{N-1}}
$$ 
$$
Z = 0
$$ 

&emsp;&emsp;其中，$N$ 表示定点数的小数位数。

&emsp;&emsp;矩阵乘法的公式变为如下：
$$
q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X-Z_Xq_W) + Z_Y
$$


#### 4.3.2.3 全连接层线性量化

&emsp;&emsp;全连接层线性量化与矩阵乘法相比多了一个bias，因此需要对bias进行线性量化。

&emsp;&emsp;对称量化的全连接层的线性量化公式为：
$$
{S_Y}(q_Y-Z_Y) = {S_WS_X}(q_Wq_X-Z_Xq_W) + S_b(q_b-Z_b)
$$

&emsp;&emsp;其中，$S_b$ 表示bias的缩放因子。

&emsp;&emsp;我们强制$Z_b=0$ , $S_b=S_WS_X$, 则全连接层的线性量化公式变为：
$$
{S_Y}(q_Y-Z_Y) = {S_WS_X}(q_Wq_X-Z_Xq_W+q_b)
$$ 
$$
q_Y = \frac{S_WS_X}{S_Y}(q_Wq_X-Z_Xq_W+q_b) + Z_Y
$$

&emsp;&emsp;其中，$-Z_Xq_W+q_b$ 可以提前计算出来。

#### 4.3.2.4 卷积层线性量化


&emsp;&emsp;卷积层线性量化与全连接层线性量化相比多了一个卷积核，因此可以推导出卷积的线性量化公式：
$$
q_Y = \frac{S_WS_X}{S_Y}(Conv(q_W,q_X)-Conv(Z_X,q_W)+q_b) + Z_Y
$$ 

&emsp;&emsp;下图所示为模型量化后的推理过程，量化的activations和量化的weight进行卷积，然后加上bias。与scale_factor相乘，再加上ouput的zero_point，就可得到最后的量化结果。

![图4-15 卷积量化](images/conv_quantization.png)

## 4.4 训练后量化 （Post-Training Quantization）

&emsp;&emsp;训练后量化（Post-Training Quantization, PTQ）是指在训练完成后，对模型进行量化，因此也叫做离线量化。根据量化零点是否为 0，训练后量化分为对称量化和非对称量化，这部分内容已在上述章节进行介绍；根据量化粒度区分，训练后量化又分为逐张量量化和逐通道量化以及组量化。

&emsp;&emsp;量化会带来精度损失，那么如何选取量化时所用参数（如scaling factor，zero point）可以尽可能地减少对准确率的影响呢？这也是我们需要关注的地方。量化误差来自两方面，一个是clip操作，一个是round操作。因此，我们还要介绍动态量化参数的计算方式，以及 round 这个操作带来的影响。

### 4.4.1 量化粒度

&emsp;&emsp;量化通常会导致模型精度下降。这就是量化粒度发挥作用的地方。选择正确的粒度有助于最大化量化，而不会大幅降低准确性性能。

&emsp;&emsp;逐张量量化（Per-Tensor Quantization）是指对每一层进行量化。在逐张量量化中，相同的量化参数应用于张量内的所有元素。在张量之间应用相同的参数会导致精度下降，因为张量内参数值的范围可能会有所不同。如下图的红框所示，3个channel共享一个量化参数。但是我们可以看到不同channel的数据范围是不同的。因此当 Layer-wise 量化效果不好时，需要对每个channel进行量化。

![图4-16 逐张量量化](images/per-tensor.png)

&emsp;&emsp;逐通道量化（Channel-wise Quantization就是将数据按照通道维度进行拆分，分别对每一通道的数据进行量化。相较于逐张量量化，逐通道量化可以减少量化误差，但需要更多的存储空间。逐通道量化可以更准确地捕获不同通道中的变化。这通常有助于 CNN 模型，因为不同通道的权重范围不同。由于现阶段模型越来越大，每个通道的参数也原来越多，参数的数值范围也越来越大，因此我们需要更细粒度的量化方式。

&emsp;&emsp;逐张量量化与逐通道量化的对比结果如下图所示。从图中可以看出：使用逐通道量化的误差更小，但付出的代价是必须存储更多信息(多个r和S) 。

![图4-17 逐张量量化与逐通道量化对比](images/channel_tensor.png)

&emsp;&emsp;组量化（Group Quantization）是指对通道内的数据拆分成多组向量，每组向量共享一个量化参数。VS-Quant 对张量的单个维度内的每个元素向量应用比例因子。它将通道维度细分为一组向量。

$$
r=S(q-Z) \rightarrow r=\gamma \cdot S_q(q-Z)
$$
其中，$\gamma$是浮点数的粗粒度缩放因子,$S_q$是每个向量的整数缩放因子。这种方法通过结合不同粒度的缩放因子，实现了精度和硬件效率的平衡：
- 较小粒度时，使用较简单的整数缩放因子；
- 较大粒度时，使用较复杂的浮点缩放因子。

&emsp;&emsp;存储开销：对于两级缩放因子，假设使用4-bit的量化，每16个元素有一个4-bit的向量缩放因子，那么有效位宽为 `4+4/16=4.25`bits。

![图4-18 VS-Quant](images/vs-quant.png)

&emsp;&emsp;为了提高能源效率，引入了两级缩放方案MX (Microscaling)。微缩放 (MX) 规范是从著名的 Microsoft 浮点 (MSFP) 数据类型升级而来的。该算法首先以每个向量的粒度计算浮点比例因子。然后，它通过将每向量比例因子分成整数逐向量分量和浮点逐通道分量来量化它们。MX 系列（如 MX4、MX6、MX9）表示了不同量化方案，它们的主要区别在于数据类型、缩放因子的设计以及组大小，目的在于通过压缩模型权重数据，优化神经网络的性能。

&emsp;&emsp;下图是不同的多级缩放方案对比结果。`有效位宽 = (L0 数据位宽 + L0 量化尺度位宽 / L0 组大小 + L1 量化尺度位宽 / L1 组大小)`。L0缩放因子通常采用较低的精度，用定点数表示，L1缩放因子则采用浮点数的表示方式。以MX6为例：L0数据类型是S1M4，表示1位符号位+4位尾数，共5位，用于表示数值。L0 量化尺度数据类型为E1M0，表示1位指数位，因此占1位。L0组大小是 2，这意味着L0量化尺度是针对每2个元素进行分组。L1 量化尺度数据类型为E8M0，表示8位指数位，因此占8 位。L1 组大小是 16，意味着 L1 量化尺度是针对每 16 个元素进行分组。所以可得到：`有效位宽 = 5 + 1/2 + 8/16 = 6` bits。


![图4-19 不同的多级缩放方案](images/scale.png)

### 4.4.2 动态量化参数的计算 ( Cliping )
##### 4.4.2.1 指数移动平均（EMA）

&emsp;&emsp;指数移动平均（Exponential Moving Average, EMA）是一种常用的统计方法，用于计算数据的指数移动平均值。

&emsp;&emsp;EMA 收集了训练过程中激活函数的取值范围 $r_{min}$ 和 $r_{max}$，然后在每个 epoch 对这些取值范围进行平滑处理。

&emsp;&emsp;EMA的计算公式如下：
$$
r^{t+1}_{min,max} = \alpha r^{t}_{min,max} + (1-\alpha) r^{t+1}_{min,max}
$$

其中，$r^{t}_{min,max}$ 表示第 $t$ 步的取值范围，$\alpha$ 表示平滑系数。


##### 4.4.2.2 Min-Max 
&emsp;&emsp;Min-Max 是一种常用的校准方法，通过在训练好的 fp32 模型上跑少量的校准数据。统计校准数据的 $r_{min,max}$ 并取平均值作为量化参数。

##### 4.4.2.3 KL 量化

&emsp;&emsp;KL 量化是用 KL 散度来衡量数据和量化后的数据之间的相似性；这种方法不是直接将$ [min, max] $v映射到 $[-127,128]$，而是去寻找一个阈值 $|T| < max(|max|, |min|)$ ，将 $ [-T, T]$ 映射到 $[-127, 128]$ 。并假设只要阈值选取得当，使得两个数据之间的分布相似，就不会对精度损失造成影响。

$$
D_{KL}(P||Q) = \sum_{i=1}^nP(x_i)\log\frac{P(x_i)}{Q(x_i)}
$$

##### 4.4.2.4 均方误差（MSE）

&emsp;&emsp;均方误差量化是指通过最小化输入数据 $X$ 和量化后的数据 $Q(X)$ 之间的均方误差，计算得到最合适的量化参数。

$$
min_{|r|_{max}}E|(X-Q(X))^2|
$$ 

&emsp;&emsp;通过动态调整 $｜r｜_{max}$ 来最小化均方误差。

### 4.4.3 Rounding

&emsp;&emsp;Rounding 是指将浮点数进行舍入操作，将浮点数映射到整数。最常用的 Rounding 方法是最近整数（Rounding-to-nearest）。权重是互相关联的，对每个权重的最好舍入不一定是对整个张量的最好舍入。如下图所示，如果我们考虑整体的数据分布，将权重 0.5 舍入为 1 不是一个好的选择。

![图4-20 Rounding](images/rounding.png)

&emsp;&emsp;我们最终想要的量化效果是输出数据的损失尽可能小，因此我们可以通过评判 rounding 对输出的影响来决定权重的舍入方式，也就是 AdaRound。简化的计算公式如下所示：

$$
argmin||(Wx-\widehat{W}x)||
$$
其中，$\widehat{W} = \lfloor\lfloor{W}\rfloor+\sigma\rceil$ , $\sigma \in [0,1]$，表示当前值是向上取整还是向下取整。


## 4.5 量化感知训练（Quantization-Aware Training）


&emsp;&emsp;量化感知训练（Quantization-Aware Training, QAT）是指在训练过程中，对模型添加模拟量化算子，模拟量化模型在推理阶段的舍入和裁剪操作，引入量化误差。并通过反向传播更新模型参数，使得模型在量化后和量化前保持一致。

### 4.5.1 前向传播

&emsp;&emsp;如下图所示，量化训练的前向传播过程如下：

- $Layer_{N-1}$ 的输出 $Q(X)$ 作为输入传入到下一层 $Layer_{N}$，其中 $Q(X)$ 表示量化反量化后的数据；

- $Layer_{N}$ 的权重 $W$ 经过量化反量化之后 $Q(W)$ 成为新的权重与 $Q(X)$ 计算得到输出 $Y$ 。

- $Y$ 量化反量化之后得到Q(Y) 输入到下一层 $Layer_{N+1}$。

![图4-21 量化感知训练计算图](images/qat.png)

&emsp;&emsp;因为int8的表示范围远小于fp32，当fp32 量化成 int8 时，不同大小的数据会映射到int8的相同数值，再反量化回 fp32 时就会产生误差。量化反量化操作就是为了将量化误差引入到模型的训练中。

&emsp;&emsp;我们要注意的是，整个量化过程中算子的计算都是在高精度下完成的。

### 4.5.2 反向传播

&emsp;&emsp;量化感知训练的损失函数与普通训练的损失函数类似，但是量化后的权重是离散值。如图所示为 $W$ 和 $Q(W)$ 的关系。

![图4-22 STE](images/ste.png)

&emsp;&emsp;可以得到以下式子：
$$
\frac{\partial Q(\mathbf{W})}{\partial \mathbf{W}}=0
$$ 

&emsp;&emsp;求导公式可以做如下转换：

$$
g_{\mathbf{W}}=\frac{\partial L}{\partial \mathbf{W}}=\frac{\partial L}{\partial Q(\mathbf{W})} \cdot \frac{\partial Q(\mathbf{W})}{\partial \mathbf{W}}=0
$$

&emsp;&emsp;如果按照上述式子进行梯度计算，这样的话梯度就永远为 0，无法进行梯度更新。因此人们提出了一个修正的方式，被称为直通估计器(Straight-Through Estimator，STE)。将 $W$ 和 $Q(W)$ 的关系假设为上图中的红色虚线，$W = Q(W)$，$
\frac{\partial{Q(W)}}{\partial{W}}=1$ ，梯度公式可以转换为如下式子：

$$
g_{\mathbf{W}}=\frac{\partial L}{\partial \mathbf{W}}=\frac{\partial L}{\partial Q(\mathbf{W})}
$$

&emsp;&emsp;这样，我们就可以进行反向传播计算。

## 4.6 混合精度量化

&emsp;&emsp;混合精度量化指同时使用低精度和高精度数据类型来减少模型的大小和计算成本的一种方法。通过针对性地对不同 layer 选择不同的量化精度，可以有效地避免量化误差的传播和积累，从而保证模型的性能不受影响。

## 4.7 其他量化方法

### 4.7.1 INT4 和 FP4

&emsp;&emsp;INT4 和 FP4 是一种特殊的定点数和浮点数，目前的模型越来越大，所以我们需要更低bits的表示方法。

&emsp;&emsp;INT4 表示的范围为 -8 到 7；FP4 表示的范围根据不同的指数位和小数位而有所不同。具体表示范围如下图所示。

![图4-23 int4和fp4](images/int4_fp4.png)

### 4.7.2 二值和三值量化

##### 4.7.2.1 二值量化 (Binarization)
&emsp;&emsp;在二值量化中，模型的权重或激活值被限制为两个离散值，通常是 -1 和 1 。这样可以大幅减少模型的存储需求，因为每个参数只需要一位 bit 来表示。二值神经网络的计算也可以大幅加速，因为二值运算比浮点运算要简单得多。

&emsp;&emsp;二值化（Binarization）的具体实现有两种方法：**确定性二值化（Deterministic Binarization）** 和 **随机二值化（Stochastic Binarization）**。

1） **确定性二值化（Deterministic Binarization）**：
- 直接根据一个阈值（通常是0）计算位值，结果为符号函数：
  $$
  q = \text{sign}(r) = 
  \begin{cases} 
  +1, & r \geq 0 \\
  -1, & r < 0 
  \end{cases}
  $$
- 即，如果输入大于等于0，则输出1；否则输出-1。

2） **随机二值化（Stochastic Binarization）**：
 - 使用全局统计或输入数据的值来确定输出为 -1 或 +1 的概率。例如，在 Binary Connect (BC) 方法中，概率由 sigmoid 函数 $\sigma(r)$ 确定：
   $$
   q = 
   \begin{cases} 
   +1, & \text{with probability } p = \sigma(r) \\
   -1, & \text{with probability } 1 - p 
   \end{cases}
   $$
   其中，$\sigma(r) = \min(\max(\frac{r+1}{2}, 0), 1)$。
 - 这种方法的实现较为困难，因为量化时需要硬件生成随机比特。

![图4-24 二值量化示例](images/binarization.png)

&emsp;&emsp;上图展示了在二值化中最小化量化误差的方法。

&emsp;&emsp;为了更好地逼近原始权重，二值权重 $ W^\mathbb{B} $ 乘以一个缩放因子 $ \alpha $：
   $$
   \alpha = \frac{1}{n} \| W \|_1
   $$
&emsp;&emsp;计算后得到缩放后的二值权重 $ \alpha W^\mathbb{B} $，其中 n 为矩阵中元素的个数。

&emsp;&emsp;通过引入缩放因子，误差从9.28减少到9.24，显示出缩放对减小误差的效果。

##### 4.7.2.2 三值量化 (Ternary Quantization)

&emsp;&emsp;在三值量化中，模型的权重或激活值被限制为三个离散值，通常是 -1 、 0 和 1 。相比二值量化，三值量化允许模型拥有一个额外的零值，这可以压缩模型参数的同时保留模型的精度。

&emsp;&emsp;三值量化的具体规则如下：
$$
q = \begin{cases} 
r_t, & r > \Delta \\
0, & |r| \leq \Delta \\
-r_t, & r < -\Delta 
\end{cases}
$$
其中， $\Delta = 0.7 \times \mathbb{E}(|r|)$， $r_t = \mathbb{E}_{|r|>\Delta}(|r|)$

![图4-25 二值量化示例](images/ternary.png)

&emsp;&emsp;如上图所示为三值量化的具体示例，展示了一个权重矩阵 $ W $ 如何被量化为三值权重矩阵。
   - 量化的阈值 $ \Delta $ 被计算为：  
     $$
     \Delta = 0.7 \times \frac{1}{16} \|W\|_1 = 0.73
     $$
     其中， $ \|W\|_1 $ 是原始权重矩阵 $ W $ 的 L1 范数，即所有元素绝对值的平均值。

   - 确定非零权重的值 $ r_t $：
     $$
     r_t = \frac{1}{11} \|W_{W^T \neq 0}\|_1 = 1.5
     $$
     其中，$ \|W_{W^T \neq 0}\|_1 $ 是非零权重的 L1 范数。

&emsp;&emsp;二值量化和三值量化可以显著减小模型的尺寸和加速推理速度，但通常也会导致模型精度的下降。因此，这些方法常用于对精度要求不太高的应用场景或需要在低计算资源环境下运行的场景中。

## 4.8 模型量化对象

&emsp;&emsp;模型量化对象主要包括以下几个方面：
- 权重（Weights）：量化权重是最常见和流行的方法，它可以减少模型大小、内存使用和空间占用。
- 激活（Activations）：在实践中，激活通常占内存使用的大部分。因此，量化激活不仅可以大大减少内存使用，而且与权重量化结合时，可以充分利用整数计算来实现性能提升。
- KV缓存（KV cache）：量化KV缓存对于提高长序列生成的吞吐量至关重要。
- 梯度（Gradients）：与上面相比，梯度稍微不常见，因为它们主要用于训练。训练深度学习模型时，梯度通常是浮点数。它们主要用于减少分布式计算中的通信开销，也可以减少后向传递过程中的成本。



## 4.9 实践

- [k-means量化实践](./notebook/1.kmeans_quantzations.ipynb)
- [线性量化实践](./notebook/2.linear_quantizations.ipynb)
- [KL量化实践](./notebook/3.KL_quantization.ipynb)
- [量化感知训练实践](./notebook/4.pytorch_QAT.ipynb)


## 引用资料

- [Model Quantization 1: Basic Concepts](https://medium.com/@florian_algo/model-quantization-1-basic-concepts-860547ec6aa9)
- [Model Quantization 3: Timing and Granularity](https://blog.gopenai.com/model-quantization-3-timing-and-granularity-a0978c6e58d4)
- [A Visual Guide to Quantization](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization#%C2%A7symmetric-quantization)
