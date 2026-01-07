# revise.py 数学流程与完整推导

本文档系统性地阐述 `revise.py` 中箭杆摆动检测算法的数学基础，从图像坐标系变换到频域分析，提供严谨的数学证明和推导过程，并详细描述每个步骤的实现细节和代码对应关系。

## 1. 坐标系与符号体系

### 1.1 像素坐标系
设图像帧 $I_t$ 的像素坐标为 $(x,y)$，其中 $x=0,1,\ldots,W-1$，$y=0,1,\ldots,H-1$，$W$ 和 $H$ 分别为图像的宽度和高度（单位：像素）。该坐标系以左上角为原点，$x$ 轴向右，$y$ 轴向下。

**代码实现**：在 `revise.py` 中，图像通过 OpenCV 的 `cv2.VideoCapture` 读取，获取的 `frame` 对象是一个三维数组（高度 × 宽度 × 通道数），其中通道顺序为 BGR（蓝-绿-红）。像素访问通过 `frame[y, x]` 进行，注意 OpenCV 使用 (行, 列) 顺序，即 y 坐标在前。

### 1.2 物理坐标系
设实际物理空间中的点用毫米（mm）度量。通过尺度标定建立像素坐标与物理坐标的线性映射关系。

### 1.3 符号约定
| 符号                        | 含义                          | 单位     |
| --------------------------- | ----------------------------- | -------- |
| $p=(x,y)$                   | 像素坐标点                    | pixel    |
| $s$                         | 尺度因子（物理长度/像素长度） | mm/pixel |
| $c_i=(x_i,y_i)$             | 第 $i$ 帧检测到的圆心         | pixel    |
| $r_i$                       | 第 $i$ 帧检测到的圆半径       | pixel    |
| $\bar c = (\bar x, \bar y)$ | 稳定参考圆心（均值）          | pixel    |
| $o_i = c_i - \bar c$        | 圆心偏移向量                  | pixel    |
| $d_i = s \| o_i \|$         | 径向偏移（物理长度）          | mm       |
| $f_s$                       | 视频采样频率                  | Hz       |
| $f_L, f_H$                  | 带通滤波器的下限和上限频率    | Hz       |
| $n$                         | 滤波器阶数                    | 无量纲   |

## 2. 尺度标定（Scale Calibration）

### 2.1 原理
在图像的首帧中选取两个已知实际距离的点 $P_1$ 和 $P_2$，通过它们在像素坐标系中的距离计算每个像素对应的物理长度。

**交互过程细节**：代码调用 `get_scale_reference(first_frame)` 函数，该函数：
1. 显示第一帧图像，允许用户通过鼠标点击选择两个点
2. 提供可视化反馈：显示点、连接线、中点和像素距离
3. 支持拖动调整点和线的位置
4. 用户确认后，通过控制台输入实际距离（单位：mm）

### 2.2 数学推导
设 $p_1=(x_1,y_1)$ 和 $p_2=(x_2,y_2)$ 为两个点在像素坐标系中的坐标。像素距离为
$$
\Delta p = \| p_1 - p_2 \| = \sqrt{(x_1-x_2)^2 + (y_1-y_2)^2} \quad \text{(pixel)}.
$$
已知两点的实际物理距离为 $D$（单位：mm），则尺度因子为
$$
s = \frac{D}{\Delta p} \quad \text{(mm/pixel)}.
$$

**代码对应**：在 `get_scale_reference` 函数中，像素距离计算使用 `np.linalg.norm(np.array(points[0]) - np.array(points[1]))`，然后计算 `scale_factor = real_distance / pixel_distance`。

### 2.3 误差分析与最佳实践
若像素坐标的测量误差为 $\delta x, \delta y$，则像素距离的相对误差为
$$
\frac{\delta(\Delta p)}{\Delta p} \approx \frac{(x_1-x_2)(\delta x_1-\delta x_2) + (y_1-y_2)(\delta y_1-\delta y_2)}{\Delta p^2}.
$$
尺度因子的相对误差与像素距离的相对误差相同。因此，为提高标定精度，应选择距离较远的两个点。

**实际操作建议**：
1. 选择图像中清晰可辨的特征点
2. 两点间的实际距离应尽可能大（覆盖感兴趣区域）
3. 重复测量取平均值可减少随机误差
4. 标定完成后，尺度因子 $s$ 将用于所有后续的物理量计算

## 3. 图像预处理（Image Preprocessing）

### 3.1 灰度转换
将彩色图像 $I(x,y)$（通常为 RGB 格式）转换为灰度图像 $g(x,y)$。常用加权平均法：
$$
g(x,y) = 0.299 \cdot R(x,y) + 0.587 \cdot G(x,y) + 0.114 \cdot B(x,y),
$$
其中 $R,G,B$ 分别为红、绿、蓝通道的强度。

**代码实现**：`cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` 自动执行此转换。OpenCV 使用与上述公式略有不同的系数（0.114, 0.587, 0.299 的 BGR 顺序），但效果相似。

### 3.2 高斯平滑（Gaussian Smoothing）
为抑制高频噪声，对灰度图像进行高斯滤波。二维高斯核函数为
$$
\mathcal{N}(x,y;\sigma) = \frac{1}{2\pi\sigma^2} \exp\left( -\frac{x^2+y^2}{2\sigma^2} \right).
$$

**参数选择细节**：代码中使用 $9\times 9$ 的卷积核，标准差 $\sigma \approx 2$。选择理由：
- 核大小 $9\times 9$：足够大以平滑噪声，但不会过度模糊边缘
- $\sigma=2$：在平滑噪声和保留边缘细节之间取得平衡
- 离散卷积核通过对连续高斯函数采样得到

平滑后的图像为
$$
g'(x,y) = (g * \mathcal{N})(x,y) = \sum_{u=-4}^{4}\sum_{v=-4}^{4} g(x-u,y-v) \cdot \mathcal{N}(u,v;\sigma).
$$

**代码对应**：`cv2.GaussianBlur(gray, (9, 9), 2)`。

### 3.3 局部掩膜（Local Mask）
为减少计算量并提高鲁棒性，仅在前一帧圆心 $c_{i-1}$ 的邻域内搜索圆。定义掩膜函数为
$$
M_{i-1}(x,y) = \mathbf{1}\left\{ \| (x,y) - c_{i-1} \| \leq r_{i-1} + \Delta r \right\},
$$
其中 $\mathbf{1}\{\cdot\}$ 为指示函数，$\Delta r = 20$ 像素为搜索容差。

**搜索容差选择**：$\Delta r = 20$ 像素的考虑：
1. 足够大以容纳帧间运动：箭杆摆动速度有限，相邻帧间位移通常小于20像素
2. 足够小以排除干扰：减少其他圆形特征的误检
3. 经验值：通过实验确定，在跟踪稳定性和计算效率间取得平衡

掩膜后的图像为
$$
g''(x,y) = g'(x,y) \cdot M_{i-1}(x,y).
$$
在代码中，这通过 `cv2.bitwise_and(blurred, blurred, mask=mask)` 实现。

**掩膜创建细节**：
```python
mask = np.zeros(frame.shape[:2], dtype=np.uint8)
cv2.circle(mask, last_known_center, search_radius, 255, -1)
```
其中 `search_radius = initial_radius + 20`。

## 4. 霍夫圆检测与残差分析（Hough Circle Detection and Residual Analysis）

### 4.1 圆方程与参数空间
圆的参数方程为
$$
(x - a)^2 + (y - b)^2 = r^2,
$$
其中 $(a,b)$ 为圆心，$r$ 为半径。

对于图像中的每个边缘点 $(x,y)$，其在参数空间 $(a,b,r)$ 中对应一个圆锥面。所有边缘点投票后，参数空间中累积值最高的点即为检测到的圆。

### 4.2 残差（Residual）定义
在圆拟合问题中，残差衡量边缘点到候选圆的距离偏差。对于候选圆 $(a_k, b_k, r_k)$ 和边缘点集 $\{(x_j, y_j)\}_{j=1}^M$，每个边缘点的径向残差为
$$
\epsilon_j = \sqrt{(x_j - a_k)^2 + (y_j - b_k)^2} - r_k.
$$
该残差表示边缘点到圆周的符号距离：正值为圆外，负值为圆内。

整体拟合残差通常用均方误差（MSE）或绝对误差和衡量：
$$
\text{MSE}_k = \frac{1}{M} \sum_{j=1}^M \epsilon_j^2, \quad
\text{MAE}_k = \frac{1}{M} \sum_{j=1}^M |\epsilon_j|.
$$

**几何意义**：残差越小，表明边缘点越接近圆周，圆拟合质量越高。

### 4.3 残差-距离综合选优准则
在理想情况下，应选择残差最小的候选圆。然而，在实际视频序列中，由于噪声、遮挡和边缘不完整，仅凭残差选优可能导致跳动。因此，结合时空连续性，采用残差与距离的综合准则：

设第 $i-1$ 帧的圆心为 $c_{i-1} = (x_{i-1}, y_{i-1})$，半径为 $r_{i-1}$。对于当前帧的候选圆集 $\{(a_k, b_k, r_k)\}$，定义综合代价函数：
$$
J_k = \alpha \cdot \frac{\text{MSE}_k}{\max(\text{MSE})} + (1-\alpha) \cdot \frac{\| (a_k, b_k) - c_{i-1} \|}{\max(\text{distance})},
$$
其中 $\alpha \in [0,1]$ 为权重系数，$\max(\text{MSE})$ 和 $\max(\text{distance})$ 为当前候选集中的最大值，用于归一化。

最优候选索引为
$$
k^* = \arg\min_k J_k.
$$

**权重选择**：
- $\alpha$ 接近 1：强调拟合质量，适合静态或高对比度场景。
- $\alpha$ 接近 0：强调运动连续性，适合快速运动或低质量图像。
- 典型值：$\alpha = 0.5$，平衡两者。

**实际实现考虑**：在 `revise.py` 中，为简化计算并保证实时性，当前版本仅采用距离最小化准则（即 $\alpha = 0$）：
$$
k^* = \arg\min_k \| (a_k, b_k) - c_{i-1} \|.
$$
若需要更精确的选优，可扩展为综合代价函数。

### 4.4 霍夫梯度法实现细节
代码中使用 OpenCV 的 `cv2.HoughCircles` 函数，它采用霍夫梯度法，主要步骤为：
1. **边缘检测**：使用 Sobel 算子计算图像梯度
2. **非极大值抑制**：沿梯度方向保留局部最大值
3. **阈值处理**：保留梯度幅值大于 `param1` 的边缘点
4. **圆心累加**：对每个边缘点，沿梯度方向在半径范围内累加圆心候选
5. **圆心选择**：对累加图像进行非极大值抑制，得到圆心候选
6. **半径估计**：对每个圆心候选，计算边缘点到圆心的距离直方图，选择峰值作为半径

**残差在霍夫检测中的隐含作用**：霍夫投票本质上是边缘点对满足圆方程的参数进行累积，累积值高的参数对应残差较小的圆。因此，霍夫检测已隐式地利用了残差信息。

**关键参数及其含义**：
- `dp=1.2`：累加器分辨率与图像分辨率的反比
- `minDist=100`：检测到的圆心之间的最小距离
- `param1=100`：Canny 边缘检测的高阈值
- `param2=30`：圆心累加器的阈值，值越小检测到的圆越多
- `minRadius=initial_radius-15`：最小半径约束
- `maxRadius=initial_radius+15`：最大半径约束

### 4.5 候选选择（Candidate Selection）
设第 $i-1$ 帧的圆心为 $c_{i-1}$，半径为 $r_{i-1}$。在当前帧的候选圆集 $\{(a_k,b_k,r_k)\}$ 中，按以下准则筛选：

1. **半径约束**：$r_k \in [r_{i-1}-15, r_{i-1}+15]$。
   - 理由：箭杆半径变化缓慢，相邻帧间半径变化通常小于15像素
   - 实现：通过 `cv2.HoughCircles` 的 `minRadius` 和 `maxRadius` 参数实现

2. **中心位置约束**：$(a_k,b_k)$ 位于掩膜区域内（即 $M_{i-1}(a_k,b_k)=1$）。
   - 实现：霍夫检测已在掩膜图像上进行，自然满足此条件

3. **距离准则**（当前实现）：选择与 $c_{i-1}$ 欧氏距离最小的候选：
   $$
   k^* = \arg\min_k \| (a_k,b_k) - c_{i-1} \|.
   $$
   - 理由：箭杆运动连续，相邻帧间圆心位置变化最小
   - 实现：代码中计算所有候选圆心与上一帧圆心的距离，选择最小者

**扩展讨论**：若采用残差-距离综合选优，则需计算每个候选圆的拟合残差。由于霍夫检测已提供候选圆，可额外计算每个候选圆与局部边缘点集的残差，然后按综合代价函数选择。这虽然增加计算量，但能提高选优的鲁棒性。

**特殊情况处理**：
- 若无候选：沿用前一帧的圆心 $c_{i-1}$
- 若多个候选距离相等：可考虑引入残差作为次级准则（当前取第一个）
- 若候选半径超出范围：已被 `cv2.HoughCircles` 过滤

**代码实现细节**：
```python
if circles is not None:
    circles = np.uint16(np.around(circles))
    if len(circles[0, :]) > 1:
        distances = np.linalg.norm(circles[0, :, :2] - np.array(last_known_center), axis=1)
        best_circle = circles[0, np.argmin(distances)]
    else:
        best_circle = circles[0, 0]
    cx, cy, r = best_circle
    center_detected = (cx, cy)
    detected_centers.append(center_detected)
    last_known_center = center_detected
else:
    detected_centers.append(last_known_center)
```

## 5. 轨迹与偏移（Trajectory and Offset）

### 5.1 稳定参考圆心计算
设共检测到 $N$ 帧的圆心 $\{c_1, c_2, \ldots, c_N\}$，定义稳定参考圆心为样本均值：
$$
\bar c = \frac{1}{N} \sum_{i=1}^N c_i.
$$
该点代表了圆心的平均位置，用于消除静态偏移。

**计算细节**：
- 使用 `np.mean(detected_centers, axis=0)` 计算
- 得到的 $\bar c$ 是二维向量 $(\bar x, \bar y)$
- 此计算在所有帧处理完成后进行

### 5.2 偏移向量计算
第 $i$ 帧的偏移向量为
$$
o_i = c_i - \bar c = (x_i - \bar x, \; y_i - \bar y).
$$
其物理意义为当前圆心相对于平均位置的偏差。

**实现**：
```python
offsets = detected_centers - stable_center
```

### 5.3 径向偏移计算
径向偏移（即偏移向量的模长）在物理坐标系中的值为
$$
d_i = s \cdot \| o_i \| = s \sqrt{(x_i-\bar x)^2 + (y_i-\bar y)^2} \quad \text{(mm)}.
$$

**代码对应**：
```python
radial_offsets = np.linalg.norm(offsets, axis=1) * scale_factor
```

### 5.4 最大两两距离计算
为评估圆心的整体波动范围，计算任意两帧圆心之间的最大物理距离：
$$
D_{\max} = \max_{1 \leq i < j \leq N} s \| c_i - c_j \|.
$$
该指标反映了圆心在整个时间段内的最大位移。

**计算复杂度**：需要计算 $O(N^2)$ 个距离，对于典型视频（几百到几千帧）是可接受的。

**代码实现**：
```python
if len(detected_centers) >= 2:
    raw_pairwise_distances = []
    for i in range(len(detected_centers)):
        for j in range(i+1, len(detected_centers)):
            dist = np.linalg.norm(detected_centers[i] - detected_centers[j]) * scale_factor
            raw_pairwise_distances.append(dist)
    max_raw_pairwise_distance = max(raw_pairwise_distances) if raw_pairwise_distances else 0
```

## 6. 带通滤波（Bandpass Filtering）

### 6.1 滤波器设计目标
保留频率在 $[f_L, f_H]$ 范围内的信号成分，抑制该频带外的噪声。采用 Butterworth 滤波器，因其在通带内具有最大平坦的幅频特性。

**频率范围选择**：箭杆摆动频率通常在 1-30 Hz 范围内，具体取决于箭杆材料和发射条件。代码中默认使用 $(1, 30)$ Hz，但可根据实际情况调整。

### 6.2 模拟 Butterworth 低通滤波器原型
$n$ 阶 Butterworth 低通滤波器的平方幅频响应为
$$
|H(j\Omega)|^2 = \frac{1}{1 + (\Omega/\Omega_c)^{2n}},
$$
其中 $\Omega$ 为模拟角频率（rad/s），$\Omega_c$ 为截止角频率。

**阶数选择**：默认 $n=4$。阶数越高，过渡带越陡峭，但：
1. 相位非线性更严重（但零相位滤波可解决此问题）
2. 计算复杂度增加
3. 数值稳定性可能降低

### 6.3 数字滤波器设计：双线性变换法
将模拟滤波器转换为数字滤波器，采用双线性变换：
$$
s = \frac{2}{T} \frac{z-1}{z+1},
$$
其中 $T = 1/f_s$ 为采样间隔。

双线性变换将模拟频率 $\Omega$ 映射到数字频率 $\omega$ 的关系为
$$
\Omega = \frac{2}{T} \tan\left( \frac{\omega}{2} \right).
$$
为避免频率畸变，需进行预畸变（pre-warping）：设计模拟滤波器时，将截止频率 $\Omega_c$ 设置为
$$
\Omega_c = 2\pi f_c, \quad f_c = \frac{2}{T} \tan\left( \frac{\pi f_c^{digital}}{f_s} \right),
$$
其中 $f_c^{digital}$ 为期望的数字截止频率。

对于带通滤波器，有两个截止频率 $f_L$ 和 $f_H$，分别进行预畸变得到 $\Omega_L$ 和 $\Omega_H$，然后设计模拟带通滤波器，再通过双线性变换得到数字滤波器系数 $(b,a)$。

**代码实现**：在 `butter_bandpass` 函数中：
```python
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs  # 奈奎斯特频率
    low = lowcut / nyq  # 归一化低频截止
    high = highcut / nyq  # 归一化高频截止
    b, a = butter(order, [low, high], btype='band')
    return b, a
```
这里 `butter` 函数来自 `scipy.signal`，自动处理预畸变和双线性变换。

### 6.4 零相位滤波（Zero-Phase Filtering）原理
常规滤波会引入相位滞后，影响时域波形。零相位滤波通过前向-后向滤波消除相位失真：

**算法步骤**：
1. 对信号 $x[n]$ 进行前向滤波，得到 $y_1[n]$
2. 将 $y_1[n]$ 反转得到 $y_1[-n]$
3. 对 $y_1[-n]$ 进行相同的滤波，得到 $y_2[n]$
4. 将 $y_2[n]$ 反转，得到最终输出 $y[n] = y_2[-n]$

**数学性质**：
- 幅频响应：$|H_{\text{zero-phase}}(e^{j\omega})| = |H(e^{j\omega})|^2$
- 相位响应：恒为零
- 滤波器阶数：等效滤波器阶数为 $2n-1$

**代码实现**：使用 `scipy.signal.filtfilt(b, a, data)`。

### 6.5 滤波应用
对三个信号分别进行带通滤波：
- $x$ 方向偏移：$\tilde o_x = \text{filtfilt}(b,a, o_x)$
- $y$ 方向偏移：$\tilde o_y = \text{filtfilt}(b,a, o_y)$
- 径向偏移：$\tilde d = \text{filtfilt}(b,a, d)$

**数据长度要求**：零相位滤波需要数据长度大于 $3n$（其中 $n$ 为滤波器阶数），否则会报错。代码中检查 `len(x_offsets) > filter_order * 3`。

滤波后的圆心轨迹为
$$
\tilde c_i = \bar c + \frac{1}{s} (\tilde o_x, \tilde o_y).
$$

## 7. 频谱分析（Frequency Spectrum Analysis）

### 7.1 周期图法（Periodogram）原理
对于离散信号 $x[n], n=0,1,\ldots,N-1$，其功率谱密度（PSD）的估计采用周期图法：
$$
\hat{P}_{xx}(f_k) = \frac{1}{N f_s} \left| X[k] \right|^2,
$$
其中 $X[k]$ 为 $x[n]$ 的离散傅里叶变换（DFT）：
$$
X[k] = \sum_{n=0}^{N-1} x[n] e^{-j 2\pi k n / N}, \quad k=0,1,\ldots,N-1.
$$
频率点 $f_k = k f_s / N$。

**窗函数**：默认使用矩形窗，即不对数据加窗。这会导致频谱泄漏，但对于较长数据段和周期性信号，影响较小。

对于实信号，单边功率谱密度定义为
$$
P_{xx}(f_k) = \begin{cases}
\hat{P}_{xx}(0), & k=0, \\
2 \hat{P}_{xx}(f_k), & 1 \leq k \leq N/2-1, \\
\hat{P}_{xx}(f_s/2), & k=N/2 \text{（当 $N$ 为偶数）}.
\end{cases}
$$

**代码实现**：使用 `scipy.signal.periodogram(signal, fs)` 计算。

### 7.2 主频检测算法
在感兴趣的频率范围 $[1,30]$ Hz 内寻找功率谱的峰值：
$$
f_{\max} = \arg \max_{f \in [1,30]} P_{xx}(f), \quad A_{\max} = \max_{f \in [1,30]} P_{xx}(f).
$$
$f_{\max}$ 即为信号的主频，$A_{\max}$ 为其功率。

**实现细节**：
```python
mask = (f >= 1) & (f <= 30)
if np.any(mask):
    dom_freq = f[mask][np.argmax(Pxx[mask])]
    dom_amp = np.max(Pxx[mask])
```

**注意事项**：
1. 频率分辨率为 $\Delta f = f_s / N$
2. 主频检测精度受频率分辨率限制
3. 若信号包含多个相近频率分量，可能检测到最大峰值而非实际主频
4. 可考虑使用插值或加窗提高频率估计精度

## 8. 统计量（Statistics）

### 8.1 均值与标准差计算
滤波后的径向偏移 $\tilde d_i$ 的样本均值和样本标准差分别为
$$
\mu = \frac{1}{N} \sum_{i=1}^N \tilde d_i, \quad
\sigma = \sqrt{ \frac{1}{N} \sum_{i=1}^N (\tilde d_i - \mu)^2 }.
$$

**代码实现**：
```python
mean_deviation = np.mean(filtered_radial)
std_deviation = np.std(filtered_radial)
```

**注意**：这里使用样本标准差（除以 $N$ 而非 $N-1$），因为数据量通常较大，两者差异可忽略。

### 8.2 最大偏移
$$
d_{\max} = \max_{1 \leq i \leq N} \tilde d_i.
$$

### 8.3 最大两两距离（滤波后）
计算滤波后圆心轨迹 $\tilde c_i$ 中任意两点之间的最大物理距离：
$$
\tilde D_{\max} = \max_{1 \leq i < j \leq N} s \| \tilde c_i - \tilde c_j \|.
$$

**物理意义**：表示滤波后圆心位置的最大波动范围。

## 9. 可视化与输出

### 9.1 时域曲线
绘制原始与滤波后的偏移曲线：
1. 径向偏移
2. X 方向偏移  
3. Y 方向偏移

**时间轴构建**：`time_axis = np.arange(len(radial_offsets)) / fps`

### 9.2 频域图谱
绘制功率谱密度图：
1. 原始信号频谱
2. 滤波后信号频谱

**绘图设置**：
- 使用对数坐标（`semilogy`）更好地显示不同幅值的频率分量
- 频率范围限制在 $[0, \min(100, f_s/2)]$ Hz
- 添加网格线便于读数

### 9.3 轨迹图
绘制圆心运动轨迹：
- 横坐标：X 位置（mm）
- 纵坐标：Y 位置（mm）
- 起点标记为绿色圆点
- 终点标记为蓝色圆点
- 轨迹线为红色

**几何解释**：轨迹图直观显示圆心的运动模式，如圆形、椭圆形或随机运动。

## 10. 残差分析与算法鲁棒性

### 10.1 残差在圆检测中的作用
残差是评估圆拟合质量的关键指标。在 `revise.py` 的算法框架中，残差从两个层面影响检测结果：

1. **隐式层面**：霍夫变换通过累加器投票机制，本质上选择了使边缘点残差最小的圆参数。每个边缘点对满足圆方程 $(x-a)^2+(y-b)^2=r^2$ 的参数进行投票，累积值高的参数对应残差较小的圆。

2. **显式层面**：若需要更精确的选优，可显式计算每个候选圆的残差，并与距离准则结合。当前代码为简化计算，仅使用距离准则，但保留了扩展为残差-距离综合选优的可能性。

### 10.2 残差的时间序列分析
将每帧的拟合残差 $\epsilon_i$ 组成时间序列 $\{\epsilon_1, \epsilon_2, \ldots, \epsilon_N\}$，可进行以下分析：

1. **残差趋势**：若残差逐渐增大，可能表明：
   - 目标逐渐失焦或模糊
   - 光照条件变化导致边缘质量下降
   - 箭杆发生形变

2. **残差突变**：单帧残差突然增大可能表示：
   - 瞬时遮挡
   - 运动模糊
   - 检测错误（误检或漏检）

3. **残差与摆动关系**：箭杆摆动可能导致残差周期性变化，因为不同相位时边缘可见性不同。

### 10.3 基于残差的异常检测
定义残差阈值 $\epsilon_{\text{th}}$，当 $\epsilon_i > \epsilon_{\text{th}}$ 时标记为异常帧。异常处理策略包括：
1. **使用历史均值**：用前 $k$ 帧的平均圆心代替当前帧
2. **插值**：用前后帧的圆心进行线性插值
3. **丢弃**：直接忽略异常帧，仅用于统计分析

### 10.4 残差与置信度
可以为每帧检测结果分配置信度 $w_i$，与残差负相关：
$$
w_i = \exp\left(-\frac{\epsilon_i^2}{2\sigma_\epsilon^2}\right),
$$
其中 $\sigma_\epsilon$ 为残差的标准差。加权后的统计量（如加权均值、加权标准差）能更准确地反映真实摆动。

## 11. 算法参数总结与调优建议

### 11.1 关键参数及其默认值
| 参数                | 默认值    | 含义               | 调优建议                       |
| ------------------- | --------- | ------------------ | ------------------------------ |
| 高斯核大小          | 9×9       | 平滑强度           | 噪声大时增大，细节重要时减小   |
| 高斯 $\sigma$       | 2         | 平滑范围           | 同核大小调整                   |
| 搜索容差 $\Delta r$ | 20        | 搜索区域半径增量   | 根据运动速度调整               |
| 半径容差            | 15        | 允许的半径变化范围 | 根据箭杆直径稳定性调整         |
| 霍夫 `param2`       | 30        | 圆心检测阈值       | 检测不到圆时减小，误检多时增大 |
| 滤波器阶数 $n$      | 4         | 滤波器陡峭度       | 需要锐利截止时增大，注意稳定性 |
| 通带 $[f_L, f_H]$   | [1,30] Hz | 感兴趣频率范围     | 根据实际摆动频率调整           |
| 残差权重 $\alpha$   | 0         | 残差在选优中的权重 | 图像质量高时增大，运动快时减小 |

### 11.2 残差相关参数调优
1. **残差阈值 $\epsilon_{\text{th}}$**：可根据历史残差数据动态设置，例如：
   $$
   \epsilon_{\text{th}} = \mu_\epsilon + 3\sigma_\epsilon,
   $$
   其中 $\mu_\epsilon$ 和 $\sigma_\epsilon$ 为残差序列的均值和标准差。

2. **权重系数 $\alpha$**：在残差-距离综合选优中，$\alpha$ 的调整原则：
   - 高对比度、清晰边缘：$\alpha \in [0.7, 1.0]$，侧重拟合质量
   - 快速运动、运动模糊：$\alpha \in [0.0, 0.3]$，侧重运动连续性
   - 一般情况：$\alpha \in [0.4, 0.6]$，平衡两者
