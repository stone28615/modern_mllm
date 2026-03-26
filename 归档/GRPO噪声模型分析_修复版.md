# GRPO噪声模型分析

## 问题背景

这是一道"把噪声当成物理量来对待"的好问题。你已经把系统抽象得足够干净了，下面我会在你给定的符号体系内，把核心问题拆成三个层次，并给出一个可计算、可推理的数学模型，而不是停在直觉层。

我会刻意避免"工程经验式"的说法，专注于：在 G=4、α>0、ε>0、多 RM 的条件下，梯度期望什么时候是对的，什么时候必然是噪声主导。

## 一、核心问题的再抽象（去掉实现细节）

你真正关心的不是 GRPO 形式本身，而是这一件事：

在 noisy reward + 小样本 group 下，Â 是否仍然是一个与"真实策略好坏"正相关的随机变量？

只要这个条件成立，PPO / GRPO 的 clip 只是个方差控制器；
如果不成立，clip 反而会"冻结错误方向"。

所以我们先把问题压缩成一个最小模型。

## 二、最小可分析模型（Single-step, Binary Good/Bad）

### 1. Ground truth（隐藏变量）

定义一个不可见的真实质量变量：
$$Y_i \in \{+1, -1\}$$

- $Y_i = +1$：第 i 个 sample 是正确输出
- $Y_i = -1$：是错误输出

由 policy 决定：
$$P(Y= -1) = \alpha,\quad P(Y=+1)=1-\alpha$$

### 2. Reward Model 的观测模型

对第 k 个 RM：
它对 pairwise 顺序判断的准确率是 $p_k = 1-\epsilon_k$

等价地，对单样本 reward 的符号判断：
$$\tilde Y_i^{(k)} = \begin{cases}
Y_i & \text{with prob } 1-\epsilon_k \\
- Y_i & \text{with prob } \epsilon_k
\end{cases}$$

这是一个对称噪声通道（binary symmetric channel）。

### 3. 连续 RM 的等价化（关键）

你现在用的是连续 reward + z-score。
在数学上，只要满足：
$$\mathbb{E}[r^k \mid Y=+1] - \mathbb{E}[r^k \mid Y=-1] = \Delta_k > 0$$

并且方差有限，那么在 group normalization 后：
$$\operatorname{sign}\!\left(\frac{r_i^k-\mu^k}{\sigma^k}\right)$$

在概率意义上 等价于一个带噪声的 $\tilde Y^{(k)}$。

你给的实验数据（均值差 1–2，σ≈0.3）意味着：
$$\text{SNR}_k = \frac{\Delta_k}{\sigma^k} \approx 3\sim6$$

这是一个相当强的单 RM 信号，但注意：
- GRPO 用的是 group 内的相对量，不是绝对量

## 三、G=4 的"结构性风险"

### 1. Group 内 Y 的分布

对一个 prompt，一组 4 个 sample：
$$N_{\text{bad}} \sim \text{Binomial}(4, \alpha)$$

当 $\alpha = 0.18$ 时：

| bad 个数 | 概率 |
|---------|------|
| 0       | 0.45 |
| 1       | 0.40 |
| 2       | 0.13 |
| 3       | 0.02 |
| 4       | 0.001 |

关键结论：
- 超过 58% 的 group 是「非纯净 group」

这意味着：
- $\mu^k, \sigma^k$ 本身是由混合分布估计的
- 归一化会引入 sign flipping 风险

### 2. 单 RM 的优势方向正确概率

对某个 sample i，我们关心：
$$P\big(\operatorname{sign}(A_i^{(k)}) = Y_i \big)$$

在 G=4、对称噪声、连续 RM 条件下，可以近似为：
$$P_k^{\text{eff}} \;\approx\; (1-\epsilon_k)\cdot P(\text{group 不退化}) \;+\; \frac{1}{2}\cdot P(\text{group 退化})$$

其中 退化 group 指：
- 全好
- 全坏
- 或 reward 排序几乎随机（σ 主导）

这一步是小 G 的硬伤：
即使 RM 很强，只要 group 信息不足，优势就会退化成 coin flip。

## 四、多 RM 叠加：什么时候"平均会救你"，什么时候不会

你现在用的是：
$$\hat A_i = \sum_{k=1}^K \frac{r_i^k - \mu^k}{\sigma^k}$$

这在统计上等价于：
$$\hat A_i = \sum_{k=1}^K w_k \cdot Z_i^{(k)} \quad (w_k=1)$$

### 1. 期望方向

$$\mathbb{E}[\hat A_i \mid Y_i] = Y_i \cdot \sum_{k=1}^K (1-2\epsilon_k)\cdot c_k$$

其中 $c_k>0$ 是由 group 结构压缩后的有效信号强度。

**必要条件（极其重要）：**
$$\boxed{\sum_{k=1}^K (1-2\epsilon_k)\cdot c_k > 0}$$

## 五、关键结论

- ε_k < 0.5 是硬门槛
- 多 RM 只能线性救信号，救不了结构性退化

❌ 必然失败的区域
- α↑（policy 已经开始变坏）
- 多 RM 中有 ε_k ≈ 0.5 的成员
- 全部 RM 用同权重
- G 固定为 4，且不引入跨 group baseline

这时你会观察到：
- loss 稳定
- KL 正常
- reward 不涨甚至反向

## 六、一个"理论一致"的改进方向（不涉及工程细节）

从理论上，最干净的修复只有三类：
1. 打破 group 退化
2. 引入跨 group baseline

---

## 理论推导的"地质剖面"

我们从"如何让 GRPO 更强"，一路被事实逼着走到"如何避免在错误信息结构下学习错方向"。
这不是悲观，而是信息论上的收敛。

### 第一阶段：最初的直觉模型（后来被打破）

初始隐含假设（你一开始也是按这个来分析的）
- Reward 是一个 标量场
- RM 的好坏可以用：Δ（好坏差值）和 p（pairwise 准确率）
- GRPO 的问题主要来自：G 小、方差大、优势被压扁
- 不需要新 RM，不需要新数据

### 改进方案

#### A1. 使用 mixed group 中的有效梯度
- 收益：把 RM 的强项（pairwise）直接用进 loss，mixed group 中的有效梯度显著增强，对 Δ 尺度完全不敏感
- 风险：pair 数是 O(G²)，算力会上升；如果 RM 在某些 prompt 上系统性偏置，会被放大；需要极小权重（这是辅助梯度，不是主目标）

#### A2. 从"group mean"改为"rank-preserving normalization"
做法是把：
$$A_i = \frac{r_i - \bar r}{\sigma}$$
换成：
$$A_i = \Phi(\operatorname{rank}(r_i))$$

#### A3. 让 RM 明确感知："你将来会被拿来在一个 group 里比较"
- 收益：RM 学到更稳定的 intra-group 排序，与 GRPO 的使用方式对齐
- 风险：需要构造 group-level 标注（成本），标注一致性要求更高，训练不稳定性上升（rank loss 很硬）

#### A4. 让 RM 显式学习"我不确定"
你已经预测了 σ，但没教它们什么时候该大、什么时候该小。

做法：
- 对 disagreement 高的样本，显式拉高 σ
- 把 σ 训练成"group-relative 不确定度"

---

## 关键洞察：single-sample 的理解

在你当前的 GRPO 设定里：
- single-sample ≠ token
- single-sample = 一个完整 trajectory / 一个完整输出
- 一个 group 里有 G 个 single-sample
- $A_i$ 是对第 i 条 trajectory 的 advantage
- 然后这个 $A_i$ 被复制广播到该 trajectory 的所有 timestep

也就是说：
$$\sum_i A_i = 0$$，但方向固定

RM 的偏好噪声在 group 内是**一致的**，而不是正负对称的白噪声。
换句话说：这是有结构的噪声，不是可抵消噪声。

在 α≈0.18、p≈0.7、G=4 下，"无条件 rank-only GRPO"在理论上是不可取的。
不是慢，不是弱，而是：在相当比例的 batch 上，更新方向与"降低 α"无关。

## 条件化 rank-only 策略

rank-only 不能是"always on"，而只能是 conditional。

唯一自洽的修正原则：只有在"这个 group 很可能是 mixed group"时，rank 才是有语义的。否则 rank 必须被压制，甚至禁用。

什么叫"很可能是 mixed group"？你可以只用现有 RM，做这三件事之一（或组合）：

### 方法 1：rank margin gate（最简单）
定义：
$$\Delta_{\text{rank}} = r_{\max} - r_{\min}$$

直觉：
- mixed group → Δ 往往更大
- pure group → Δ 来自噪声，分布更窄

规则：若 $\Delta_{\text{rank}} < \tau$：rank-only 被禁用

### 方法 2：variance-based detection
- pure group → reward 方差主要来自 ε
- mixed group → reward 方差来自语义差异

### 方法 3：cross-RM disagreement
- 对 pure group：多个 RM 应该给出相似的"这个 group 是 good/bad"判断
- 对 mixed group：多个 RM 之间 disagreement 更高

---

## 结论

问题根本上是空间不匹配：你已经把问题投影到了一个 single-sample + group-normalized 的标量空间，而这个投影对它来说是信息次优的。

不是你 RM 的错，也不是你 advantage 广播的错，而是空间不匹配。

你已经把问题挖到了不能再深的地方了。
再往下走，就只剩下一个问题了：
"是否接受这个投影，还是绕开它。"

## 技术路线图

pairwise RM 是对的，GRPO 的投影方式是错配的，但工程不能推倒重来。
于是优化只能分成两条线：
1. 算法侧：在不改变 RM 本体的前提下，尽量少丢信息
2. RM 训练范式侧：让 RM 学到"更适合被投影"的东西

这个可计算的中间变量 $S_{\text{mixed}}$ 可以作为方法论型论文的主线。