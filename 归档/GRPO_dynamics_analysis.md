# GRPO 训练动力学：核心问题抽象与数学模型

## 主题
在小样本组 ($G=4$) 与 多个非完美奖励信号 (Noisy Reward)构成的Reward Model 奖励系统约束下的策略优化稳定性分析。

## 1. 系统定义与符号抽象

我们将整个训练过程抽象为一个在噪声环境中通过稀疏采样估计梯度的随机过程。

- 系统状态 ($\pi_\theta$)： 由策略错误率 $\alpha$ 表征 ($P(Bad Output) = \alpha$)。
- 环境反馈 ($RM$)： 由奖励错误率 $\epsilon$ 表征 ($P(Wrong Signal) = \epsilon$)。
- 采样机制 ($G$)： 组大小 $G=4$。
- 核心算子 ($A$)：第k个Reward中的第i个样本的组内相对优势 $A_i^k = \frac{R_i^k - \mu^k}{\sigma^k}$。
- 最后的总体的优势：$A^{total}= \sum_k w_kA^k$

## 2. 符号说明 (Notation System)

为了保证分析的一致性，我们将对话中出现的所有符号统一如下：

| 符号 | 含义 | 备注/取值 |
|------|------|-----------|
| G | Group Size（组大小） | 文中固定 G = 4 |
| r, R | Reward（奖励） | 二分 0/1 或连续实数 |
| μ / mean | 组内奖励均值 | |
| σ / std | 组内奖励标准差 | |
| A | Advantage（优势值） | 最终梯度系数 |
| p | Reward Model 准确率 | p = P(判对) |
| ε | Reward Model 错误率 | ε = 1 − p；对称噪声 P(FP)=P(FN)=ε |
| α | Policy Model 错误率 | α = P(生成错误答案)；文中α = 0.18|
| ν | 连续 RM 噪声标准差 | 反映 RM 不确定性 |
| K | 多目标奖励个数 | |
| w_k | 第 k 个奖励权重 | 人工调节对 RM_k 的信任度 |
| σ_batch_max | Batch 级最大标准差 | 替代组内 σ 的稳定化方案 |

## RM 训练
### CoT RM
CoT RM 使用Qwen3-VL-8B 模型进行训练，最后根据畸形和正常的数据，在我们设定的规则下，模型输出CoT和判断的结果yes/no 以及畸形的类别。
### Head RM
Head RM则是Qwen3-VL-8B + MLP的head头进行训练。训练的数据是Pairwise的格式。
训练有使用BT Loss 来训练偏好。
也有使用Uncertainty Loss来训练模型结果
```
loss_type == "uncertainty":
            batch_size = rewards_A.shape[0]
            mean_chosen = rewards_A[:, 0]
            mean_rejected = rewards_B[:, 0]
            sigma_chosen = torch.exp(rewards_A[:, 1])
            sigma_rejected = torch.exp(rewards_B[:, 1])

            mean_z = mean_chosen - mean_rejected
            sigma_z = torch.sqrt(sigma_chosen**2 + sigma_rejected**2)

            z_samples = torch.randn(batch_size, 1000).to(sigma_z.device).to(
                torch.float16
            ) * sigma_z.unsqueeze(1).repeat(1, 1000) + mean_z.unsqueeze(1).repeat(
                1, 1000
            )
            loss = -torch.nn.functional.logsigmoid(z_samples).mean()
```

## 需要分析的大问题

在Policy Model在错误率为$\alpha$的情况下，第k个Reward Model 的错误率是 epsilon_k(分类冲激函数形式的奖励）或者是噪声 $ν_k$,分析模型在多个Reward Model优化目标优化下能够进行有效的Policy Model 模型优化与梯度更新。

## 需要分析的子问题

### 1. 连续性和二分型的Reward Model在Pairwise下的准确率具体的意义是什么？


### 2. 不同的Reward 之间他们如何相互影响和制约

### 3. 如何能够稳定地优化模型，使得效果性能提升

### 4. Reward Hacking的崩溃前兆是什么，会发生怎么样的情况
