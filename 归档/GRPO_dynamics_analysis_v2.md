# GRPO 多Reward系统训练：核心问题抽象与数学模型

## 主题

在小样本组 ($G=4$) 与 多个非完美奖励信号 (Noisy Reward)构成的Reward Model 奖励系统约束下的策略优化稳定性分析。

## 1. 系统定义与符号抽象

将整个训练过程抽象为一个在噪声环境中通过稀疏采样估计梯度的随机过程。

- 系统状态 ($\pi_\theta$)： 由策略错误率 $\alpha$ 表征 ($P(Bad Output) = \alpha$)。
- 环境反馈 ($RM$)： 由奖励错误率 $\epsilon$ 表征 ($P(Wrong Signal) = \epsilon$)。
- 采样机制 ($G$)： 组大小 $G=4$。
- 核心算子 ($A$)：第k个Reward中的第i个样本的组内相对优势 $A_i^k = \frac{R_i^k - \mu^k}{\sigma^k}$。
- 最后的总体的优势：$A^{total}= \sum_k w_kA^k$
- pure group:在GRPO推理过程的一个Group $\{o_i\}^G_{i=1} $中，全对或者全错的Group。
- mixed group:在GRPO推理过程的一个Group $\{o_i\}^G_{i=1} $中，有对有错的Group。

## 2. 符号说明 (Notation System)

为了保证分析的一致性，我们将对话中出现的所有符号统一如下：

| 符号 | 含义 | 备注/取值 |
|------|------|-----------|
| G | Group Size（组大小） | 文中固定 G = 4 |
| r, R | Reward（奖励） |连续实数 |
| μ / mean | 组内奖励均值 | |
| σ / std | 组内奖励标准差 | |
| A | Advantage（优势值） |  |
| p | Reward Model Piarwise 评测集下的判断的准确率 | p = P(判对pair) |
| ε | Reward Model 错误率 | ε = 1 − p |
| α | Policy Model 错误率 | α = P(生成错误答案)；文中α = 0.18的量级|
| ν | 连续 RM 噪声标准差 | 反映 RM 不确定性 |
| K | 多目标奖励个数 | |
| w_k | 第 k 个奖励权重 | 对 RM_k 的信任度 |
| σ_batch_max | Batch 级最大标准差 | 替代组内 σ 的稳定化方案 |

## RM 训练

### Head RM

Head RM则是Qwen3-VL-8B + MLP的head头进行训练。训练的数据是Pairwise的格式。
训练使用Uncertainty Loss来训练模型结果。
推理的时候推理回归的是一个连续的实数（也就是Reward[:,0] ）。
pairwise 空间里，RM 实际学到的是什么？
我这个的 Head RM 的训练目标本质是：

$$P\big(\mu(o^+) > \mu(o^-)\big) = p \approx 0.7$$

这是一个**二元比较器**，它学到的是：

> "当我看到两个 trajectory 时，我更可能判断哪一个更好"

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

### 其中一个 Head RM训练结果

训练的Head RM结果来说，其中在Pairwise格式的测试集的准确率在 70% （仅使用 $\mu$ 作为一个回归的结果，$\sigma$预测之后并未在GRPO训练的时候使用。

### 目前训练 GRPO使用的总共优势估计公式：


$$
\mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E}_{(q,a) \sim \mathcal{D}, \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|q)} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min\left( r_{i,t}(\theta) \hat{A}_{i,t},\ \text{clip}\left(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon\right) \hat{A}_{i,t} \right) ) \right) \right] 
$$

其中：

$$
r_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})} \tag{6}
$$

在 DanceGRPO 中，优势函数 $\hat{A}_{i,t}$ 为多奖励归一化优势：

$$
\hat{A}_{i,t} = \sum_{k=1}^K \frac{r_{i,t}^k - \mu^k}{\sigma^k} \tag{7}
$$

其中：
- $r_{i,t}^k$ 是第 $i$ 个样本、第 $t$ 步在第 $k$ 个奖励模型下的奖励；
- $\mu^k, \sigma^k$ 是第 $k$ 个奖励模型的全局均值与标准差（预计算或在线估计）；
- $\varepsilon$ 为 clip 范围参数；
- $\beta$ 为 KL 正则项系数；
- $\pi_{\text{ref}}$ 为参考策略（如初始策略或历史平均策略）。

$Δθ ∝∑_{i=1}^{G}​A_i​ \cdot ∇_θ\log{π_θ​(a_i​)}$
$$ A^k_i =\frac{R^{k}_i -\mu^k}{\sigma_{batchmax}^k} $$
$$ A^{total}_i = \sum_k w_k \cdot (\sum_{i}^{G}(\frac{R^{k}_i -\mu^k}{\sigma_{batchmax}^k}))$$

## 分析中可能容易忽略的问题

### 1. 连续RM和离散RM的区别
相比于离散的RM来说，连续的RM在pure group中会产生非零的梯度更新，因为 $ \mu_i^+ \neq u_j^+ $，而$ \mu_i^+ \neq u_j^+ $在离散RM中这是严格成立的。

### 2. 对于连续RM噪声的数学建模问题
这里是需要详细考虑的，是否和可能的各类因素有关系，做出细致的分析。

### 3. pairwise 准确率的详细解释
其中$o^+$和$o^-$分别表示静态评测集中对应的人类偏好的好样本和坏样本
$$P\big(\mu(o^+) > \mu(o^-)\big) = p $$

### 4. 对于GRPO产生的样本组 $ \{o_i\}_{i = 1}^{G} $可能会有一定的概率由于RM的准确率有限而造成误判


### 5. 不同Reward之间的组合会带来的影响

## 需要分析的大问题

在Policy Model在错误率为$\alpha$的情况下，第k个Reward Model 的噪声是 $ν_k$,Pairwise比较的准确率是p，分析模型在多个Reward Model优化目标优化下能够进行有效的Policy Model 模型优化与梯度更新。

## 需要分析的子问题

### 1. 连续性和二分型的Reward Model在Pairwise下的准确率具体的意义是什么？

### 2. 不同的Reward 之间他们如何相互影响和制约

### 3. 如何能够稳定地优化模型，使得效果性能提升

### 4. Reward Hacking的崩溃前兆是什么，会发生怎么样的情况

### 5. 连续性RM对于四个好样本或者四个坏样本即使经过了 $\sigma_{batchmax}$之后，由于奖励是连续性实数，并非完全相同，归一化之后会出现有正有负的相对优势。


```
**Algorithm 1: DanceGRPO Training Algorithm**

Require: Initial policy model $\pi_\theta$; reward models $\{R_k\}_{k=1}^K$; prompt dataset $\mathcal{D}$; timestep selection ratio $\tau$; total sampling steps $T$

Ensure: Optimized policy model $\pi_\theta$

1: for training iteration = 1 to $M$ do
2:     Sample batch $\mathcal{D}_b \sim \mathcal{D}$  $\triangleright$ Batch of prompts
3:     Update old policy: $\pi_{\theta_{\text{old}}} \leftarrow \pi_\theta$
4:     for each prompt $\mathbf{c} \in \mathcal{D}_b$ do
5:         Generate $G$ samples: $\{\mathbf{o}_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot|\mathbf{c})$ with the same random initialization noise
6:         Compute rewards $\{r_i^k\}_{i=1}^G$ using each $R_k$
7:         for each sample $i \in 1..G$ do
8:             Calculate multi-reward advantage: $A_i \leftarrow \sum_{k=1}^K \frac{r_i^k - \mu^k}{\sigma^k}$  $\triangleright$ $\mu^k, \sigma^k$ per-reward statistics
9:         end for
10:        Subsample $\lfloor \tau T \rfloor$ timesteps $\mathcal{T}_{\text{sub}} \subset \{1..T\}$
11:        for $t \in \mathcal{T}_{\text{sub}}$ do
12:            Update policy via gradient ascent: $\theta \leftarrow \theta + \eta \nabla_\theta \mathcal{J}$
13:        end for
14:    end for
15: end for
```