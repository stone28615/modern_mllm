# On-Policy Distillation 核心推导学习笔记（含必要背景）

> 目标：把 Thinking Machines blog 里“on-policy distillation = token-level reverse KL + RL(PPO) 脚手架”的核心推导整理成可复习的笔记。
> 关键来源：原文对 reverse KL、advantages 的定义与训练循环说明。

---

## 0. 基础符号（Notation）

- **Teacher policy（老师策略）**：$\pi_T(a\mid s)$  
- **Student policy（学生策略）**：$\pi_\theta(a\mid s)$，参数为 $\theta$
- **状态（state）** $s$：在语言模型中通常是前缀 $x_{<t}$
- **动作（action）** $a$：下一 token $x_t$
- 一段生成序列：$x_{1:T}$，其中第 $t$ 步满足
  $$
  x_t \sim \pi_\theta(\cdot \mid x_{<t})
  $$

---

## 1. On-policy distillation 的关键目标：最小化 Reverse KL

### 1.1 Reverse KL（反向 KL）定义

在单个状态 $s$ 上，student 相对 teacher 的 **reverse KL**：
$$
D_{\mathrm{KL}}\!\left(\pi_\theta(\cdot\mid s)\,\Vert\,\pi_T(\cdot\mid s)\right)
= \sum_a \pi_\theta(a\mid s)\Bigl(\log \pi_\theta(a\mid s)-\log \pi_T(a\mid s)\Bigr)
$$

整段序列（逐 token）的目标可以写成对 student 轨迹分布的期望（on-policy）：
$$
\mathcal{L}(\theta)
= \mathbb{E}_{x\sim \pi_\theta}\left[\sum_{t=1}^{T} 
\Bigl(\log \pi_\theta(x_t\mid x_{<t})-\log \pi_T(x_t\mid x_{<t})\Bigr)\right]
$$

---

## 2. 采样估计：token-level reverse KL（原文伪代码的数学化）

在一条 student 采样到的轨迹上，对每个时间步 $t$：
$$
\widehat{\mathrm{rKL}}_t
= \log \pi_\theta(x_t\mid x_{<t})-\log \pi_T(x_t\mid x_{<t})
$$

对应原文实现（logprob 形式）：
- `reverse_kl = sampled_logprobs - teacher_logprobs` 

---

## 3. 把它“翻译成 RL 的奖励”：reward/advantage 的构造

### 3.1 定义每步 reward（把 reverse KL 当负奖励）

令 per-token reward：
$$
r_t \equiv \log \pi_T(x_t\mid x_{<t})-\log \pi_\theta(x_t\mid x_{<t})
= -\widehat{\mathrm{rKL}}_t
$$

于是：teacher 越“认可”的 token（$\log\pi_T$ 高）会得到更大奖励；student 越“自信但老师不认可”（$\log\pi_\theta$ 高而 $\log\pi_T$ 低）会被惩罚。

### 3.2 原文的 advantage 定义

原文直接设：
- `advantages = -reverse_kl` 

数学上就是：
$$
A_t \equiv -\widehat{\mathrm{rKL}}_t
= \log \pi_T(x_t\mid x_{<t})-\log \pi_\theta(x_t\mid x_{<t})
$$

> 注：这里把 advantage 直接取为即时 reward（并且文中经验_relied_上折扣因子设为 0）。

---

## 4. 关键梯度推导：为什么它等价于“在 student 访问到的状态上逼近 teacher”

考虑单个状态 $s$ 的 reverse KL：
$$
D_{\mathrm{KL}}(\pi_\theta\Vert\pi_T)
= \sum_a \pi_\theta(a\mid s)\left(\log \pi_\theta(a\mid s)-\log \pi_T(a\mid s)\right)
$$

对 $\theta$ 求梯度，用 log-derivative trick（$\nabla \pi=\pi\nabla\log\pi$）：
$$
\nabla_\theta D_{\mathrm{KL}}
= \sum_a \nabla_\theta \pi_\theta(a\mid s)\left(\log \pi_\theta(a\mid s)-\log \pi_T(a\mid s)\right)
+ \sum_a \pi_\theta(a\mid s)\nabla_\theta \log \pi_\theta(a\mid s)
$$

整理得（将两部分合并）：
$$
\nabla_\theta D_{\mathrm{KL}}
= \mathbb{E}_{a\sim \pi_\theta(\cdot\mid s)}
\left[\nabla_\theta \log \pi_\theta(a\mid s)\cdot 
\Bigl(\log \pi_\theta(a\mid s)-\log \pi_T(a\mid s)+1\Bigr)\right]
$$

因此最小化 reverse KL 的梯度方向（下降）相当于最大化：
$$
\mathbb{E}_{a\sim\pi_\theta}\left[\nabla_\theta\log\pi_\theta(a\mid s)\cdot 
\Bigl(\log \pi_T(a\mid s)-\log \pi_\theta(a\mid s)-1\Bigr)\right]
$$

> 直觉：如果 student 在某 token 上的概率比 teacher “高得不合理”，那么 $(\log\pi_\theta-\log\pi_T)$ 大，梯度会把它压下去；反之则拉上来。

---

## 5. 为什么它能复用 PPO/RL 脚手架（概念级对应）

PPO 的核心是对旧策略 $\pi_{\theta_{\text{old}}}$ 采样的动作，使用重要性比率：
$$
\rho_t(\theta)=\frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}
$$

并最大化 clipped surrogate：
$$
\mathcal{L}_{\mathrm{PPO}}(\theta)
= \mathbb{E}\left[\min\bigl(\rho_t A_t,\ \mathrm{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t\bigr)\right]
$$

在 on-policy distillation 中，只要把 advantage 设为：
$$
A_t = \log \pi_T(a_t\mid s_t)-\log \pi_{\theta_{\text{old}}}(a_t\mid s_t)
$$
就可以直接把它喂给 PPO 的更新（这对应原文“swap regularizer model from reference to teacher、reuse RL infra”的说法）。

---

## 6. 训练循环（最简算法描述）

对每个 batch 的 prompts：

1. **Rollout（学生采样）**：用 $\pi_{\theta_{\text{old}}}$ 生成序列 $x_{1:T}$（on-policy）
2. **Teacher scoring（老师打分）**：计算每个 token 的 $\log\pi_T(x_t\mid x_{<t})$
3. **Compute reverse KL**：
  $$ 
   \widehat{\mathrm{rKL}}_t=\log\pi_{\theta_{\text{old}}}(x_t\mid x_{<t})-\log\pi_T(x_t\mid x_{<t})
    $$
4. **Advantages**：
   $$
   A_t=-\widehat{\mathrm{rKL}}_t
   $$
5. **PPO update**：用 $(x_{1:T}, A_{1:T})$ 更新 $\theta$

---

## 7. 两个常见注意点（原文明确提到的边界）

- **Support（支持集）问题**：reverse KL 更像在已有 support 内做“收缩/选择”；通常需要先用 SFT/forward KL 把正确模式引入 support。  
- **Discount factor（折扣因子）**：文中经验上设 $\gamma=0$，并称 $\gamma>0$ 无明显收益。
