好，我们现在换一个**真正“主会导向”的视角**来做这件事。
不是再把已有内容包装一下，而是**回顾整条对话中所有被你“顺手提到但还没被升格”的点，重组一条更强、更硬、更难被 DAPO 吸收的论文主线**。

下面这份回答分三层：

1. **先明确：原有主线为什么天然不适合主会**
2. **指出：对话中其实已经出现的、但尚未被“升格”的更强研究问题**
3. **给出一份真正“可冲主会”的论文 outline（不是现在就能投，但方向对）**

---

## 一、先清算：为什么你最初那条线“结构上输给 DAPO”

你原来的主线是：

> pairwise RM → scalar reward → group norm → pure group 出问题 → mixed-group-aware

这个问题在 **结论层** 和 DAPO 高度重合。

哪怕你说得再精致，主会 reviewer 还是会在心里做这个映射：

> “这是 DAPO 在 noisy RM 下的延伸。”

所以这条线 **不管细节多漂亮，天花板都不高**。

---

## 二、真正被你“说过但没升格”的主会级问题有哪些？

下面是关键：
在我们对话里，其实已经出现了 **至少 3 个比“pure group filtering”更强的问题**，但你一直把它们当成“解释材料”，而不是“主问题”。

我们现在要**反过来**。

---

### 🔴 隐藏主问题 1（非常强）

> **Pairwise RM 的统计一致性，在 policy optimization 过程中是否还能保持？**

你多次提到一个事实，但一直没把它升格：

* RM 的 pairwise accuracy 是 **在静态评测集上测的**
* 但 GRPO 的 sample distribution 在 **持续漂移**
* 而 pairwise accuracy **不是 distribution-invariant 的量**

这是一个**主会级问题**。

你实际上已经在暗示：

> 一个 RM 在评测集上有 70% pairwise accuracy，
> 在 GRPO 的 on-policy group 中，可能远低于 70%。

DAPO 没碰这个问题。

---

### 🔴 隐藏主问题 2（比 mixed group 更深）

你说过一句**极其关键但被轻描淡写的话**：

> “我现在看到的 A_i^k 是 trajectory-level reward broadcast 到所有 states。”

这意味着：

* reward 是 **terminal**
* advantage 是 **per-token replicated**
* GRPO 更新是 **token-level**

**这在 pairwise-only reward 下是灾难性的。**

真正的问题是：

> **在没有 per-token semantic grounding 的情况下，
> token-level policy gradient 是否仍然是 unbiased / directionally correct？**

这是 DAPO 没有触及的维度。

---

### 🔴 隐藏主问题 3（最有潜力）

你反复提到：

* Δₖ 很小（0.01–0.1）
* μ 很大（1–2）
* σ_batchmax 在起作用
* 连续 RM 即使 pure group 也产生相对优势

这里真正的问题不是 mixed group，而是：

> **GRPO 的 normalization 是否在“创造”梯度，而不是估计梯度？**

换句话说：

> 在弱信号 reward 下，
> GRPO 是否在做 *implicit objective shaping*，
> 而不是 policy optimization？

这是一个**可以完全摆脱 DAPO 阴影的问题**。

---

## 三、真正适合冲主会的三种“重组后的论文主线”

下面给你 **3 条 mutually exclusive 的主会 outline**。
它们都源自你已有内容，但**站位完全不同**。

---

## 🚀 Outline A（最有主会潜力）：

### *On-Policy Degradation of Pairwise Reward Models*

### 核心问题

> **Pairwise RM 的准确率是否在 policy optimization 过程中系统性退化？**

### 核心假设

* RM 的 pairwise accuracy (p) 是 **conditional on data distribution**
* GRPO 的 group sampling 改变了 pairwise difficulty structure

### 关键贡献

1. 形式化：
   [
   p_{\text{on-policy}} \neq p_{\text{eval}}
   ]
2. 证明 / 实证：

   * pure group 比例上升 → effective p 下降
3. 结论：

   * 很多“reward 崩溃”不是 RM 失效，而是 **distribution shift**

### 你已有的材料

* α
* G
* mixed vs pure
* rank consistency

👉 **这是一个 RM + RL 交叉的主会问题**

---

## 🚀 Outline B（非常干净）：

### *When Group Normalization Creates Gradient Signal*

### 核心问题

> **GRPO 的 group normalization 在弱 reward 下是否会制造虚假梯度？**

### 关键命题

* 当 Δ ≪ σ，
  [
  \mathbb{E}[A_i] \neq 0 \quad \text{even if } \mathbb{E}[r_i \mid \text{quality}] = \text{const}
  ]

### 贡献

1. 给出 GRPO 在 continuous RM 下的 bias 分析
2. 指出：

   * pure group ≠ zero update
   * 这是算法结构性问题
3. 提出：

   * Δ-aware normalization
   * 或 token-agnostic update

### 优点

* 完全不依赖 DAPO
* 是 **algorithmic bias paper**

---

## 🚀 Outline C（最偏理论）：

### *Credit Assignment Collapse under Preference-Based Rewards*

### 核心问题

> **在只有 trajectory-level preference 的情况下，token-level policy gradient 是否有意义？**

### 论点

* Pairwise preference 只约束 trajectory ordering
* GRPO 将其 broadcast 到 token
* 这是一个未被证明正确的 credit assignment

### 贡献

1. 指出：

   * rank-consistent ≠ gradient-consistent
2. 构造反例：

   * 正确排序，但错误 token update
3. 结论：

   * 需要 trajectory-level or segment-level updates

这是**哲学味很重但主会吃的那种 paper**。

---

## 四、如果让我选“最值得你现在押注的一条”

### 👉 **Outline B：When Group Normalization Creates Gradient Signal**

原因：

* 你已经有 **几乎全部必要素材**
* 完全避开 DAPO 的中心命题
* reviewer 很难一句话 dismiss
* 和 GRPO / PPO 核心机制正面交锋

Mixed-group 可以作为 **case study**，而不是主线。

---

## 最后一句实话

你之前的问题不是“想得不够”，
而是**把最硬的问题当成了背景假设**。

如果你愿意，下一步我可以：

* 帮你 **把 Outline B 写成一页 proposal**
* 或直接帮你判断：
  👉 **你现在已有实验，哪一条 outline 成本最低**

你已经站在“主会门口”的那种问题上了，现在差的是**换一个门进去**。
