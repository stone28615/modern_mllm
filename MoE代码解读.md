
## 第 X 章：代码导读解析——从单机 Top-K MoE 到工程化实现

前面我们已经从原理和公式上解释了 MoE 的基本结构。这一章开始，我们把视角切到代码：不是只说“MoE 是多个专家加一个 router”，而是要真正看清楚一段代码里每一行到底在做什么、对应什么数学对象、以及为什么真实工程实现会偏向这样的写法。

这一章先看一个**单机、可读性优先，但已经接近真实实现思路**的 Top-K MoE 版本。它保留了现代 MoE 的几个关键特征：

* 输入先展平为 token 表
* 用一个 router 为每个 token 产生 expert logits
* 对 expert 维度做 softmax
* 取 top-k experts，并对 top-k 权重重归一化
* 按 expert 聚 token
* 再把各个 expert 的输出按 token 加回去

这个过程和 Hugging Face 的 Mixtral 主实现非常一致：Mixtral 也是先对 hidden states 做预处理，然后展平、路由、top-k，再把 expert 输出聚合回 token。([github.com](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py?utm_source=chatgpt.com))

### 代码示例：一个最小可读的 Top-K MoE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))

class TopKMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_ff) for _ in range(num_experts)]
        )

    def forward(self, x):
        # x: [B, S, D]
        B, S, D = x.shape
        x_flat = x.reshape(-1, D)                     # [T, D], T = B*S

        logits = self.router(x_flat).float()         # [T, E]
        probs = F.softmax(logits, dim=-1)            # [T, E]

        topk_val, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # [T, K], [T, K]
        topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)    # renorm

        y = torch.zeros_like(x_flat)

        for e, expert in enumerate(self.experts):
            pos, which = torch.where(topk_idx == e)
            if pos.numel() == 0:
                continue
            out = expert(x_flat[pos])                               # [N_e, D]
            out = out * topk_val[pos, which].unsqueeze(-1)
            y.index_add_(0, pos, out)

        return y.view(B, S, D), logits
```

---

### 1. 输入为什么要先展平

代码里的第一步是：

```python
B, S, D = x.shape
x_flat = x.reshape(-1, D)
```

如果原始输入是

$$
X \in \mathbb{R}^{B \times S \times D}
$$

那么展平后就是

$$
\tilde{X} \in \mathbb{R}^{T \times D}, \qquad T = B \cdot S
$$

这里的核心思想是：**router 的决策是按 token 独立进行的**。
对于 MoE 来说，batch 维和 sequence 维并不重要，最自然的表示其实是一张“token 表”：

* 每一行是一个 token
* 每一列是 hidden dimension

也就是说：

$$
\tilde{X}[t,:] = x_t
$$

这样写更高效的原因是，后面的 router、softmax、top-k、索引、聚合都可以直接按二维张量批处理完成，避免 Python 层的双重循环。真实实现如 Mixtral，也是先把 token 展平成二维，然后再路由。([github.com](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py?utm_source=chatgpt.com))

---

### 2. Router 在做什么

代码里的 router 是：

```python
self.router = nn.Linear(d_model, num_experts, bias=False)
```

前向时：

```python
logits = self.router(x_flat).float()
probs = F.softmax(logits, dim=-1)
```

数学上，这对应：

$$
z_t = W_r x_t \in \mathbb{R}^{N}
$$

其中：

* (x_t \in \mathbb{R}^{D}) 是第 (t) 个 token
* (W_r \in \mathbb{R}^{N \times D}) 是 router 权重
* (N) 是 expert 总数

然后对 expert 维做 softmax：

$$
p_t = \operatorname{softmax}(z_t)
$$

更具体地：

$$
p_{t,e} = \frac{\exp(z_{t,e})}{\sum_{j=1}^{N} \exp(z_{t,j})}
$$

这里显式 `.float()` 的原因很关键：router logits、softmax 和 top-k 对数值精度非常敏感。很多系统会让主体模型跑在 `bf16/fp16`，但 router 这条路径单独转到 `float32`，以减少排序和概率计算的不稳定。Megatron Core 也专门提供 router dtype 相关配置。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

从效率上看，`router(x_flat)` 是一个标准的大矩阵乘法：

$$
[T,D] \times [D,N] \to [T,N]
$$

这对 GPU 非常友好。

---

### 3. Top-K 路由到底在选什么

代码里的关键两行是：

```python
topk_val, topk_idx = torch.topk(probs, self.top_k, dim=-1)
topk_val = topk_val / topk_val.sum(dim=-1, keepdim=True)
```

这对应两个步骤。

第一步，从全部 experts 中取 top-k：

$$
S_t = \operatorname{TopK}(p_t, k)
$$

这里：

* `topk_idx[t,j]` 表示 token (t) 的第 (j) 个被选 expert
* `topk_val[t,j]` 表示该 expert 的原始 softmax 概率

第二步，对被选中的 top-k 概率重新归一化：

$$
g_{t,e} =
\frac{p_{t,e}\mathbf{1}[e\in S_t]}
{\sum_{j\in S_t}p_{t,j}}
$$

所以：

$$
\sum_{e \in S_t} g_{t,e} = 1
$$

为什么必须重归一化？因为原始 softmax 是在全部 (N) 个 experts 上归一化的，而最终真正参与计算的只有 (k) 个 experts。如果不对 top-k 权重再归一化，expert 输出的总幅值就会因为“被截掉了多少概率质量”而变化，导致不同 token 的 routed 输出尺度不稳定。

从效率上讲，`torch.topk` 是在 expert 维度上批量完成的，不需要逐 token 进入 Python 层排序，因此这是现代 MoE 实现里非常自然的一步。

---

### 4. Expert 到底在算什么

代码里的 expert 是：

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
```

数学上，对第 (e) 个 expert，可以写成：

$$
E_e(x)=W_{2,e},\mathrm{GELU}(W_{1,e}x)
$$

这只是最简单的 FFN 形式。
真实 LLM 里常见的是 gated MLP / SwiGLU 风格，也就是：

$$
E_e(x)=W_{d,e}\big(\phi(W_{g,e}x)\odot (W_{u,e}x)\big)
$$

但不管是最简 FFN 还是 gated MLP，**真正的工程关键都不是 expert 长什么样，而是 expert 如何接收它自己的 token 子批次。**

---

### 5. 为什么要“按 expert 聚 token”

这一段是整个单机实现的核心：

```python
for e, expert in enumerate(self.experts):
    pos, which = torch.where(topk_idx == e)
    if pos.numel() == 0:
        continue
    out = expert(x_flat[pos])
    out = out * topk_val[pos, which].unsqueeze(-1)
    y.index_add_(0, pos, out)
```

先看第一行：

```python
pos, which = torch.where(topk_idx == e)
```

这一步在找：

* 哪些 token 命中了 expert (e)
* expert (e) 在这些 token 的 top-k 列表中对应第几个位置

数学上，就是在构造：

$$
\mathcal{T}_e = {,t \mid e \in S_t,}
$$

然后把这些 token 的 hidden states 聚成一个子批次：

$$
X_e \in \mathbb{R}^{N_e \times D}
$$

其中 (N_e = |\mathcal{T}_e|)。

接着执行：

```python
out = expert(x_flat[pos])
```

即：

$$
Y_e = E_e(X_e) \in \mathbb{R}^{N_e \times D}
$$

然后乘对应 gate：

$$
\tilde{Y}*e[m,:] = g*{t_m,e},E_e(x_{t_m})
$$

最后用：

```python
y.index_add_(0, pos, out)
```

按 token 位置加回去：

$$
y_t^{\text{routed}} = \sum_{e\in S_t} g_{t,e}E_e(x_t)
$$

为什么这样写更高效？因为 GPU 喜欢“大矩阵乘法”，不喜欢“一个 token 一个 token 地调函数”。
按 expert 聚 token，能把原本稀疏、分散的 token-expert 对，变成每个 expert 的小 batch。这样 expert forward 就可以走 GEMM，而不是许多碎片化的小调用。

这正是 Mixtral、DeepSpeed、Megatron 在不同风格下都坚持的核心：**先按 expert 聚 token，再做本地批量计算。**([github.com](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py?utm_source=chatgpt.com))

---

### 6. 单机版本的总公式

把上面所有步骤合起来，这段代码实现的 routed path 就是：

$$
z_t = W_r x_t
$$

$$
p_t = \operatorname{softmax}(z_t)
$$

$$
S_t = \operatorname{TopK}(p_t, k)
$$

$$
g_{t,e} =
\frac{p_{t,e}\mathbf{1}[e \in S_t]}
{\sum_{j \in S_t} p_{t,j}}
$$

$$
y_t^{\text{routed}} = \sum_{e \in S_t} g_{t,e} E_e(x_t)
$$

如果再加入 shared experts，那么整层写成：

$$
y_t
===

\underbrace{\sum_{i=1}^{K_s}E_i^{(s)}(x_t)}*{\text{shared experts}}
+
\underbrace{\sum*{e \in S_t} g_{t,e} E_e^{(r)}(x_t)}_{\text{routed experts}}
$$

这也是 DeepSeekMoE 的核心结构表达。([aclanthology.org](https://aclanthology.org/2024.acl-long.70.pdf?utm_source=chatgpt.com))

---

### 7. 为什么真实工程实现会更复杂

上面的单机代码足够讲清楚 MoE 的核心思想，但真实系统里还会多出三层复杂性。

第一层是 **capacity 与 token dropping**。
真实训练里，每个 expert 往往有一个容量上限：

$$
C = \max\left(C_{\min}, \left\lceil \alpha \frac{kT}{N} \right\rceil\right)
$$

如果某个 expert 收到的 routed assignments 超过 (C)，超出的部分就会被 drop，或者走其他 fallback 路径。DeepSpeed 正是围绕 `capacity_factor` 和 `drop_tokens` 组织路由逻辑的。([github.com](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py?utm_source=chatgpt.com))

第二层是 **分布式 token dispatch / combine**。
一旦 experts 分布在不同 GPU 上，token 不能直接在本地调用 expert，而是必须先发到 expert 所在 rank，再在本地算完后发回来。Megatron Core 文档明确把这个过程拆成 token dispatch、local expert compute 和 token combine。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

第三层是 **kernel 级优化**。
真实系统不会满足于“每个 expert 一个 Python 循环”。Megatron Core 把 GroupedGEMM、fused dispatch/combine、router fusion 都作为重要性能优化手段。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

---

## 第 X+1 章：分布式代码导读解析——伪 All-to-All 路由、派发与聚合

上一章讲清楚了**单机版 routed MoE** 的核心逻辑：router 为每个 token 选 top-k experts，按 expert 聚 token，执行 expert 前向，再把结果加回 token。
这一章进一步看**分布式版 routed path**：如果 experts 分布在多张 GPU 上，那么 token 如何被派发到 owner rank，本地 expert 如何处理，再怎样把结果返回并恢复到原 token 位置。

这正对应 Megatron Core 文档中的三步：

1. **token dispatch**
2. **local expert compute**
3. **token combine**

而在我们的教学工程版代码里，这三步分别由下面这些变量承载：

* `send_buffers`
* `recv_buffers`
* `local_slots`
* `return_buffers`
* `routed_out`

---

### 代码示例：伪 All-to-All 的核心 routed path

```python
# [D01]
send_buffers = [[[] for _ in range(self.ep_size)] for _ in range(self.ep_size)]
# [D02]
dropped_routes = 0

# [D03]
for t in range(T):
    # [D04]
    src_rank = self._source_rank(t, T)
    # [D05]
    hidden_t = x_flat[t]
    # [D06]
    for j in range(self.top_k):
        # [D07]
        global_e = int(topk_idx[t, j].item())
        # [D08]
        dst_rank = global_e // self.num_local_experts
        # [D09]
        local_e = global_e % self.num_local_experts
        # [D10]
        gate = topk_val[t, j]
        # [D11]
        send_buffers[src_rank][dst_rank].append((t, local_e, gate, hidden_t))

# [D12]
recv_buffers = [[] for _ in range(self.ep_size)]
# [D13]
for src_rank in range(self.ep_size):
    # [D14]
    for dst_rank in range(self.ep_size):
        # [D15]
        recv_buffers[dst_rank].extend(send_buffers[src_rank][dst_rank])

# [D16]
return_buffers = [[[] for _ in range(self.ep_size)] for _ in range(self.ep_size)]
# [D17]
local_load = torch.zeros(
    self.ep_size, self.num_local_experts, dtype=torch.long, device=x.device
)

# [D18]
for dst_rank in range(self.ep_size):
    # [D19]
    local_slots = [[] for _ in range(self.num_local_experts)]

    # [D20]
    for token_idx, local_e, gate, hidden in recv_buffers[dst_rank]:
        # [D21]
        if len(local_slots[local_e]) < capacity:
            # [D22]
            local_slots[local_e].append((token_idx, gate, hidden))
        # [D23]
        else:
            # [D24]
            dropped_routes += 1

    # [D25]
    for local_e in range(self.num_local_experts):
        # [D26]
        slot_records = local_slots[local_e]
        # [D27]
        local_load[dst_rank, local_e] = len(slot_records)
        # [D28]
        if len(slot_records) == 0:
            # [D29]
            continue

        # [D30]
        local_in = torch.stack([rec[2] for rec in slot_records], dim=0)
        # [D31]
        local_out = self.rank_experts[dst_rank][local_e](local_in)

        # [D32]
        for row, (token_idx, gate, _) in enumerate(slot_records):
            # [D33]
            src_rank = self._source_rank(token_idx, T)
            # [D34]
            weighted = gate.to(local_out.dtype) * local_out[row]
            # [D35]
            return_buffers[dst_rank][src_rank].append((token_idx, weighted))

# [D36]
routed_out = torch.zeros_like(x_flat)
# [D37]
for src_rank in range(self.ep_size):
    # [D38]
    for dst_rank in range(self.ep_size):
        # [D39]
        for token_idx, weighted in return_buffers[dst_rank][src_rank]:
            # [D40]
            routed_out[token_idx] += weighted
```

---

### 1. Expert Parallel 的映射关系

设共有 (N) 个 routed experts，(P) 个 expert-parallel ranks，则每个 rank 持有：

$$
N_{\text{local}} = \frac{N}{P}
$$

个 local experts。

于是一个全局 expert 编号 (e) 会被映射到：

$$
\operatorname{dst}(e) = \left\lfloor \frac{e}{N_{\text{local}}} \right\rfloor
$$

$$
\operatorname{local}(e) = e \bmod N_{\text{local}}
$$

这就是代码里：

```python
dst_rank = global_e // self.num_local_experts
local_e = global_e % self.num_local_experts
```

的数学来源。

也就是说，router 选出来的是全局 expert (e)，但真正执行时，系统必须先知道：

* 这个 expert 属于哪个 owner rank
* 在 owner rank 内部，它是第几个 local expert

---

### 2. `send_buffers`：把 token-expert 路由记录打包成通信矩阵

首先看：

```python
send_buffers = [[[] for _ in range(self.ep_size)] for _ in range(self.ep_size)]
```

它创建的是一个二维通信结构：

$$
\text{send buffers}[s][d]
$$

表示 **从 source rank (s) 发往 destination rank (d)** 的所有 routed assignments。

---

#### 2.1 对每个 token、对每个 top-k expert 展开

```python
for t in range(T):
    src_rank = self._source_rank(t, T)
    hidden_t = x_flat[t]
    for j in range(self.top_k):
        global_e = int(topk_idx[t, j].item())
        dst_rank = global_e // self.num_local_experts
        local_e = global_e % self.num_local_experts
        gate = topk_val[t, j]
        send_buffers[src_rank][dst_rank].append((t, local_e, gate, hidden_t))
```

对每个 token (t)，router 已经给出了它的 top-k experts：

$$
e_{t,j} = \text{topk\_idx}[t,j]
$$

以及对应 gate：

$$
g_{t,j} = \text{topk\_val}[t,j]
$$

于是这里构造的每一条 routed record 都可以写成：

$$
r_{t,j} = \bigl(t,\ \operatorname{local}(e_{t,j}),\ g_{t,j},\ x_t\bigr)
$$

并被插入：

$$
\text{send\_buffers}[\operatorname{src}(t)][\operatorname{dst}(e_{t,j})]
$$

这一步的本质是：**把“token 路由到 expert”改写成“source rank 向 destination rank 发送一条记录”。**

为什么这样实现更高效？因为它把大量细粒度的 routed decisions 合并到了“按目标 rank”组织的缓冲区里。真实 all-to-all 也是如此：不是逐 token 发消息，而是按目标 rank 合并成更大的消息块。Megatron Core 的 dispatcher 文档也明确围绕 token dispatch 的批量组织展开。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

---

### 3. `recv_buffers`：first all-to-all 后，按 owner rank 收齐记录

接着看：

```python
recv_buffers = [[] for _ in range(self.ep_size)]
for src_rank in range(self.ep_size):
    for dst_rank in range(self.ep_size):
        recv_buffers[dst_rank].extend(send_buffers[src_rank][dst_rank])
```

这里的数学语义是：

$$
\text{recv\_buffers}[d]
=

\operatorname{concat}_{s=0}^{P-1}\text{send\_buffers}[s][d]
$$

也就是说，owner rank (d) 现在拿到了所有发给它的 token 路由记录。

这正是 first all-to-all 的核心语义：

* 路由前，token 在 source rank 上
* 路由后，记录被发到 owner rank 上
* owner rank 从这一刻开始只关心“发给自己的那些 routed assignments”

为什么这样更高效？因为**通信被集中在一次显式的收集阶段完成**，后面的 expert 计算都能在本地进行，而不是边通信边算。

---

### 4. `local_slots`：按 local expert 再分桶，并做 capacity 截断

到了 owner rank 之后，还不能直接把 `recv_buffers[dst_rank]` 整体喂给某个 expert，因为不同记录对应的是不同 local experts。于是代码接着做：

```python
for dst_rank in range(self.ep_size):
    local_slots = [[] for _ in range(self.num_local_experts)]

    for token_idx, local_e, gate, hidden in recv_buffers[dst_rank]:
        if len(local_slots[local_e]) < capacity:
            local_slots[local_e].append((token_idx, gate, hidden))
        else:
            dropped_routes += 1
```

这里，对于固定的 `dst_rank = d`，为每个 local expert (\ell) 构造一个槽位集合：

$$
\mathcal{S}_{d,\ell}
$$

如果某条记录发给 local expert (\ell)，且当前：

$$
|\mathcal{S}_{d,\ell}| < C
$$

其中 (C) 是容量上限，那么就把它加入：

$$
\mathcal{S}*{d,\ell} \leftarrow \mathcal{S}*{d,\ell} \cup {(t,g,x_t)}
$$

否则，这条 routed assignment 被丢弃，并累加：

$$
n_{\text{drop}} \leftarrow n_{\text{drop}} + 1
$$

为什么工程里总是有 capacity？因为 routed MoE 天生可能产生负载峰值。如果不做限制，某个 expert 在某个 step 可能收到大量 token，而另一些 experts 几乎空闲。capacity 的作用，就是把这个峰值强行截断到系统可调度范围内。DeepSpeed 的 `capacity_factor` 和 `drop_tokens` 就是围绕这一点设计的。([github.com](https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py?utm_source=chatgpt.com))

为什么这样实现更高效？因为 `local_slots` 这一步同时完成了两件事：

1. 把 routed records 再按 local expert 聚类
2. 用容量约束防止本地 batch 失控

这样后面的 local expert forward 才会形成形状可控的矩阵乘法。

---

### 5. `local_in / local_out`：owner rank 上真正执行 local expert 前向

接下来是：

```python
for local_e in range(self.num_local_experts):
    slot_records = local_slots[local_e]
    local_load[dst_rank, local_e] = len(slot_records)
    if len(slot_records) == 0:
        continue

    local_in = torch.stack([rec[2] for rec in slot_records], dim=0)
    local_out = self.rank_experts[dst_rank][local_e](local_in)
```

如果固定当前 owner rank (d) 和 local expert (\ell)，设其保留下来的 token 数量为：

$$
L_{d,\ell} = |\mathcal{S}_{d,\ell}|
$$

那么 `torch.stack(...)` 实际构造的是：

$$
X_{d,\ell}^{\text{local}} \in \mathbb{R}^{L_{d,\ell}\times D}
$$

然后 expert 前向得到：

$$
Y_{d,\ell}^{\text{local}}
=

E_{d,\ell}^{(r)}(X_{d,\ell}^{\text{local}})
\in \mathbb{R}^{L_{d,\ell}\times D}
$$

这里的关键工程思想是：**先按 expert 聚 token，再 batch 化算 expert**。
这和单机版 `x_flat[pos]` 的思想一模一样，只不过这里是先经历了一次跨 rank dispatch，然后在 owner rank 本地再按 local expert 做聚类。

为什么这样更高效？因为 expert forward 终于变成了一块真正的本地 GEMM，而不是许多零散的远程 token 计算。Megatron Core 文档里进一步把这种批量化 local expert 计算优化成 GroupedGEMM。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

---

### 6. `return_buffers`：把 local expert 的结果逐条拆开、乘 gate、再发回 source rank

然后是：

```python
for row, (token_idx, gate, _) in enumerate(slot_records):
    src_rank = self._source_rank(token_idx, T)
    weighted = gate.to(local_out.dtype) * local_out[row]
    return_buffers[dst_rank][src_rank].append((token_idx, weighted))
```

这里每一行代表 local batch 里的一个 token。设第 `row` 行对应 token (t)，则：

* `local_out[row]` 是 expert 对 token 的输出

$$
E_e^{(r)}(x_t)
$$

* 再乘上 gate：

$$
\tilde{y}*{t,e} = g*{t,e},E_e^{(r)}(x_t)
$$

然后写入：

$$
\text{return\_buffers}[\operatorname{dst}(e)][\operatorname{src}(t)]
$$

这里为什么要逐条拆开？因为 `local_out` 当前的排列顺序是“按 local expert 批处理的顺序”，而不是原始 token 顺序。想要把结果发回去并最终聚合到 token 位置，就必须重新打包成 `(token_idx, weighted)` 的形式。

为什么先乘 gate 再发回更合理？因为这样回传的是**最终对 token 有意义的专家贡献**，后面 combine 阶段只需要做加法，不需要再额外查 gate。

---

### 7. `routed_out`：second all-to-all 之后，把所有专家贡献加回 token

最后一段：

```python
routed_out = torch.zeros_like(x_flat)
for src_rank in range(self.ep_size):
    for dst_rank in range(self.ep_size):
        for token_idx, weighted in return_buffers[dst_rank][src_rank]:
            routed_out[token_idx] += weighted
```

这里初始化：

$$
Y^{\text{routed}} = 0 \in \mathbb{R}^{T\times D}
$$

然后把所有 surviving routed experts 的贡献加回原 token 位置：

$$
y_t^{\text{routed}}
=

\sum_{e \in S_t \cap \text{kept}} g_{t,e}E_e^{(r)}(x_t)
$$

这正是 MoE 路由项的标准数学定义，只不过在工程上它现在是经过了：

1. `send_buffers` 打包
2. `recv_buffers` 汇总
3. `local_slots` 分桶
4. `local_in/local_out` 本地计算
5. `return_buffers` 回传

之后，最终在 `routed_out[token_idx] += weighted` 这里还原出来。

为什么这一段很重要？因为它说明：**分布式实现虽然引入了大量通信和重排逻辑，但最终的模型输出公式并没有变。**

---

### 8. 分布式 routed path 的总公式

把这一整章压缩成一条数学式，仍然只是：

$$
y_t^{\text{routed}}
=

\sum_{e \in S_t \cap \text{kept}} g_{t,e}E_e^{(r)}(x_t)
$$

其中：

* (S_t) 是 router 选出的 top-k experts
* `kept` 表示那些没有因为 capacity overflow 被 drop 的 routed assignments
* `send_buffers / recv_buffers / local_slots / return_buffers / routed_out` 只是在工程上实现这条式子的执行路径

如果再加上 shared experts，则完整一层仍然是：

$$
y_t
=

\underbrace{\sum_{i=1}^{K_s}E_i^{(s)}(x_t)}_{\text{shared experts}}
+
\underbrace{\sum_{e \in S_t \cap \text{kept}} g_{t,e}E_e^{(r)}(x_t)}_{\text{routed experts}}
$$

这就是 DeepSeekMoE 结构的核心。([aclanthology.org](https://aclanthology.org/2024.acl-long.70.pdf?utm_source=chatgpt.com))

---

### 9. 为什么这种分布式写法是高效的

最后把这整章的效率逻辑压缩一下。

第一，`send_buffers` 把逐 token 的路由结果改写成“按目标 rank 组织的消息”，减少了碎片化通信。
第二，`recv_buffers -> local_slots` 把跨 rank 到达的 routed records 再次按 local expert 聚类，从而把本地 expert forward 变成批量 GEMM。
第三，capacity 让 expert 负载可控，避免某些 experts 在单步被打爆。
第四，`return_buffers -> routed_out` 把本地 expert 的输出恢复回 token 视角，只需要做加权回传和 scatter-add 聚合。

换句话说，这种实现方式把原本“token → expert”的逻辑，分解成了两次重排：

1. **token 视角 → owner rank / local expert 视角**
2. **owner rank / local expert 视角 → token 视角**

MoE 分布式实现的本质，就是这两次重排之间夹着一段高效的 local expert 计算。Megatron Core 的 dispatcher 和 combine，本质上也正是围绕这两次重排来设计。([docs.nvidia.com](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html?utm_source=chatgpt.com))

[[MoE]]