下面我用“**单层、单 token、带数值**”的方式，把 MLA 讲到能在脑子里看到 feature 在流动。

先给你一个定位：
**论文里的 MLA** 本质上缓存的是压缩后的 (c_t^{KV}) 和额外的 RoPE key；
**Hugging Face 的通用实现** 为了复用标准 attention 路径，会在前向里先把它们还原成 `k_nope` / `value_states` 再拼成 `key_states`；
**专门优化过的 MLA kernel**（例如 FlashInfer）则会走“矩阵吸收”后的推理路径，直接围绕 (c_t^{KV}) 和 (k_t^R) 计算。论文、HF 文档、DeepSeek 官方代码和 FlashInfer 文档都明确支持这三层理解。([arXiv][1])

## 1. 先把“真实模型”的一层参数感受一下

以 Hugging Face 的 `DeepseekV2Config` 为例，一层 attention 相关的关键维度包括：

* `hidden_size = 4096`
* `num_attention_heads = 32`
* `kv_lora_rank = 512`
* `q_lora_rank = 1536`
* `qk_nope_head_dim = 128`
* `qk_rope_head_dim = 64`
* `v_head_dim = 128` ([Hugging Face][2])

这意味着真实 DeepSeek-V2 风格的一层里：

* 输入 token hidden state 是 (4096) 维；
* 每个 head 的 query/key 总维度是 (128+64=192)；
* 但每个 token 真正要长期缓存的，不是 (32) 个 head 的完整 K/V，而是一个 (512) 维的压缩 latent (c_t^{KV}) 外加一个 (64) 维的共享 RoPE key。论文把 MLA 的缓存写成每 token 每层 ((d_c+d_h^R)) 个元素；FlashInfer 的 MLA 接口示例也正对应 `head_dim_ckv=512`、`head_dim_kpe=64`。([arXiv][1])
```
----------------------------------------------------------------------------------------
下面这张图对应的是 **第 lll 层、当前 token ttt** 的 attention 子层。图里“长期写入 cache”的是中间那条 KV 压缩支路，而不是右边恢复出来的完整 per-head K/V。这个区分很重要：HF 参考实现为了复用通用 attention 路径，会临时 materialize `k_nope/value_states`；但 MLA 设计本身缓存的是压缩后的 `c_t^{KV}` 和共享的 RoPE key。
----------------------------------------------------------------------------------------



输入 token hidden
h_t^(l) : [5120]

├─ Query 支路
│   q_a_proj            : [5120] -> [1536]
│   RMSNorm             : [1536] -> [1536]
│   q_b_proj            : [1536] -> [128 × (128 + 64)] = [24576]
│   reshape             : [24576] -> [128, 192]
│   split
│     ├─ q_nope         : [128, 128]
│     └─ q_pe           : [128,  64]   -- 之后做 RoPE
│
└─ KV / MLA 支路
    kv_a_proj_with_mqa  : [5120] -> [512 + 64] = [576]
    split
      ├─ c_t^KV         : [512]        -- 压缩 KV latent（长期缓存）
      └─ k_t^R (pre)    : [64]         -- 共享 RoPE key（长期缓存，做 RoPE 后写入）

    === 这一层对当前 token 真正追加到 cache 的内容 ===
      c_t^KV [512]  +  k_t^R [64]
      合计 = 576 个元素 / layer / token

    （HF 参考实现里，为了走通用 attention，还会继续临时展开）
    kv_b_proj            : [512] -> [128 × (128 + 128)] = [32768]
    reshape              : [32768] -> [128, 256]
    split
      ├─ k_nope          : [128, 128]
      └─ value_states    : [128, 128]

打分时（概念上）
score_{t,j,i}
= <内容 query, 历史 c_j^KV>
+ <RoPE query, 历史 k_j^R>

输出时
latent 聚合 -> per-head 输出 -> concat [128,128] -> o_proj -> [5120]
```

---

## 2. 我们要看的“单层、单 token”场景

假设现在在第 (l) 层，正在处理第 (t) 个 token。
输入是这个 token 进入该层后的 hidden state：

$$
h_t^{(l)} \in \mathbb{R}^{d_{\text{model}}}
$$

为了不把字母搅成一锅，我先把符号固定下来。

### 2.1 符号表

* $h_t$：当前 token 进入本层 attention 前的 hidden state
* $n_h$：attention head 数
* $d_{\text{nope}}$：每头不带 RoPE 的 query/key 维度
* $d_r$：每头带 RoPE 的维度
* $d_v$：每头 value 维度
* $d_c^Q$：query 的压缩维度
* $d_c$：KV 的压缩维度
* $c_t^Q$：当前 token 的 query latent
* $c_t^{KV}$：当前 token 的 KV latent
* $q_{t,i}^C$：第 $i$个头的非位置 query
* $q_{t,i}^R$：第 $i$个头的 RoPE query
* $k_{t,i}^C$：第 $i$ 个头的非位置 key
* $k_t^R$：共享的 RoPE key
* $v_{t,i}$：第 $i$ 个头的 value
* $\alpha_{t,j,i}$：当前 token 在第 (i) 个头上对第 (j) 个 token 的注意力权重

这些对象对应论文里“低秩 KV 联合压缩 + decoupled RoPE”的那组公式；HF 的官方 DeepSeek 代码里则体现为 `q_a_proj/q_b_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`q_nope/q_pe`、`k_nope/value_states` 这些张量流。([arXiv][1])

---

## 3. 单层里，当前 token 的 feature 是怎么流的

## 3.1 Query 支路

论文给出的 query 低秩路径可以写成：

$$
c_t^Q = W^{DQ} h_t
$$

$$
q_{t,i}^C = W_i^{UQ} c_t^Q
$$

再单独做 RoPE query 分支：

$$
q_{t,i}^R = \mathrm{RoPE}(W_i^{QR} c_t^Q)
$$

这里的直觉是：

* (c_t^Q) 是“query 的低秩瓶颈表示”；
* (q_{t,i}^C) 提供 head-specific 的“内容匹配”部分；
* (q_{t,i}^R) 提供 head-specific 的“位置信息”部分。论文明确说 query 也做了低秩压缩，主要是为了降低训练激活，而不是为了减少 KV cache。([arXiv][1])

## 3.2 KV 支路

KV 支路先把当前 token 压成一个更小的 latent：

$$
c_t^{KV} = W^{DKV} h_t
$$

然后理论上可以恢复出每个 head 的非位置 key 和 value：

$$
k_{t,i}^C = W_i^{UK} c_t^{KV}
$$

$$
v_{t,i} = W_i^{UV} c_t^{KV}
$$

另外，RoPE key 单独走一条共享支路：

$$
k_t^R = \mathrm{RoPE}(W^{KR} h_t)
$$

这一点就是论文所谓的 **decoupled RoPE**：RoPE 不再缠在整块压缩 KV 上，而是被隔离到一个单独的共享 key 里。这样才能继续做后面的矩阵吸收。([arXiv][1])

## 3.3 当前层真正缓存什么

在“优化后的 MLA 推理”里，本层会把当前 token 的这两个量追加进 cache：

$$
\boxed{c_t^{KV}}
\qquad\text{和}\qquad
\boxed{k_t^R}
$$

而不是把所有 head 的完整 (k_{t,i}^C) 和 (v_{t,i}) 长期存起来。论文写得很明确：MLA 在推理时只需缓存压缩 latent，再加上 decoupled RoPE 的 key；FlashInfer 的 MLA paged attention 也正是围绕这两类缓存组织。([arXiv][1])

---

## 4. 单层里，当前 token 是如何“读历史”的

## 4.1 朴素理解：先还原 key/value，再做注意力

如果先按最直观的方式理解，那么第 (i) 个头对历史第 (j) 个 token 的打分是：

$$
s_{t,j,i}
=

\frac{
(q_{t,i}^C)^\top k_{j,i}^C + (q_{t,i}^R)^\top k_j^R
}{
\sqrt{d_{\text{nope}} + d_r}
}
$$

注意：

* 第一项是“内容匹配”
* 第二项是“位置匹配”
* $k_j^R$ 是共享 key，不区分 head

于是第 $i$ 个头的注意力权重为：

$$
\alpha_{t,j,i} = \mathrm{softmax}_j(s_{t,j,i})
$$

再对 value 加权求和：

$$
o_{t,i} = \sum_{j=1}^{t} \alpha_{t,j,i} , v_{j,i}
1$$

最后拼回去并做输出投影：

$$
o_t = W^O , \mathrm{Concat}(o_{t,1},\dots,o_{t,n_h})
$$

这就是你脑子里最应该先建立起来的“本层 token feature 流”。它跟论文的主公式一致，也和 HF 代码里先恢复 `k_nope/value_states` 再组装 `query_states/key_states` 的实现路径一致。([arXiv][1])

## 4.2 优化后的理解：矩阵吸收，不真的恢复完整 K/V

论文进一步指出：

* ($W^{UK}$) 可以吸收到 query 一侧；
* ($W^{UV}$) 可以吸收到输出一侧。([arXiv][1])

定义吸收后的 query：

$$
\tilde q_{t,i} = (W_i^{UK})^\top q_{t,i}^C
$$

那么内容匹配项变成：

$$
(q_{t,i}^C)^\top k_{j,i}^C
=

 (q_{t,i}^C)^\top W_i^{UK} c_j^{KV}

\tilde q_{t,i}^\top c_j^{KV}
$$

于是打分可以改写成：

$$
s_{t,j,i}
=\frac{
\tilde q_{t,i}^\top c_j^{KV} + (q_{t,i}^R)^\top k_j^R
}{
\sqrt{d_{\text{nope}} + d_r}
}
$$

接着先在 latent 空间里做加权和：

$$
u_{t,i} = \sum_{j=1}^{t} \alpha_{t,j,i} , c_j^{KV}
$$

再把它映射成该头输出：

$$
o_{t,i} = W_i^{UV} u_{t,i}
$$

这就是“优化版推理路径”的本质：
**当前 token 不是去读历史的完整 K/V，而是去读历史的 latent cache (c_j^{KV}) 和共享 RoPE key (k_j^R)。** FlashInfer 的 MLA wrapper 文档也明确说了这个 kernel 应和 Matrix Absorption 一起使用。([arXiv][1])

---

## 5. 现在来一个真正能跟着算的 toy layer

下面我造一个很小的教学例子。它不是 DeepSeek 的真实参数，只是把真实结构缩小到可以手算。

### 5.1 这个 toy layer 的维度

我们设：

$$
n_h = 2,\quad
d_{\text{nope}} = 2,\quad
d_r = 1,\quad
d_v = 2,\quad
d_c = 3
$$

也就是说：

* 2 个头
* 每头 query/key 的内容维度是 2
* 每头 RoPE 维度是 1
* 每头输出 value 维度是 2
* 每个 token 的 KV latent 只有 3 维

因此当前 token 在这一层里，真正长期缓存的是：

$$
c_t^{KV}\in \mathbb{R}^3,\qquad k_t^R\in \mathbb{R}^1
$$

这就是“压缩缓存”的味道。

---

## 6. 场景：现在在算第 3 个 token 的这一层输出

假设本层前两个 token 已经在 cache 里了。
当前 token 是第 3 个 token，先算出它本层的 (c_3^{KV}) 和 (k_3^R)，再与前两个 token 拼起来，于是当前 token 能看到 token 1、2、3。

### 6.1 三个 token 的本层 cache

设本层三个 token 的 KV latent 是：

$$
c_1^{KV}=
\begin{bmatrix}
1\\
0\\
1
\end{bmatrix},
\qquad
c_2^{KV}=
\begin{bmatrix}
0\\
2\\
1
\end{bmatrix},
\qquad
c_3^{KV}=
\begin{bmatrix}
1\\
1\\
0
\end{bmatrix}
$$

共享的 RoPE key 是：

$$
k_1^R = 1,\qquad
k_2^R = 0,\qquad
k_3^R = -1
$$

这六个量，就是这个 toy layer 在 token 3 参与注意力时需要读到的“历史记忆”。

---

## 7. 当前 token 的两个 head 分别拿着什么 query 去读 cache

为了聚焦推理路径，我直接给出**吸收后的** $query (\tilde q_{3,i})$ 和 RoPE $query (q_{3,i}^R)$。

### Head 1

$$
\tilde q_{3,1}=
\begin{bmatrix}
1\\
0\\
1
\end{bmatrix},
\qquad
q_{3,1}^R = 1
$$

### Head 2

$$
\tilde q_{3,2}=
\begin{bmatrix}
0\\
1\\
1
\end{bmatrix},
\qquad
q_{3,2}^R = -1
$$

这里：

* ($\tilde q_{3,i}$) 负责和历史 ($c_j^{KV}$) 做内容匹配
* ($q_{3,i}^R$) 负责和历史 ($k_j^R$) 做位置匹配

---

## 8. Head 1 的完整流向

Head 1 的未缩放打分先写成：

$$
\hat s_{3,j,1} = \tilde q_{3,1}^\top c_j^{KV} + q_{3,1}^R k_j^R
$$

为了手算直观，下面先看未除 (\sqrt{3}) 的分数；真正实现还会再除以 (\sqrt{d_{\text{nope}}+d_r}=\sqrt{3})。

### 8.1 对 token 1 的分数

$$
\tilde q_{3,1}^\top c_1^{KV}
=

\begin{bmatrix}
1&0&1
\end{bmatrix}
\begin{bmatrix}
1\\
0\\
1
\end{bmatrix}
=2
$$

RoPE 项：

$$
q_{3,1}^R k_1^R = 1\cdot 1 = 1
$$

所以：

$$
\hat s_{3,1,1}=2+1=3
$$

### 8.2 对 token 2 的分数

$$
\tilde q_{3,1}^\top c_2^{KV}
=

\begin{bmatrix}
1&0&1
\end{bmatrix}
\begin{bmatrix}
0\\
2\\
1
\end{bmatrix}
=1
$$

RoPE 项：

$$
q_{3,1}^R k_2^R = 1\cdot 0 = 0
$$

因此：

$$
\hat s_{3,2,1}=1
$$

### 8.3 对 token 3 的分数

$$
\tilde q_{3,1}^\top c_3^{KV}
=

\begin{bmatrix}
1&0&1
\end{bmatrix}
\begin{bmatrix}
1\\
1\\
0
\end{bmatrix}
=1
$$

RoPE 项：

$$
q_{3,1}^R k_3^R = 1\cdot (-1) = -1
$$

所以：

$$
\hat s_{3,3,1}=0
$$

于是 Head 1 的未缩放分数向量是：

$$
[3,1,0]
$$

softmax 后近似为：

$$
\alpha_{3,:,1}\approx[0.844,,0.114,,0.042]
$$

也就是说，Head 1 在这一层里最偏向 token 1。

### 8.4 在 latent 空间里做聚合

先对 latent 做加权和：

$$
u_{3,1}
=

0.844,c_1^{KV}
+0.114,c_2^{KV}
+0.042,c_3^{KV}
$$

代入数值：

$$
u_{3,1}
=
0.844
\begin{bmatrix}
1\\
0\\
1
\end{bmatrix}
+
0.114
\begin{bmatrix}
0\\
2\\
1
\end{bmatrix}
+
0.042
\begin{bmatrix}
1\\
1\\
0
\end{bmatrix}
=

\begin{bmatrix}
0.886\\
0.270\\
0.958
\end{bmatrix}
$$

### 8.5 从 latent 变成 head 输出

为了简单，假设 Head 1 的 $W_1^{UV}$ 只取 latent 的前两维：

$$
W_1^{UV}(x,y,z)=(x,y)
$$

于是：

$$
o_{3,1}=
\begin{bmatrix}
0.886\\
0.270
\end{bmatrix}
$$

---

## 9. Head 2 的完整流向

Head 2 的未缩放分数是：

$$
\hat s_{3,j,2} = \tilde q_{3,2}^\top c_j^{KV} + q_{3,2}^R k_j^R
$$

其中：

$$
\tilde q_{3,2}=
\begin{bmatrix}
0\\
1\\
1
\end{bmatrix},
\qquad
q_{3,2}^R=-1
$$

### 9.1 对 token 1

$$
\tilde q_{3,2}^\top c_1^{KV}
============================

\begin{bmatrix}
0&1&1
\end{bmatrix}
\begin{bmatrix}
1\0\1
\end{bmatrix}
=1
$$

RoPE 项：

$$
q_{3,2}^R k_1^R = (-1)\cdot 1 = -1
$$

所以：

$$
\hat s_{3,1,2}=0
$$

### 9.2 对 token 2

$$
\tilde q_{3,2}^\top c_2^{KV}
============================

\begin{bmatrix}
0&1&1
\end{bmatrix}
\begin{bmatrix}
0\2\1
\end{bmatrix}
=3
$$

RoPE 项：

$$
q_{3,2}^R k_2^R = (-1)\cdot 0 = 0
$$

所以：

$$
\hat s_{3,2,2}=3
$$

### 9.3 对 token 3

$$
\tilde q_{3,2}^\top c_3^{KV}
=

\begin{bmatrix}
0&1&1
\end{bmatrix}
\begin{bmatrix}
1\\
1\\
0
\end{bmatrix}
=1
$$

RoPE 项：

$$
q_{3,2}^R k_3^R = (-1)\cdot(-1)=1
$$

所以：

$$
\hat s_{3,3,2}=2
$$

于是 Head 2 的未缩放分数向量是：

$$
[0,3,2]
$$

softmax 后近似为：

$$
\alpha_{3,:,2}\approx[0.035,,0.705,,0.259]
$$

也就是说，Head 2 在这一层里最偏向 token 2，其次是 token 3。

### 9.4 latent 聚合

$$
u_{3,2}
=

0.035,c_1^{KV}
+0.705,c_2^{KV}
+0.259,c_3^{KV}
$$

代入数值：

$$
u_{3,2}
=

\begin{bmatrix}
0.294\\
1.669\\
0.741
\end{bmatrix}
$$

### 9.5 变成 Head 2 输出

假设 Head 2 的 ($W_2^{UV}$) 取 latent 的后两维 的 ($W_2^{UV}$) 取 latent 的后两维：

$$
W_2^{UV}(x,y,z)=(y,z)
$$

于是：

$$
o_{3,2}
=

\begin{bmatrix}
1.669\\
0.741
\end{bmatrix}
$$

---

## 10. 这一层的 token feature 最终怎么流出去

把两个 head 的输出拼起来：

$$
\mathrm{Concat}(o_{3,1},o_{3,2})
=

\begin{bmatrix}
0.886\\
0.270\\
1.669\\
0.741
\end{bmatrix}
$$

再经过输出投影 (W^O)：

$$
y_3 = W^O \mathrm{Concat}(o_{3,1},o_{3,2})
$$

这就是 **token 3 在这一层 attention 子层的输出 feature**。
然后它会继续经过 residual、MLP/MoE、再送到下一层。论文附录 C 也把 MLA 的完整计算流程整理成一整串公式，并明确指出 generation 时需要缓存的是那几个蓝框向量；([arXiv][1])重建 K/V。citeturn842555view1

---

## 11. 把这个 toy 例子翻译成一句“token feature 流向图”

当前层里，token (t) 的流向其实就是：

$$
h_t
\rightarrow
c_t^Q
\rightarrow
(q_t^C,\ q_t^R)
$$

并行地：

$$
h_t
\rightarrow
(c_t^{KV},\ k_t^R)
$$

然后：

$$
(c_1^{KV},k_1^R),\dots,(c_t^{KV},k_t^R)
\rightarrow
\text{attention scores}
\rightarrow
\alpha_{t,j,i}
\rightarrow
u_{t,i}
\rightarrow
o_{t,i}
\rightarrow
W^O
\rightarrow
y_t
$$

如果换成更口语的说法，就是：

**当前 token 的本层 hidden 先分成“查询我想找什么”和“记忆我能提供什么”两股；历史 token 只留下压缩记忆；当前 token 用自己的 query 去扫这些压缩记忆，拿回加权后的结果，再变成本层输出。**

---

## 12. 这和源码怎么对上

在 DeepSeek 官方 `modeling_deepseek.py` 里，这条流向几乎可以一一对上：

* `q_a_proj -> q_a_layernorm -> q_b_proj`：对应 query 低秩路径
* `q` 被切成 `q_nope` 和 `q_pe`
* `kv_a_proj_with_mqa(hidden_states)`：先产生压缩后的 `compressed_kv` 与 `k_pe`
* `kv_b_proj(kv_a_layernorm(compressed_kv))`：把 `compressed_kv` 展开成 `k_nope` 和 `value_states`
* 再把 `q_nope/q_pe`、`k_nope/k_pe` 拼成通用 attention 所([Hugging Face][3])n284622view3turn284622view4

而 FlashInfer 的 MLA 文档则明确表明，专门优化的 decode / incremental prefill kernel 走的是 **Matrix Absorption + paged cache** 路线，直接围绕 `ckv` 和 ([FlashInfer][4])期缓存主形态。citeturn284622view5


---

## 13. 最后把这个例子浓缩成 3 个关键理解
|概念|标准 MHA 常见情况|DeepSeek-V2 / MLA 这里|
|---|---|---|
|`hidden_size`|主干 token 表示宽度|仍是主干 token 表示宽度|
|`num_attention_heads`|attention 头数|仍是 attention 头数|
|`head_dim`|每头子空间维度|这里 query 每头维度是 `128+64=192`|
|`hidden_size = num_heads × head_dim`|常常成立|不必成立|
|attention 内部总宽度|常与 hidden_size 相同|可以大于 hidden_size|
|最终输出|回到 hidden_size|仍回到 hidden_size|


第一，**MLA 在 layer 层面不是“少几个 KV 头”那么简单，而是把每个 token 的“记忆载体”改成了更小的 latent (c_t^{KV})**。论文把这定义成 low-rank([arXiv][1])n764825view0turn284622view1

第二，**decoupled RoPE 的作用是把位置信息从压缩 KV 主体里分离出来**，这样 (W^{UK}) 才能被吸收到 query 侧；否则推理时就不得([arXiv][1])n284622view1turn842555view0

第三，**从 token feature 流向上看，MLA 这一层做的是：当前 token 产生 query，历史 token 提供 latent 记忆，query 在 latent 空间里做检索，最后再把读回来的 latent 变换成输出 ([arXiv][1])n842555view1turn284622view5

你要是愿意，我下一步可以把这个例子再升级一层：把它画成一张 **“单层 MLA 张量维度流转图”**，或者直接换成 **DeepSeek-V2 真实维度**，给你算一

“每层每 token 到底写进 cache 多少字节”。

[1]: https://arxiv.org/html/2405.04434v4 "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
[2]: https://huggingface.co/docs/transformers/en/model_doc/deepseek_v2 "DeepSeek-V2 · Hugging Face"
[3]: https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/modeling_deepseek.py "modeling_deepseek.py · deepseek-ai/DeepSeek-V2 at main"
[4]: https://docs.flashinfer.ai/api/attention.html "FlashInfer Attention Kernels - FlashInfer 0.6.6 documentation"

