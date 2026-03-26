下面是一份把前面内容重新编排后的 **学习笔记版**。我把它整理成“先懂原理，再看 MLA，最后落到工程和显存”的顺序，适合系统复习。

---

# KV Cache 与 MLA 学习笔记

## 1. 这份笔记要解决什么问题

这份笔记围绕两个核心问题展开：

1. **为什么 Transformer 推理需要 KV Cache**
2. **为什么 MLA（Multi-head Latent Attention）能进一步压缩 KV Cache，同时尽量保持模型能力**

Hugging Face 官方文档明确指出，KV cache 的作用是保存历史 token 在注意力层里产生的 key/value，避免后续生成时重复计算；并且 cache 应只在推理时使用，不建议训练时开启。DeepSeek-V2 论文则提出 MLA，通过把 KV 压缩成 latent 表示来显著降低推理时的缓存开销。([Hugging Face][1])

---

## 2. 背景：为什么没有 KV Cache 会很慢

对 decoder-only Transformer，自回归生成是“一个 token 一个 token 地往后吐”。

设第 (l) 层输入为：

$$
X^{(l)} \in \mathbb{R}^{T \times d_{\text{model}}}
$$

标准注意力的线性投影是：

$$
Q^{(l)} = X^{(l)}W_Q^{(l)},\qquad
K^{(l)} = X^{(l)}W_K^{(l)},\qquad
V^{(l)} = X^{(l)}W_V^{(l)}
$$

注意力输出为：

$$
\mathrm{Attn}(Q,K,V)=\mathrm{softmax}!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V
$$

其中 (d_h) 是单头维度，(M) 是 causal mask。这个形式正是 Transformer 自注意力的基础，而 HF 的缓存文档讨论的也是这里的 K/V 复用问题。([Hugging Face][1])

### 2.1 没有 KV Cache 时

假设 prompt 长度为 (P)，接下来还要生成 (N) 个 token。
如果每一步都把“当前已有的整段前缀”重新 forward 一遍，那么第 (s) 步的大致 attention 工作量与 ((P+s)^2) 成正比，因此总量近似为：

$$
C_{\text{no-cache}} \propto \sum_{s=1}^{N}(P+s)^2
$$

展开后：

$$
\sum_{s=1}^{N}(P+s)^2
=
NP^2 + P N(N+1) + \frac{N(N+1)(2N+1)}{6}
$$

所以生成越长，重复计算越严重。HF 文档对这一点的描述很直接：KV cache 的目的，就是避免在每个新 token 上重新计算历史 token 的 K/V。([Hugging Face][1])

---

## 3. KV Cache 的核心思想

### 3.1 只缓存 K 和 V，不缓存 Q

对第 (t) 个 token，在第 (l) 层：

$$
q_t^{(l)} = x_t^{(l)}W_Q^{(l)},\qquad
k_t^{(l)} = x_t^{(l)}W_K^{(l)},\qquad
v_t^{(l)} = x_t^{(l)}W_V^{(l)}
$$

缓存递推写成：

$$
\mathcal{K}*t^{(l)} = [\mathcal{K}*{t-1}^{(l)};; k_t^{(l)}],\qquad
\mathcal{V}*t^{(l)} = [\mathcal{V}*{t-1}^{(l)};; v_t^{(l)}]
$$

然后当前 token 只需要用自己的 query 去读取历史：

$$
o_t^{(l)}=
\mathrm{softmax}!\left(
\frac{q_t^{(l)}(\mathcal{K}_t^{(l)})^\top}{\sqrt{d_h}} + m_t
\right)\mathcal{V}_t^{(l)}
$$

直观上：

* **Q 不缓存**，因为历史 token 的 query 以后不会再拿来“主动查询”
* **K/V 缓存**，因为未来每一步都要读取它们

这正是 HF 官方对 KV cache 的定义：存储历史 token 的 K/V，并在后续 token 上复用。([Hugging Face][1])

### 3.2 有 KV Cache 后的复杂度直觉

有了 cache 之后，prefill 仍要处理整段 prompt，但 decode 阶段不需要反复重算历史 K/V。总工作量可粗略写成：

$$
C_{\text{cache}} \propto P^2 + \sum_{s=1}^{N}(P+s)
$$

即：

$$
C_{\text{cache}}
=
P^2 + NP + \frac{N(N+1)}{2}
$$

因此：

* **prefill**：仍然昂贵
* **decode**：由“不断重算整段前缀”变成“只算新 token + 读旧 cache”

这也是 KV cache 在推理里最核心的价值。([Hugging Face][1])

---

## 4. KV Cache 的张量形状与显存公式

常见情况下，一层的缓存可记为：

$$
K,V \in \mathbb{R}^{B \times H_{\text{kv}} \times T \times D}
$$

其中：

* (B)：batch size
* ($H_{\text{kv}}$)：KV 头数
* (T)：缓存的 token 数
* (D)：每个头的维度

HF 当前文档也围绕不同 cache 策略讨论了这类缓存结构及其管理方式。([Hugging Face][2])

### 4.1 每层、每 token 的缓存大小

如果 K 和 V 都用同一种精度，每个元素占 (b) 字节，则：

$$
\text{bytes\_per\_token\_per\_layer}
=
2 \cdot H_{\text{kv}} \cdot D \cdot b
$$

乘上层数 (L)、上下文长度 (T)、batch (B)，总 KV cache 为：

$$
\boxed{
M_{\text{KV}}
=
B \cdot T \cdot L \cdot 2 \cdot H_{\text{kv}} \cdot D \cdot b
}
$$

这个公式解释了为什么长上下文推理特别吃显存：**KV cache 随上下文长度 (T) 线性增长**。HF 文档和 vLLM 的设计文档都把 KV cache 视为推理性能和容量的关键因素。([Hugging Face][2])

---

## 5. MHA、GQA、MQA 与 KV Cache

三者的差别，本质是 **KV 头数不同**：

### 5.1 MHA

多头注意力中，query/head 与 key/value/head 一一对应：

$$
H_{\text{kv}} = H_q
$$

### 5.2 GQA

多个 query head 共享一组 key/value head：

$$
H_{\text{kv}} < H_q
$$

### 5.3 MQA

所有 query head 共用 1 组 key/value：

$$
H_{\text{kv}} = 1
$$

因此 KV cache 显存与 **KV 头数** 成正比，而不是与 query 头数成正比。DeepSeek-V2 论文在对比 MHA、GQA、MQA 与 MLA 时，也是以“每 token 的缓存元素数”来做横向比较。([arXiv][3])

---

## 6. 推理显存由哪些部分组成

推理时的显存可以粗略拆成：

$$
M_{\text{peak}}
\approx
M_{\text{weights}}
+
M_{\text{KV}}
+
M_{\text{act\_peak}}
+
M_{\text{workspace}}
$$

其中：

* $M_{\text{weights}}$：模型权重
* $M_{\text{KV}}$：缓存，随上下文变长
* $M_{\text{act\_peak}}$：瞬时激活
* $M_{\text{workspace}}$：kernel 临时工作区

### 6.1 prefill 与 decode 的区别

**Prefill**：一次处理整段 prompt，计算量大，激活峰值通常更高。
**Decode**：一次只处理 1 个新 token，更偏 memory-bound，长期常驻的大头往往是 KV cache。

这也是为什么工程里经常会把 prefill 和 decode 分开优化；vLLM、FlashInfer 等推理系统都围绕 paged cache、kernel 优化、decode 路径做大量设计。([vLLM][4])

---

## 7. 工程实现：顶尖开源库怎么看 KV Cache

## 7.1 Hugging Face Transformers

HF 官方文档区分了不同 cache strategy，并强调：

* KV cache 用于推理提速
* 不同策略会在灵活性、内存占用、是否适合编译之间做权衡

这使 HF 成为理解“标准参考实现”的最好入口。([Hugging Face][2])

## 7.2 vLLM：PagedAttention

vLLM 的核心设计之一是 **PagedAttention**。官方设计文档明确指出：
KV cache 会被划分成固定大小的 **KV Blocks**，这些块可以存放在**非连续物理内存**中，从而按需分配并减少碎片。([vLLM][4])

直观理解：

* 不再要求“一个请求占一大块连续显存”
* 改成“一个请求由多个 block 组成”
* 逻辑序列和物理内存通过 block table 映射

所以 vLLM 的优势不只是“有 cache”，而是 **更适合高并发服务场景下的 cache 管理**。([vLLM][4])

## 7.3 量化 KV Cache

vLLM 官方文档还支持 **Quantized KV Cache**，例如 FP8 KV cache；文档说明这能进一步降低内存占用，而在某些后端上 attention 本身也会在量化域中运行。([vLLM][5])

---

# 8. MLA：为什么它比 GQA/MQA 走得更远

## 8.1 MLA 的出发点

DeepSeek-V2 提出的 MLA（Multi-head Latent Attention）不是简单减少 KV 头数，而是**重新定义缓存里到底存什么**。论文直接指出，MLA 通过低秩联合压缩，把推理时的 KV cache 压到一个 latent 向量上，并报告整体上显著降低 KV cache、提升吞吐。([arXiv][3])

---

## 9. MLA 的第一步：把 K/V 压缩成 latent

设 token 表示为 (h_t \in \mathbb{R}^{d})。
MLA 先做一个联合下投影：

$$
c_t^{KV}=W^{DKV}h_t
$$

其中 (c_t^{KV}\in\mathbb{R}^{d_c})，且通常：

$$
d_c \ll n_h d_h
$$

随后再通过上投影恢复出 K/V 相关部分：

$$
k_t^C=W^{UK}c_t^{KV},\qquad
v_t^C=W^{UV}c_t^{KV}
$$

MLA 的关键不是“推理时总把 (k_t^C,v_t^C) 真正展开出来再缓存”，而是：**推理阶段主要缓存 (c_t^{KV})**。DeepSeek-V2 论文对这一点讲得非常明确。([arXiv][3])

因此，如果先忽略位置编码，MLA 的理想缓存大小是：

$$
\boxed{
M_{\text{MLA-ideal}} = B\cdot T\cdot L\cdot d_c \cdot b
}
$$

相比普通 MHA 的

$$
M_{\text{MHA}} = B\cdot T\cdot L\cdot 2n_h d_h \cdot b
$$

MLA 的思路是：**不存完整 K/V，而是存一个更小的 latent。** ([arXiv][3])

---

## 10. MLA 的数学灵魂：矩阵吸收（Matrix Absorption）

看 attention 打分项。对第 (i) 个头：

$$
(q_{t,i}^C)^\top k_{j,i}^C
$$

而：

$$
k_{j,i}^C = W_i^{UK} c_j^{KV}
$$

代入得：

$$
(q_{t,i}^C)^\top k_{j,i}^C
=

 (q_{t,i}^C)^\top W_i^{UK} c_j^{KV}

\big((W_i^{UK})^\top q_{t,i}^C\big)^\top c_j^{KV}
$$

定义吸收后的 query：

$$
\tilde q_{t,i} = (W_i^{UK})^\top q_{t,i}^C
$$

则有：

$$
(q_{t,i}^C)^\top k_{j,i}^C = \tilde q_{t,i}^\top c_j^{KV}
$$

这说明：
**不必真的把每个 token 的完整 key 展开出来缓存；只要把上投影矩阵吸收到 query 一侧，就能直接和 latent 做打分。**

同理，value 侧也能吸收到输出变换里。DeepSeek-V2 论文明确说明，MLA 在推理时可通过这种矩阵吸收避免显式恢复完整 K/V。([arXiv][3])

---

## 11. 但普通 RoPE 会破坏吸收，于是需要 decoupled RoPE

如果直接把 RoPE 作用在会被压缩/展开的 key 上，那么位置旋转与上投影矩阵 (W^{UK}) 会耦合起来，矩阵吸收就不成立。DeepSeek-V2 因此提出 **decoupled RoPE**。([arXiv][3])

它把 query/key 分成两部分：

### 11.1 非位置部分

用于 latent 相关计算。

### 11.2 RoPE 部分

单独处理：

$$
q_t^R = \mathrm{RoPE}(W^{QR} c_t^Q),\qquad
k_t^R = \mathrm{RoPE}(W^{KR} h_t)
$$

最终每个头的 query/key 可以写成拼接：

$$
q_{t,i}=[q_{t,i}^C;; q_{t,i}^R]
$$

$$
k_{t,i}=[k_{t,i}^C;; k_t^R]
$$

论文特别强调：这里的 RoPE key 可以是**跨头共享**的，这使 MLA 只需额外缓存一小段位置相关 key，而不是再回到完整多头 K 的形态。([arXiv][3])

于是 MLA 最终的每 token 缓存元素数变成：

$$
\boxed{
(d_c + d_h^R)L
}
$$

其中  $d_h^R$ 是额外的 RoPE 维度。([arXiv][3])

---

## 12. MLA 与 MHA / GQA / MQA 的缓存比较

DeepSeek-V2 给出的对比可以概括为：

$$
\text{MHA: } 2n_h d_h L
$$

$$
\text{GQA: } 2n_g d_h L
$$

$$
\text{MQA: } 2 d_h L
$$

$$
\text{MLA: } (d_c+d_h^R)L
$$

这几项都是“每 token 的缓存元素数”。因此 MLA 并不是简单的“更极端的 MQA”，而是**通过 latent 表示 + decoupled RoPE 重构了缓存形式**。这正是 DeepSeek-V2 论文对 MLA 的定位。([arXiv][3])

---

## 13. 用一个数值例子感受 MLA 的压缩力度

假设：

$$
d_h = 128,\qquad d_c = 512,\qquad d_h^R = 64
$$

则 MLA 每层每 token 的缓存元素数为：

$$
d_c+d_h^R = 512+64=576
$$

如果使用 bf16，每元素 2 字节，则：

$$
576\times 2 = 1152\ \text{bytes / layer / token}
$$

而若使用 32 头 MHA，则每层每 token 为：

$$
2\times 32\times 128 = 8192\ \text{elements}
$$

对应 bf16：

$$
8192\times 2 = 16384\ \text{bytes / layer / token}
$$

也就是从每层每 token **16 KB** 左右下降到 **1.125 KB** 左右。DeepSeek-V2 论文报告的整体效果是：KV cache 显著下降，并带来更高吞吐。HF 的 DeepSeek-V2 文档也公开了与 MLA 相关的配置字段。([arXiv][3])

---

## 14. MLA 省掉的主要是什么

MLA 省掉的主要是：

* **长上下文下常驻的 cache 显存**
* **decode 阶段持续读取大块 K/V 的带宽压力**

它并不意味着“所有推理开销都被消灭”。
特别是在 prefill 阶段，整段 prompt 的处理仍然昂贵，因此工程上往往会针对 prefill 和 decode 设计不同路径。FlashInfer 的 MLA 文档就明确区分了 paged MLA attention、Matrix Absorption 以及不同 wrapper 的使用场景。([FlashInfer][6])

---

## 15. 工程实现：MLA 在开源系统中的形态

## 15.1 FlashInfer

FlashInfer 提供了面向 MLA 的 paged attention 接口，并在文档里直接提到：

* 这是 DeepSeek 风格 MLA 的 kernel
* 支持 paged KV cache
* 实现中会结合 **Matrix Absorption**

这说明 MLA 不只是论文概念，而是已经进入高性能推理内核实现。([FlashInfer][6])

FlashInfer 还提供了 `append_paged_mla_kv_cache` 一类接口，并明确提到目前支持的 `ckv=512`、`kpe=64` 等布局，这与 DeepSeek 系列 MLA 配置相呼应。([FlashInfer][7])

## 15.2 vLLM

vLLM 的基础设施本来就围绕 paged KV cache 设计，因此很适合承载这类“缓存形式更复杂”的 attention 机制。其 PagedAttention 设计文档强调了 block 化与非连续物理存储对高并发服务的重要性。([vLLM][4])

---

## 16. 一个统一脑图：从普通 KV Cache 到 MLA

可以把几种机制放在一张图里理解：

### 16.1 普通 MHA

* 每个头都有自己的 K/V
* 全部缓存
* 表达力强，但 cache 最大

### 16.2 GQA

* 多个 query head 共享若干组 K/V
* cache 比 MHA 小
* 是“减少 KV 头数”的路线

### 16.3 MQA

* 所有 query head 共用一组 K/V
* cache 很小
* 但表达能力压缩得更猛

### 16.4 MLA

* 不再把“完整 K/V”当作缓存主形态
* 改成缓存 **latent KV + 少量 RoPE key**
* 用矩阵吸收完成注意力打分与聚合
* 在压缩 cache 的同时尽量维持能力

DeepSeek-V2 对 MLA 的表述正是：通过低秩联合压缩显著缩小 KV cache，并保持很强的性能。([arXiv][3])

---

## 17. 公式速记区

### 17.1 普通 KV Cache 总公式

$$
\boxed{
M_{\text{KV}}
=

B \cdot T \cdot L \cdot 2 \cdot H_{\text{kv}} \cdot D \cdot b
}
$$

### 17.2 MHA / GQA / MQA 每 token 缓存元素数

$$
\text{MHA: } 2n_h d_h
$$

$$
\text{GQA: } 2n_g d_h
$$

$$
\text{MQA: } 2 d_h
$$

### 17.3 MLA 每 token 缓存元素数

$$
\text{MLA: } d_c + d_h^R
$$

### 17.4 推理峰值显存近似

$$
M_{\text{peak}}
\approx
M_{\text{weights}}
+
M_{\text{cache}}
+
M_{\text{act\_peak}}
+
M_{\text{workspace}}
$$

---

## 18. 常见误区

### 误区 1：KV Cache 缓存了所有注意力计算

不对。它主要缓存的是**历史 token 的 K/V**，不是把整个 attention 结果都缓存起来。([Hugging Face][1])

### 误区 2：KV Cache 主要解决的是注意力矩阵读取

更准确地说，它解决的是**每一步都重复计算历史前缀 K/V**的问题。([Hugging Face][1])

### 误区 3：MLA 就是另一种 MQA

不对。MQA 是“极限共享 K/V 头”；MLA 是“把缓存主形态改成 latent + 少量 RoPE key”。DeepSeek-V2 论文把这两者明确区分。([arXiv][3])

### 误区 4：MLA 会让所有推理开销都大幅消失

不对。它主要优化的是**decode 阶段的 cache 容量与带宽压力**；prefill 仍然可能很重。([FlashInfer][6])

---

## 19. 最后一段总结

**KV Cache** 的本质，是在自回归推理中保存历史 token 的 K/V，避免每生成一个新 token 都重新计算整段前缀。它显著加快 decode，但 cache 本身会随上下文长度线性膨胀，成为长上下文推理的重要显存瓶颈。([Hugging Face][1])

**MLA** 则更进一步：它不再把完整 K/V 当作缓存主对象，而是把历史信息压缩进 latent 表示，并通过矩阵吸收与解耦 RoPE 完成注意力计算。这使它在保持较强性能的同时，大幅压缩 KV cache，是 DeepSeek-V2 系列的一项关键设计。([arXiv][3])

---

## 20. 复习时建议优先记住的三句话

1. **KV Cache 省的是“重复计算历史前缀”的算力，不是把整层 attention 都缓存下来。** ([Hugging Face][1])
2. **长上下文推理里，常驻大头往往是 KV Cache，而不是瞬时激活。** 这也是 paged cache、quantized KV cache、GQA/MQA 如此重要的原因。([vLLM][4])
3. **MLA 的创新点不是“更少的 KV 头”，而是“把 KV Cache 的存储对象改成 latent”。** ([arXiv][3])

如果你愿意，我下一步可以把这份笔记继续整理成 **“适合背诵的提纲版”**，或者直接扩成 **“带示意图思路的讲义版”**。

[1]: https://huggingface.co/docs/transformers/cache_explanation?utm_source=chatgpt.com "Caching"
[2]: https://huggingface.co/docs/transformers/kv_cache?utm_source=chatgpt.com "Cache strategies"
[3]: https://arxiv.org/abs/2405.04434?utm_source=chatgpt.com "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
[4]: https://docs.vllm.ai/en/latest/design/paged_attention/?utm_source=chatgpt.com "Paged Attention - vLLM"
[5]: https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/?utm_source=chatgpt.com "Quantized KV Cache - vLLM"
[6]: https://docs.flashinfer.ai/api/attention.html?utm_source=chatgpt.com "FlashInfer Attention Kernels"
[7]: https://docs.flashinfer.ai/api/page.html?utm_source=chatgpt.com "flashinfer.page - FlashInfer 0.6.6 documentation"
