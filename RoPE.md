
# 讲义标题

**从 Transformer 输出头到 RoPE、YaRN 与 Qwen2-VL M-RoPE**

---

## 第 1 章：Transformer 的输出头到底在做什么

### 这一章讲什么

生成式 Transformer 的最后一层通常是把隐藏状态从 `hidden_size` 投影到 `vocab_size`，输出每个位置对整个词表的 logits。Hugging Face 文档明确说明，语言模型 `logits` 的形状是 `(batch_size, sequence_length, config.vocab_size)`。([Hugging Face][2])

### 核心公式

若最后一层隐藏状态为

$$
H \in \mathbb{R}^{B\times T\times D},
$$

输出头权重为

$$
W \in \mathbb{R}^{V\times D},
$$

则

$$
\text{logits} = H W^\top + b,
$$

其中：

* $D = \text{hidden size}$
* $V = \text{vocab size}$

输出 shape：

$$
(B,T,D)\rightarrow(B,T,V)
$$

### 最小代码片段

```python
import torch
import torch.nn as nn

B, T, D, V = 2, 4, 8, 20
hidden_states = torch.randn(B, T, D)

lm_head = nn.Linear(D, V, bias=False)
logits = lm_head(hidden_states)

print(hidden_states.shape)  # [2, 4, 8]
print(lm_head.weight.shape) # [20, 8]
print(logits.shape)         # [2, 4, 20]
```

### 一个手算例子

设某个 token 的隐藏状态是

$$
h = [1,2]
$$

词表大小为 $V=3$，权重矩阵

$$
W=
\begin{bmatrix}
1 & 0\\
0 & 1\\
1 & 1
\end{bmatrix}.
$$

则

$$
\text{logits}=Wh=
\begin{bmatrix}
1\\
2\\
3
\end{bmatrix}.
$$

这表示模型对 3 个候选 token 的打分分别是 1、2、3。

### 一个常见误区

**误区**：最后一层 linear 负责“理解上下文”。
**纠正**：上下文建模主要发生在 attention/MLP 堆栈里；最后一层只是把已经建好的表示映射到任务空间。([Hugging Face][2])

---

## 第 2 章：位置编码的三大范式

### 这一章讲什么

位置编码不是只有一种做法。主流有三类：

1. **加到输入 embedding 上**：ViT / BERT 风格
2. **加到 Q/K 上**：RoPE 风格
3. **加到 attention score 上**：T5 相对位置 bias 风格

ViT 论文是 patch embedding 后加位置嵌入；RoFormer 是旋转 `q/k`；Flamingo、BLIP-2、Qwen2-VL 则代表不同的 VL 融合策略。([开放评审][3])

### 核心公式

输入相加型：

$$
X = E_{\text{token}} + E_{\text{position}}
$$

RoPE 型：

$$
q' = q\odot \cos + \operatorname{rotate_half}(q)\odot \sin
$$

$$
k' = k\odot \cos + \operatorname{rotate_half}(k)\odot \sin
$$

score bias 型：

$$ \text{scores} = \frac{QK^\top}{\sqrt{d}} + position _{bias} $$

### 最小代码片段

```python
# 1) 加到输入
x = token_embed + pos_embed

# 2) 加到 Q/K
q = q * cos + rotate_half(q) * sin
k = k * cos + rotate_half(k) * sin

# 3) 加到 score
scores = q @ k.transpose(-2, -1) / (d ** 0.5)
scores = scores + position_bias
```

### 一个手算例子

假设两个 token embedding：

$$
e_1=[1,1],\quad e_2=[2,2]
$$

位置向量：

$$
p_1=[0,1],\quad p_2=[1,0]
$$

则输入相加后：

$$
x_1=[1,2],\quad x_2=[3,2]
$$

这是“位置先混进表示里”的直观例子。

### 一个常见误区

**误区**：所有“位置编码”本质都一样。
**纠正**：它们注入的位置不同，决定了模型对位置的利用方式也不同。RoPE 是写进 attention 几何里，不等价于“再加一个位置向量”。([开放评审][3])

---

## 第 3 章：RoPE 的数学本体——为什么等价于复数乘法

### 这一章讲什么

RoPE 的核心不是“加位置”，而是**把每两维看成一个二维平面，对它做旋转**。RoFormer 论文明确指出，RoPE 用旋转矩阵编码绝对位置，并在 attention 中显式产生相对位置依赖。([arXiv][1])

### 核心公式

把一对维度写成复数：

$$
z_j = q_{2j}+ i q_{2j+1}
$$

位置为 $m$ 时，RoPE 后为

$$
\tilde z_j(m)= z_j e^{i m\theta_j}
$$

对应的实数二维旋转矩阵形式：

$$
\begin{pmatrix}
q'_{2j}\\
q'_{2j+1}
\end{pmatrix}
 =
\begin{pmatrix}
\cos(m\theta_j) & -\sin(m\theta_j)\\
\sin(m\theta_j) & \cos(m\theta_j)
\end{pmatrix}
\begin{pmatrix}
q_{2j}\\
q_{2j+1}
\end{pmatrix}
$$


### 最小代码片段

```python
import torch

def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)

def apply_rope(q, cos, sin):
    return q * cos + rotate_half(q) * sin
```

### 一个手算例子

设

$$
q=[1,0,0,1]
$$

把它拆成两对：

$$
(1,0),\quad (0,1)
$$

若旋转角是 $90^\circ$，则

$$
(1,0)\rightarrow(0,1),\quad (0,1)\rightarrow(-1,0)
$$

所以

$$
q'=[0,1,-1,0]
$$

这就是“旋转”的真正含义：**每一对维度在自己的二维子空间里转**。

### 一个常见误区

**误区**：RoPE 旋转的是整个 token 向量。
**纠正**：RoPE 旋转的是**每两维组成的二维子空间**，不是整个高维向量一把转过去。([arXiv][1])

---

## 第 4 章：为什么 RoPE 会让 attention 变成 relative position

### 这一章讲什么

RoPE 用绝对位置 $m,n$ 分别旋转 `q/k`，但进 attention 内积后，位置只以 $m-n$ 的形式出现，所以 attention 对位置的依赖是**相对位置**。这是 RoFormer 最关键的理论点之一。([arXiv][1])

### 核心公式

RoPE 后的 query / key：

$$
\tilde z_j(m)=z_j e^{i m\theta_j},\quad
\tilde w_j(n)=w_j e^{i n\theta_j}
$$

进入内积：

$$ \tilde z_j(m)\overline{\tilde w_j(n)} z_j\overline{w_j}e^{i(m-n)\theta_j} $$

因此位置项只依赖

$$
m-n
$$

### 最小代码片段

```python
# 伪代码：attention score 只会通过 q_rope, k_rope 的相位差反映位置
q_rope = apply_rope(q, cos_q, sin_q)
k_rope = apply_rope(k, cos_k, sin_k)
score = q_rope @ k_rope.transpose(-2, -1)
```

### 一个手算例子

若位置分别为 $m=5,n=3$，则相位差是

$$
m-n=2
$$

如果同时给两者都加 100，变成 $105,103$，相位差仍是 2，因此 relative position 不变。

### 一个常见误区

**误区**：RoPE 只编码绝对位置。
**纠正**：RoPE 用绝对位置旋转 `q/k`，但在 attention 分数里自然转化为相对位置差。([arXiv][1])

---

## 第 5 章：长上下文扩展——PI、NTK-aware、YaRN

### 这一章讲什么

RoPE 模型超过训练长度后容易失稳。PI、NTK-aware、YaRN 都是在“只动位置编码”的前提下，尽量把长上下文问题变得可控。PI 的主意是“统一压缩位置”；YaRN 的主意是“按频率分维度压缩，再补一个 attention 温度校正”。([arXiv][4])

### 核心公式

PI：

$$
m'=\frac{m}{s}
\qquad\Longleftrightarrow\qquad
\theta_d'=\frac{\theta_d}{s}
$$

YaRN / NTK-by-parts 的抽象写法：

$$
\theta_d'=(1-\gamma_d)\frac{\theta_d}{s}+\gamma_d\theta_d
$$

其中 $\gamma_d$ 由该维度在训练窗口中“转了多少圈”决定。

YaRN 温度修正：

$$
\sqrt{\frac{1}{t}} = 0.1\ln s + 1
$$

### 最小代码片段

```python
import math
import torch

def pi_position(pos, scale):
    return pos / scale

def yarn_attention_factor(scale):
    return 1.0 if scale <= 1 else 0.1 * math.log(scale) + 1.0
```

### 一个手算例子

原训练长度 $L=4096$，目标长度 $L'=32768$，则

$$
s=8
$$

PI 会把新位置 16000 映射成

$$
16000/8=2000
$$

也就是把它压回原始训练长度范围里。

YaRN 的温度因子此时是

$$
0.1\ln 8+1 \approx 1.208
$$

表示 attention 中会做额外缩放补偿。([arXiv][4])

### 一个常见误区

**误区**：YaRN 是纯 train-free 方法。
**纠正**：YaRN 原论文的主贡献是**高效微调扩窗**，同时它也系统化了可在推理时使用的缩放思路；真正更接近“未微调也尽量稳”的是 Dynamic Scaling / Dynamic-YaRN。Hugging Face 当前内置的 `rope_type="yarn"` 是静态参数化，不是论文里的动态版本。([arXiv][5])

---

## 第 6 章：VL 模型的位置编码三种范式

### 这一章讲什么

视觉语言模型里，“位置”有三处可能出现：

1. 视觉塔内部
2. 语言塔内部
3. 桥接层里

主流可分三类：LLaVA 类、Flamingo/BLIP-2 类、Qwen2-VL 类。Flamingo 用 cross-attention 桥接视觉和语言；BLIP-2 用 Q-Former；Qwen2-VL 则把视觉位置继续带进语言模型。([arXiv][6])

### 核心公式

LLaVA 类可抽象为：

$$
\text{image} \xrightarrow{\text{vision tower}} z_v
\xrightarrow{\text{projector}} z'_v
\rightarrow \text{insert into text sequence}
$$

Flamingo / BLIP-2：

$$
\text{text query} \xrightarrow{\text{cross-attn}} \text{visual memory}
$$

Qwen2-VL：

$$\text{vision tokens} +\text{text tokens} \rightarrow \text{LLM with M-RoPE}(t,h,w)
$$

### 最小代码片段

```python
# LLaVA-like
inputs_embeds = text_embed(input_ids)
inputs_embeds[image_mask] = image_features

# Flamingo/BLIP-2-like
text_states = cross_attention(text_states, visual_memory)
```

### 一个手算例子

假设图像切成 4 个 patch token：

$$
v_1,v_2,v_3,v_4
$$

文本是：

$$
t_1,t_2,t_3
$$

LLaVA 类会把它们拼成类似

$$
[v_1,v_2,v_3,v_4,t_1,t_2,t_3]
$$

交给语言模型。
Flamingo/BLIP-2 类则更像是：文本序列还是 `[t_1,t_2,t_3]`，但在某些层通过 cross-attention 去读视觉 memory。

### 一个常见误区

**误区**：所有 VL 模型都是“图像 token 插进文本序列”这一种。
**纠正**：Flamingo、BLIP-2 等模型并不必须把视觉 token 并到主序列里；它们往往通过 cross-attention 或 query bridge 保留独立视觉坐标系。([arXiv][6])

---

## 第 7 章：Qwen2-VL 的 M-RoPE

### 这一章讲什么

Qwen2-VL 的关键创新之一是 **M-RoPE（Multimodal Rotary Position Embedding）**。Qwen2-VL 论文和官方文档都明确指出，它把文本、图像、视频的位置统一进一个多模态 RoPE 框架。([arXiv][7])

### 核心公式

对视觉 token，用三元组位置

$$
(t,h,w)
$$

对文本 token，用

$$
(p,p,p)
$$

每个 head 的通道被切成三段：

$$
q=[q_t \Vert q_h \Vert q_w]
$$

然后分别做旋转：

$$
q'=
[R_t(t)q_t \Vert R_h(h)q_h \Vert R_w(w)q_w]
$$

### 最小代码片段

```python
# 概念化伪代码
q_t, q_h, q_w = torch.split(q, [32, 48, 48], dim=-1)

q_t = apply_rope(q_t, cos_t, sin_t)
q_h = apply_rope(q_h, cos_h, sin_h)
q_w = apply_rope(q_w, cos_w, sin_w)

q_out = torch.cat([q_t, q_h, q_w], dim=-1)
```

### 一个手算例子

假设某个视觉 token 在语言塔里的位置是：

$$
(t,h,w)=(0,5,10)
$$

某个 128 维 head 会按

$$
[32 \mid 48 \mid 48]
$$

切成三段：

* 前 32 维：用 $t=0$ 旋转
* 中 48 维：用 $h=5$ 旋转
* 后 48 维：用 $w=10$ 旋转

而一个文本 token 若在序列位置 $p=7$，则用

$$
(7,7,7)
$$

所以它退化为普通 1D RoPE。([arXiv][7])

### 一个常见误区

**误区**：Qwen2-VL 的视觉位置只在视觉塔里编码一次。
**纠正**：Qwen2-VL 既在视觉塔里做视觉 rotary，也在语言塔里用 M-RoPE 继续显式维护视觉 token 的时空坐标。([arXiv][7])

---

## 第 8 章：Qwen2-VL 一层 attention 的 shape 主线

### 这一章讲什么

Qwen2-VL 7B 的官方配置给了足够信息来手推 shape：视觉塔 `embed_dim=1280, num_heads=16`，语言塔 `hidden_size=3584, num_attention_heads=28, num_key_value_heads=4, mrope_section=[16,24,24]`。([Hugging Face][8])

### 核心公式

视觉头每头维度：

$$
1280/16=80
$$

语言头每个 query head 维度：

$$
3584/28=128
$$

语言头 GQA：

$$
q:[B,28,T,128],\quad k,v:[B,4,T,128]
$$

`mrope_section=[16,24,24]` 在实现中会乘 2，对应：

$$
32 \mid 48 \mid 48
$$

### 最小代码片段

```python
# 视觉头
x = torch.randn(1024, 1280)
qkv = torch.nn.Linear(1280, 1280 * 3)(x)
q, k, v = qkv.view(1024, 3, 16, 80).unbind(dim=1)

# 语言头
x = torch.randn(1, 356, 3584)
q = torch.nn.Linear(3584, 3584)(x).view(1, 356, 28, 128).transpose(1, 2)
k = torch.nn.Linear(3584, 512)(x).view(1, 356, 4, 128).transpose(1, 2)
v = torch.nn.Linear(3584, 512)(x).view(1, 356, 4, 128).transpose(1, 2)
```

### 一个手算例子

假设一张 `448×448` 图片，patch size=14，则

$$
32\times 32=1024
$$

个视觉 patch。视觉头一层 attention 输入可看作：

$$
[1024,1280]
\rightarrow q,k,v:[1024,16,80]
$$

再经过 `spatial_merge_size=2` 的 patch merger，视觉 token 数大致缩成

$$
1024/4=256
$$

并映射到语言维度 3584。若文本有 100 个 token，则语言层总长度约为

$$
T=256+100=356
$$

于是语言 attention 大致是：

$$
q:[1,28,356,128],\quad k,v:[1,4,356,128]
$$

再通过 GQA 扩成 28 个 heads。([arXiv][7])

### 一个常见误区

**误区**：视觉塔和语言塔的 attention 只是维度大小不同。
**纠正**：它们不仅维度不同，位置编码机制也不同：视觉塔做视觉 rotary，语言塔做按 `t/h/w` 分段的 M-RoPE。([arXiv][7])

---

## 第 9 章：复习用的总串联

### 这一章讲什么

把整条主线压成一句话：

> **输出头负责投影到目标空间；位置编码负责让 attention 感知顺序/空间；RoPE 把位置写进 `q/k` 的相位；YaRN 在长上下文里重写频率分布；Qwen2-VL 用 M-RoPE 把视觉空间继续带进语言推理。** ([arXiv][1])

### 核心公式

总串联可写成：

$$ \text{tokens} \xrightarrow{\text{embed}} H^{(0)}\xrightarrow{\text{pos inject}}\text{attention stack}\xrightarrow{\text{final hidden states}}\text{lm head}\xrightarrow{}\text{logits} $$

其中“pos inject”可以是：

* 输入相加
* Q/K 旋转
* score bias

### 最小代码片段

```python
x = embed(input_ids)
for layer in layers:
    q, k, v = proj(x)
    q, k = apply_position_mechanism(q, k)  # 这里可能是 RoPE/M-RoPE/bias
    x = attention(q, k, v) + x
logits = lm_head(x)
```

### 一个手算例子

你可以用最小 4 维 RoPE 例子复习整条链：

* 输入 token hidden state：`[1,0,0,1]`
* 位置 $m=1$
* 旋转后变成新 `q/k`
* attention score 只依赖相对位置差
* 最后 hidden state 再经 `lm_head` 输出 logits

### 一个常见误区

**误区**：位置编码只是实现细节。
**纠正**：位置编码直接决定模型如何理解“顺序、距离、时空结构”，在长上下文和多模态里尤其不是细枝末节，而是主架构问题。([arXiv][1])

---

## 参考资料（建议按这个顺序读）

先读 **RoFormer**，把 RoPE 的数学本体吃透；再读 **PI** 和 **YaRN**，理解为什么长上下文会坏、为什么只改位置编码也能改善；接着看 **ViT、Flamingo、BLIP-2、Qwen2-VL**，理解视觉位置如何进入模型；最后对照 **Hugging Face `rope_utils`、Qwen2-VL 文档/配置/源码** 看实现。([arXiv][1])

---

[1]: https://arxiv.org/abs/2104.09864?utm_source=chatgpt.com "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[2]: https://huggingface.co/docs/transformers/en/model_doc/qwen2?utm_source=chatgpt.com "Qwen2"
[3]: https://openreview.net/forum?id=YicbFdNTTy&utm_source=chatgpt.com "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
[4]: https://arxiv.org/abs/2306.15595?utm_source=chatgpt.com "Extending Context Window of Large Language Models via Positional Interpolation"
[5]: https://arxiv.org/abs/2309.00071?utm_source=chatgpt.com "YaRN: Efficient Context Window Extension of Large Language Models"
[6]: https://arxiv.org/abs/2204.14198?utm_source=chatgpt.com "Flamingo: a Visual Language Model for Few-Shot Learning"
[7]: https://arxiv.org/abs/2409.12191?utm_source=chatgpt.com "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution"
[8]: https://huggingface.co/docs/transformers/model_doc/qwen2_vl?utm_source=chatgpt.com "Qwen2-VL"

如果你要，我可以继续把这一版再整理成**可直接粘贴到 Typora / Obsidian / GitHub README 的最终排版版**。
