先给结论：**Context Engineering** 和 **Agentic Engineering** 不是同义词，也不是一前一后的简单升级关系。更准确地说，**Context Engineering 关注“模型在这一刻应当看到什么”**；**Agentic Engineering 关注“整个代理系统应当如何计划、调用工具、执行、交接、评估和受控地行动”**。在今天的工业语境里，前者通常是后者的核心子问题，但后者还多了运行时控制、编排、评估、治理和人类监督这些层面。([Anthropic][1])

这个问题难，难在两点。第一，这两个词本身都很“新”，尤其是 **agentic engineering**，它的**术语史**主要活在博客、仓库、公司页面和社交帖里，而不是已经稳定沉淀进教材或统一的学术定义里。第二，**概念的根**反而比**词本身**更早：很多今天被叫做 context engineering 或 agent engineering 的东西，早就在 GPT-3、RAG、ReAct、Toolformer、Generative Agents 这些工作里以别的名字存在了。所以下面我会把“**术语史**”和“**机制史**”分开讲，不把社区黑话冒充学术共识。([NeurIPS Proceedings][2])

## 1. Context Engineering：它到底是什么

如果只看高证据等级的“机制史”，Context Engineering 的真正源头，不是某条推文，而是 **“上下文本身决定模型行为”** 这条研究主线。GPT-3 的 few-shot 结果证明：**不用更新参数，仅靠文本上下文就能显著改变模型表现**。RAG 则把“上下文”从提示词扩展成了**参数记忆 + 外部非参数记忆**的组合，并明确提出了**可更新知识**与**可追溯 provenance** 的问题。随后 ReAct、Toolformer、Generative Agents 又把上下文推进到**工具调用、行动循环、记忆、反思、计划**这些结构化元素上。换句话说，今天所谓的 context engineering，本质上是把“prompt”扩展成“**整个推理时工作内存**”。([NeurIPS Proceedings][2])

这条路之所以会在 2025 年被单独命名，还有一个很硬的原因：**长上下文不是越长越好**。TACL 的 *Lost in the Middle* 明确发现，模型对长上下文中信息的利用并不稳健，尤其当关键信息位于中部时性能会显著下降。也就是说，问题不是“能塞多少 token”，而是“该塞哪些 token、以什么形式、在什么时机塞进去”。这正是 context engineering 从“提示词技巧”变成“系统工程问题”的分水岭。([ACL Anthology][3])

从**术语史**看，我能核实到的、面向 LLM/agent 语境的早期一手来源，是 Dex Horthy 的 **12-Factor Agents** 仓库（搜索结果标注为 2025 年 4 月 3 日）；其中第三条 “Own your context window” 直接写出 “**Everything is context engineering**”。HumanLayer 随后又在官网明确声称，这个仓库是“**the original repo that coined the term**”。这里要诚实一点：这种“谁首创”的说法本身证据等级不高，我**不能排除更早的零散用法**；但就我能核验的一手材料，它确实是最早、最有影响力的候选来源之一。([GitHub][4])

到了 2025 年 6 月，这个词开始快速成形。Walden Yan 在 Cognition 的《Don’t Build Multi-Agents》里已经把它当成一个正式问题域来讲，直接列出 “Principles of Context Engineering”；Harrison Chase 随后给出了一个被广泛引用的定义：**“构建动态系统，以正确的格式在正确时机提供正确的信息和工具，使 LLM 有可能完成任务。”** 到 2025 年 9 月，Anthropic 又把它进一步收紧为**推理时对最优 token 集合的策划与维护**，并明确说它是 prompt engineering 的自然延伸。这个演化很有意思：从“own your context window”这种工程直觉，发展到“dynamic systems”，再到“optimal set of tokens during inference”这种更接近形式化的定义。([Cognition][5])

所以，**今天较稳妥的定义**是：Context Engineering 不是单写一个好 prompt，而是**围绕一次或多次 LLM 调用，管理全部可进入上下文窗口的信息状态**——包括系统指令、消息历史、检索结果、工具描述、工具输出、短期状态、长期记忆、压缩摘要、权限与环境信息。LangChain 甚至把它细分成 **model context / tool context / life-cycle context** 三层，这个分法在工程上很实用。([LangChain 文档][6])

## 2. Agentic Engineering：它到底是什么

先说“概念根”。如果不拘泥于词，而看高证据等级的研究脉络，那么它来自 **LLM-based agents** 的发展：ReAct 把语言模型从“只回答”推进到“**边推理边行动**”；Generative Agents 把记忆、反思、计划做成了明确架构；2023 年的 agent survey 则已经把这个方向系统化为**构建、应用、评估**三大块。官方工程文档也很一致：OpenAI 把 agent 定义为带有**循环运行、工具调用、单/多 agent 编排、护栏、人类介入**的系统；Anthropic 也强调真正有效的 agent 往往来自**简单、可组合的模式**，不是花里胡哨的框架。([开放评审][7])

但 **Agentic Engineering** 这个词的**词义**现在其实有两条分支，而且这正是很多讨论会拧成麻花的地方。第一条分支，是 **“engineering AI agents”**：也就是“构建代理系统”的工程学。LangChain 在 2025 年 12 月写过《Agent Engineering: A New Discipline》，把它定义为当团队构建**会推理、会适应、行为非确定**的系统时需要承担的一组职责；同年一篇 biomedical 的社论更明确把 **Agent Engineering** 定义成 agent specification、orchestration、evaluation、governance 四部分。这个分支关心的是：**怎么把 agent 做出来，并让它在生产中可靠。** ([LangChain Blog][8])

第二条分支，是 **“doing engineering with agents”**：也就是**工程师借助 coding agents 做软件工程**。这一分支里，**“agentic engineering”** 这个精确短语在我能核实到的资料中，至少在 **Zed 2025 年 6 月 12 日**的页面上就已经出现，用来指“把 AI 集成进现有开发流程、学习与随机性共处的新工程技能”。随后在 2026 年 2 月，Karpathy 通过一条广泛传播的帖子把它推成了 “vibe coding 之后”的新说法；Business Insider、Observer 和 IBM 的解释都指向同一个意思：**不是完全不看代码的“凭感觉编程”，而是由专业工程师指挥、监督能够写代码并迭代测试的 agent。** Simon Willison 也把它概括成：**专业软件工程师使用 coding agents 放大自身能力。** ([Zed][9])

因此，**Agentic Engineering 目前不是一个单义词**。在“AI systems”语境里，它常常接近 **agent engineering**，意思是“构建与运营 agent 系统”；在“AI coding”语境里，它又常指“工程师如何与 coding agents 协作完成软件开发”。两者有重叠，但不等价。前者偏**系统构建**，后者偏**人机协作式的软件生产方式**。这也是为什么你会看到不同文章都在用这个词，但谈的其实不是同一件东西。([LangChain Blog][8])

## 3. 两者最本质的区别

最简洁的说法是：**Context Engineering 管“脑内”，Agentic Engineering 管“手脚和组织”。** 前者决定模型在每一步“看到什么、记得什么、忽略什么”；后者决定系统“何时调用模型、何时调用工具、何时分派子代理、何时中止、何时让人接管、如何评估与治理”。OpenAI 的 agent 指南把 while loop、single/multi-agent、guardrails、human oversight 都放在 agent design 里；而 Anthropic、LangChain 则把 context engineering 聚焦在 token 集、消息、工具、记忆、总结和生命周期钩子上。([OpenAI][10])

所以它们的**设计对象**不同。Context Engineering 的设计对象是**信息状态**：系统提示词怎么写，检索内容怎么选，工具描述是否歧义，旧消息何时压缩，长期记忆何时召回，哪些内容应该进入这一次模型调用。Agentic Engineering 的设计对象则是**行动系统**：单 agent 还是多 agent，manager 还是 handoff，工具边界怎么设，失败如何恢复，护栏怎么落，评估和追踪怎么做。前者的最小单位是“**一次调用的上下文装配**”，后者的最小单位是“**一个可持续运行的 agent loop / workflow**”。([LangChain 文档][6])

它们的**失败模式**也不同。Context Engineering 出问题时，更常见的是：信息缺失、信息冲突、无关信息过多、工具说明模糊、上下文膨胀、长上下文中关键事实被“丢在中间”。Agentic Engineering 出问题时，更常见的是：目标漂移、错误工具调用链、子代理交接失真、无限循环、缺乏护栏、没有人类审核高风险动作、以及评估闭环缺失。别看都叫 engineering，故障的味道完全不一样。([ACL Anthology][3])

## 4. 两者的关系：不是并列，而是“包含但不穷尽”

**不是所有 context engineering 都是 agentic engineering。** 一个做得很精细的 RAG 问答系统，即便没有自治规划、没有工具链循环、没有多代理编排，也可能已经包含很重的 context engineering：检索、重排、摘要、few-shot、记忆选择、上下文裁剪，全都在里面。RAG 论文和后来的 context engineering 综述都支持这一点：上下文设计本身就足以成为独立工程问题。([arXiv][11])

但反过来，**几乎所有严肃的 agentic systems 都离不开 context engineering**。Anthropic 直说，随着 agent 进入多轮、长时程任务，重点不再只是“怎么写 prompt”，而是“怎么管理整个 context state”；LangChain 文档更直接：agent 失败通常有两个原因，模型不够强，或者**没有把“正确”的上下文传给模型**，而且后者往往才是主因。说得直白一点：很多所谓 agent 很笨，不是它没脑子，而是你给它喂了一锅信息乱炖。宇宙很神秘，提示词糊成一锅粥也很神秘，但不是同一种神秘。([Anthropic][1])

在 coding-agent 的语境里，这种包含关系更明显。Simon Willison 所说的 agentic engineering，本质上是**工程师监督会写会跑代码的代理**；而这类代理能否真的在大代码库里干活，核心又落回 context engineering：它能否选对文件、保留关键轨迹、把错误压缩成高信号反馈、把工具结果以模型可消费的形式回灌。Zed 的表述“学习如何有效应对随机性是一种新的工程技能”，其实说的也是同一件事：**你不是在驯服一个确定性编译器，而是在调度一个概率性执行者。** ([Simon Willison’s Weblog][12])

## 5. 我认为最稳妥的一套定义

如果你希望一套尽量不随社区热词漂移的定义，我会这样下：

**Context Engineering**：在 LLM 推理过程中，系统性地**选择、组织、压缩、维护和更新**会进入模型工作上下文的信息，使模型在当前步骤“有可能”完成任务。它关注的是**信息供应链**。([Anthropic][1])

**Agentic Engineering**：围绕具有一定自主性的 LLM agent，系统性地**设计、编排、运行、评估和治理**其目标、工具、记忆、循环、交接与人类监督机制；在 coding-agent 语境里，它也可特指**工程师借助 coding agents 完成专业软件工程**的实践。它关注的是**行动控制系统**。([PMC][13])

## 6. 最后的判断

所以，若要一句话区分：

**Context Engineering 问的是：模型这一步该知道什么。**
**Agentic Engineering 问的是：系统下一步该做什么。**

再狠一点说：
没有 context engineering，agent 往往会**又忙又乱又自信**；
没有 agentic engineering，context engineering 再精致，也常常只是一个**很会读资料、但不会真正行动**的系统。([LangChain 文档][6])

你这个问题之所以“看似简单、实际上很难”，根子就在这：**术语是新的，但问题本身不是新的；词在变，工程约束没变。** 真正稳定的东西，不是 buzzword，而是那几条硬约束：有限上下文、非确定性、工具边界、反馈闭环、评估和治理。把这几条抓住，很多社区里的雾气就会散掉不少。([ACL Anthology][3])

接下来我可以把这篇内容继续整理成一版“学术型综述结构”，按“定义—词源—研究脉络—争议—实践框架—参考文献”展开。

[1]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents "Effective context engineering for AI agents \ Anthropic"
[2]: https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html "Language Models are Few-Shot Learners"
[3]: https://aclanthology.org/2024.tacl-1.9/ "Lost in the Middle: How Language Models Use Long Contexts - ACL Anthology"
[4]: https://github.com/humanlayer/12-factor-agents?utm_source=chatgpt.com "12-Factor Agents - Principles for building reliable LLM ..."
[5]: https://cognition.ai/blog/dont-build-multi-agents "Cognition | Don’t Build Multi-Agents"
[6]: https://docs.langchain.com/oss/python/langchain/context-engineering "Context engineering in agents - Docs by LangChain"
[7]: https://openreview.net/forum?id=tvI4u1ylcqs "ReAct: Synergizing Reasoning and Acting in Language Models | OpenReview"
[8]: https://blog.langchain.com/agent-engineering-a-new-discipline/ "Agent Engineering: A New Discipline"
[9]: https://zed.dev/agentic-engineering?utm_source=chatgpt.com "Agentic Engineering"
[10]: https://openai.com/business/guides-and-resources/a-practical-guide-to-building-ai-agents/ "A practical guide to building agents | OpenAI"
[11]: https://arxiv.org/abs/2005.11401 "[2005.11401] Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[12]: https://simonwillison.net/2026/Feb/23/agentic-engineering-patterns/ "Writing about Agentic Engineering Patterns"
[13]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12613637/ "
            From prompt engineering to agent engineering: expanding the AI toolbox with autonomous agentic AI collaborators for biomedical discovery - PMC
        "
