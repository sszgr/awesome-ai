# AI 术语解析 (AI Terminology Explained)

## 人工智能与机器学习（Artificial Intelligence & Machine Learning）

### 人工智能（Artificial Intelligence, AI）
人工智能是一门计算机科学，致力于研究如何让计算机具备模拟人类智能的能力，涵盖机器学习、深度学习、自然语言处理等多个领域。[1956年达特茅斯会议首次提出]

### 机器学习（Machine Learning, ML）
- **监督学习（Supervised Learning）**: 依赖标注数据进行训练，使其能够预测新输入的数据标签。
- **无监督学习（Unsupervised Learning）**: 通过数据自身的结构进行学习。处理未标注的数据，模型尝试发现数据中的结构或模式。
- **半监督学习（Semi-Supervised Learning）**: 结合少量标注数据和大量未标注数据进行训练，旨在提高模型的学习效果。
- **强化学习（Reinforcement Learning, RL）**: 通过奖励机制优化策略，以最大化累积奖励。
- **自监督学习（Self-Supervised Learning）**: ​从数据自身生成监督信号进行训练，无需人工标注。
- **元学习（Meta Learning/Learning to Learn）**: 训练模型学习如何学习，使其能快速适应新任务，提升泛化能力。
- **联邦学习（Federated Learning）**: 分布式学习方法，保护数据隐私。在保护数据隐私的前提下，多个设备或机构协同训练共享模型，而无需集中数据。
- **图神经网络（Graph Neural Networks, GNN）**: 处理图结构数据的深度学习方法，能够捕捉节点和边之间的复杂关系。
- **自然语言处理（Natural Language Processing, NLP）**: 处理和理解人类语言的机器学习方法，应用于语音识别、情感分析、机器翻译等任务。
---

## 深度学习框架（Deep Learning Frameworks）

- **PyTorch**: 由 Meta（Facebook）开发的深度学习框架，以动态图计算和易用性著称。
- **TensorFlow**: 由 Google 研发，广泛用于生产环境的深度学习框架，支持分布式训练和高效推理。
- **JAX**: 由 Google 研发，优化自动微分和高效计算。
- **Keras**: 基于 TensorFlow 的高级深度学习 API，简化神经网络构建。
- **DeepSpeed**: 由微软开发的深度学习训练优化库，专注于大规模训练的效率优化，支持高效的分布式训练、混合精度训练和内存优化。
- ~~**MXNet**: 高效的分布式深度学习框架，提供高效、灵活且可扩展的深度学习模型训练和部署能力。​但由于生态等原因，已于2023年9月退役~~

---

## 大语言模型（LLM, Large Language Model）

### 模型架构
- **Embedding（嵌入表示）**: 将文本或其他数据转换为向量，便于计算。通过神经网络将文本、图像或其他数据转换为高维向量，使其能够在向量空间中进行计算。
- **Rerank（重排序）**: 在信息检索或推荐系统中，对初步筛选出的候选项进行重新排序，以提高相关性。
- **Agent（智能体）**: 具备自主决策能力的 AI 系统，能够感知环境、规划行动并执行任务。
- **Workflow（工作流）**: 任务或操作的执行流程，AI 可用于自动化工作流，减少人工干预。
- **Reasoner（推理器）**: 负责逻辑推理的 AI 组件，使模型能够进行更复杂的推理和决策。例如，CoT（Chain of Thought, 思维链）推理通过分步推理提升 LLM 在数学和逻辑任务上的能力。现代推理器可以结合外部工具（如 Python 计算、数据库查询）来增强准确性。
- **Prompt（提示词）**: 影响 LLM 生成内容的输入指令，决定模型的输出方式、格式和内容方向。
- **Instructions（指令）**:  设定 LLM 的行为规则，提高模型的可控性。常用于 System Prompt（系统提示） 或 API 调用中，以控制 AI 的语气、风格、限制范围。

### 学习范式
- **Few-shot Learning（少样本学习）**: 仅使用少量示例进行学习。
- **Zero-shot Learning（零样本学习）**: 在没有示例的情况下进行推理。
- **Multimodal Learning（多模态学习）**: 结合文本、图像、音频等多种数据类型进行学习。
- **Adapters（适配器）**: 用于高效微调大型预训练模型。

### 关键技术
- **CoT（Chain of Thought, 思维链）**: 引导模型进行逐步推理，提升推理能力和结果透明度。这个概念由 Google Research 的研究人员在一篇名为《Chain-of-Thought Prompting Elicits Reasoning in Large Language Models》（《思维链提示促使大语言模型进行推理》）的论文中提出。
- **Hallucination（幻觉）**: 由于概率预测机制，大模型可能会生成不准确或虚假的信息，即‘幻觉’现象。可以通过 RAG、CoT 等技术缓解。
- **RAG（Retrieval-Augmented Generation, 检索增强生成）**: 结合内部知识库进行信息补充。通过查询外部知识库来丰富生成内容，从而提升语言模型在知识密集型任务中的表现。最早在2020年由 Facebook AI Research 的论文《Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks》（《检索增强生成：知识密集型自然语言处理任务的新方法》）中提出。
- **RAGFlow**: 对RAG进行优化的数据流方法，提高检索与生成的结合效率。
- **PAL（Program-Aided Language Models, 程序辅助语言模型）**: 让模型调用程序进行计算。通过程序辅助的方法提升语言模型在解决复杂推理任务时的表现。最早在2022年论文《PAL: Program-aided Language Models》（《PAL：程序辅助语言模型》）中提出。
- **ReAct（Reasoning and Acting, 推理+行动）**: 结合推理与行动，例如让AI通过搜索获取最新信息。主要目标是将推理（Reasoning）与执行（Acting）结合起来，使得模型能够在回答问题时进行推理并采取行动。2022年论文《ReAct: Synergizing Reasoning and Acting in Language Models》（《ReAct: 在语言模型中协同推理和行动》）中提出。
- **MoE（Mixture of Experts, 专家混合模型）**: 通过多个专家子模型提升性能。为每个输入动态选择最相关的子模型，从而在扩展模型容量的同时保持计算效率。最早在2017在 Google 的论文《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》（《超大规模神经网络：稀疏门控专家混合层》）中提出。
- **Function Calling（函数调用）**: 允许 LLM 以结构化方式调用外部 API、数据库查询或执行计算，提高 AI 在应用场景中的可操作性。最早由 OpenAI 在 GPT-4 Turbo 版本中广泛应用，能够让模型与插件或外部工具交互，实现数据查询、代码执行和任务自动化。
- **MCP（Model Context Protocol，模型上下文协议）**: 由 Anthropic 于 2024 年 11 月推出的一项开放标准，旨在为大型语言模型（LLM）应用提供标准化接口，使其能够连接和交互外部数据源和工具。MCP 的目标是克服 LLM 应用仅依赖训练数据的局限性，使其能够访问所需的上下文信息，并执行更广泛的任务。


## 模型优化技术 (Optimization Techniques)
### 高效微调
- **LoRA（Low-Rank Adaptation）** 通过低秩矩阵对预训练模型进行高效微调，仅调整少量参数，从而显著减少计算和存储需求，提高微调效率。这个概念在2021年由 Microsoft 研究人员在论文《LoRA: Low-Rank Adaptation of Large Language Models》（《LoRA：大语言模型的低秩适应》）中提出。
- **QLoRA（Quantized Low-Rank Adaptation）** 在 LoRA 的基础上，进一步引入量化技术，减少模型存储需求，并保持微调的高效性和性能。适用于在资源受限的设备上运行大型语言模型。
- **SFT（Supervised Fine-Tuning, 监督微调）**: 在特定任务上进行微调。

### 推理加速
- **vLLM** 采用了优化的硬件加速技术和智能内存管理，提升了推理的效率和吞吐量。
- **Triton** OpenAI 提出的深度学习编程语言和框架，旨在通过简化高效 GPU 编程来加速模型训练和推理过程。

## Transformer 架构与变体（Transformer Architecture & Variants）

### 原始论文
2017 年 Google 发布 **《Attention Is All You Need》（《注意力即一切》）** 论文，提出 Transformer 架构，彻底改变自然语言处理（NLP）领域。

### 核心概念
- **Encoder/Decoder（编码器/解码器）**: 编码器-解码器结构。
- **Token（标记）**: 语言单位，通常为单词、子词或字符。文本数据的基本构建块，模型通过将文本拆分为这些标记来处理和理解语言。
- **Vector（向量）**: 一种数学表示方法，用于表示文本、数据或对象的特征。它通常是一个有序的数值数组，其中每个数值代表某个特定特征。Transformer提出的模型使用了512维的向量，到GPT-1是768，GPT-2是1600，GPT-3是12288，DeepSeek V3是7168。
- **Positional Encoding（位置编码）**: 用于保留序列信息。

### 主要变体
- **Encoder-only（仅编码器）**: 主要用于理解任务，例如文本分类、命名实体识别（NER）、句子嵌入等。采用双向注意力（Bidirectional Attention），可同时关注上下文信息。适用于自然语言理解任务，如情感分析、问答系统、信息检索等。
- **Decoder-only（仅解码器）**: 主要用于文本生成任务，例如对话系统、代码生成、机器翻译等。采用 **自回归（Auto-regressive）** 方式生成文本，根据前面的 token 逐步预测下一个token。使用单向注意力（Causal Attention），避免看到未来的信息。
- **Encoder-Decoder（编码器-解码器）**: 同时使用编码器（Encoder）和解码器（Decoder），主要用于序列到序列（sequence-to-sequence）任务，如机器翻译、文本摘要等。
- **Vision Transformer（ViT）**: 将 Transformer 应用于计算机视觉任务，如图像分类、目标检测等。由 Google Research 的研究人员在一篇名为《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》（《一张图像值 16x16 个词：Transformer 在大规模图像识别中的应用》）的论文中提出。
- **Swin Transformer**：改进的 ViT 版本，采用滑动窗口机制，提高计算效率和精度，广泛应用于计算机视觉任务。由 Microsoft Research 的研究人员在一篇名为《Swin Transformer: Hierarchical Vision Transformer using Shifted Windows》（《Swin Transformer：使用平移窗口的分层视觉Transformer》）的论文中提出。
---

## 生成式AI（Generative AI）

- **AIGC（AI Generated Content）**: 利用人工智能自动生成文本、图像、音频、视频等多种内容形式。AIGC正在改变内容生产模式，提高创作效率，并推动个性化内容推荐的发展。
- **GPT（Generative Pre-trained Transformer）**: 生成式预训练 Transformer。从 GPT-1 到最新的 GPT-4，这些模型在自然语言处理（NLP）任务上表现卓越，被广泛应用于对话系统、代码生成和文本摘要等任务。
- **Diffusion Model（扩散模型）**: 生成图像的关键架构，如 Stable Diffusion。核心思想是逐步向数据添加噪声，然后通过反向过程去噪来生成清晰的样本。
- **StyleGAN**: 高质量图像生成的GAN变体，专门用于高质量图像生成。核心技术是风格混合（Style Mixing） 和 自适应实例归一化（Adaptive Instance Normalization, AdaIN），使得模型能够在不同级别上控制图像的风格特征，如面部特征、纹理、背景等。
- **DALL·E**: 由OpenAI 开发的文本生成图像模型。能够根据文本描述（Prompt）生成高质量的图像。
- **Stable Diffusion**: 基于扩散模型的文本生成图像工具。由 Stability AI 开发。它可以根据文本输入（Prompt）生成高质量的图像，并且可以在本地设备上运行，无需依赖云端计算。
- **DeepSeek VL**: 一种基于深度学习的视觉语言模型，结合了视觉和语言理解，用于图像和文本的跨模态生成任务。

---

## 经典神经网络模型（Classic Neural Networks）

- **RNN（Recurrent Neural Network, 循环神经网络）**: Transformer之前的主要序列处理模型。与传统神经网络不同，RNN具有内部循环连接，使其能够保留之前时间步的信息。然而，RNN在长期依赖问题上存在困难，容易出现梯度消失或梯度爆炸的问题。
- **LSTM（Long Short-Term Memory, 长短期记忆网络）**: RNN改良版，可处理长期依赖。它通过引入门控机制（包括输入门、遗忘门和输出门），使得模型能够在长时间步长中存储和读取信息，从而有效地处理长期依赖问题。
- **GRU（Gated Recurrent Unit, 门控循环单元）**: LSTM的简化版本，计算效率更高。由于其更简单的结构，GRU在一些任务中相较于LSTM训练更快，尤其是在数据量较小或需要快速训练的情况下。
- **CNN（Convolutional Neural Network, 卷积神经网络）**: 主要用于图像处理。通过使用卷积操作对输入数据进行局部特征提取，减少了参数量，同时提高了模型的泛化能力。
- **GAN（Generative Adversarial Networks, 生成对抗网络）**: 生成式模型的关键架构，通过两个神经网络（生成器和判别器）进行对抗训练。在生成深度伪造（DeepFake）图像时具有巨大的影响力。

---

## 其他AI相关技术（Other AI Technologies）
- **NeRF（Neural Radiance Fields, 神经辐射场）**: 用于3D场景生成，通过神经网络建模光线在 3D 场景中的传播，实现逼真渲染。
---
