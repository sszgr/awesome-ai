# DeepSeek
## 模型
### DeepSeek-R1
- [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)
- [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
- [DeepSeek-R1-Distill-Llama-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B)
- [DeepSeek-R1-Distill-Qwen-14B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B)
- [DeepSeek-R1-Distill-Qwen-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)
- [DeepSeek-R1-Distill-Llama-70B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B)
- [DeepSeek-R1-671B](https://huggingface.co/deepseek-ai/DeepSeek-R1) 
- [DeepSeek-R1-Zero-671B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Zero)  
  
### DeepSeek-V3
- [DeepSeek-V3-671B](https://huggingface.co/deepseek-ai/DeepSeek-V3)
- [DeepSeek-V3-Base-671B](https://huggingface.co/deepseek-ai/DeepSeek-V3-Base)
- [DeepSeek-V3-0324-671B](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)   
  2025.03.24 DeepSeek 发布更新版本 DeepSeek-V3-0324。   
  https://www.youtube.com/watch?v=LdYImZwa82o

## 开源周
- [FlashMLA](https://github.com/deepseek-ai/FlashMLA)  
  FlashMLA 是一个高效的多模态学习框架，旨在加速深度学习模型的训练和推理。通过优化计算过程，FlashMLA 提供了更快的处理速度和更高的性能。

- [DeepEP](https://github.com/deepseek-ai/DeepEP)  
  DeepEP 是 DeepSeek 提供的一个强化学习平台，专为动态环境中的模型优化设计，支持多种强化学习算法和环境。

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)  
  DeepGEMM 是一个针对矩阵运算的高效优化库，提供对大型矩阵计算的加速，支持多种硬件平台，提高了深度学习模型的训练效率。

- [DualPipe](https://github.com/deepseek-ai/DualPipe) & [EPLB](https://github.com/deepseek-ai/eplb)  
  DualPipe 通过优化计算与通信的重叠，减少传统流水线并行中的“气泡”（空闲等待时间），提升训练效率。EPLB 解决混合专家模型（MoE）训练中专家负载不均衡的问题，优化GPU资源分配。

- [3FS](https://github.com/deepseek-ai/3FS ) & [smallpond](https://github.com/deepseek-ai/smallpond)   
  3FS 通过整合高速SSD和RDMA网络技术，极大提升了分布式训练中数据的加载速度，减少了传输延迟。依托 DuckDB 和 3FS，smallpond 实现了轻量级分布式数据处理，简化了大规模数据分析的流程。

## 模型框架
- [Ollama](https://ollama.com/)  
  Ollama 是一个开源的大语言模型运行框架，让用户在本地轻松下载、运行和管理预训练模型。支持 Llama 3.3, DeepSeek-R1, Phi-4, Gemma 3 等主流大模型。  
  源码: [MIT] - https://github.com/ollama/ollama
- [Dify.AI](https://dify.ai/)  
  Dify 是一款开源的大语言模型(LLM) 应用开发平台。它融合了后端即服务（Backend as Service）和 LLMOps 的理念，使开发者可以快速搭建生产级的生成式 AI 应用。  
  源码: [Apache 2.0] - https://github.com/langgenius/dify
- [LangChain](https://www.langchain.com/)  
  LangChain 是一个用于构建由 LLM 驱动的应用程序的框架。提供了丰富的模块和工具，帮助开发者快速构建基于大语言模型的应用。  
  源码: [MIT] - https://github.com/langchain-ai/langchain

## 前端工具
- [LobeChat](https://lobehub.com/)  
  开源的对话聊天界面，让用户通过直观的 UI 与本地或远程的大型语言模型互动，快速构建聊天应用。  
  源码/社区版开源: [Apache 2.0] - https://github.com/lobehub/lobe-chat
- [Open WebUI](https://openwebui.com/)  
  开源、轻量级的 Web 界面项目，兼容 Ollama、OpenAI等API，并内置 RAG 推理引擎，帮助开发者快速构建和调试 AI 应用。  
  源码: [BSD 3-Clause New or Revised] - https://github.com/open-webui/open-webui
- [Chatbox](https://chatboxai.app/)
  一款跨平台的开源人工智能桌面客户端，支持 Ollama、OpenAI等多个 LLM 提供商。支持 Windows、macOS、Linux、iOS、Android 以及网页版。  
  源码/社区版开源: [GPL v3.0] - https://github.com/chatboxai/chatbox
- [AnythingLLM](https://anythingllm.com/)  
  全栈应用程序，可以将任何文档、资源（如网址链接、音频、视频）或内容片段转换为上下文，结合向量数据库解决方案，实现与用户提供的任何文档的智能交互。  
  源码: [MIT] - https://github.com/Mintplex-Labs/anything-llm
- [Cherry Studio](https://cherry-ai.com/)  
  跨平台的 AI 客户端软件，支持 Windows、macOS 和 Linux 系统。 ​  
  源码: [Apache 2.0-Clause New or Revised] - https://github.com/CherryHQ/cherry-studio/blob/main/LICENSE
