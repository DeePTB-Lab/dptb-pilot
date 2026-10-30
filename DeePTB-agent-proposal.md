# 开发DeePTB-agent项目计划书
项目名称：`DeePTB-agent`
## 一、项目背景
### 1.1 项目背景描述
  `DeePTB`方法是一套基于深度学习的电子哈密顿量高效建模工具，包含SK经验TB模型，E3 等变哈密顿模型。支持高效电子结构计算，以及结合dpnegf 实现器件量子输运模拟。但是目前`DeePTB`方法软件入门和使用门槛较高，特别是`DeePTB-SK`部分有较高的学习成本。
  在AI辅助科研的时代下，开发软件智能体成为了主要趋势。本项目旨在开发一套以`DeePTB`软件为核心，高效智能的agent，实现自动化的模型训练、测试以及性质计算。通过对`DeePTB-agent`的开发，可以使更多的用户尝试使用`DeePTB`解决科学问题。
### 1.2 项目目标
  开发方便可用的科研智能体，提供生成、修改输入文件的功能，通过`dpdispatcher`、`dflow`等远程任务流软件包实现远程自动提交任务、测试模型以及后处理等。尽可能实现用户需要的各种操作。
## 二、项目内容
### 2.1 软件开发
`DeePTB-agent`的新程序实现为`DeePTB-Pilot`。其使用`mcp-tools`和`FastAPI`构建提供工具和接口的后端，使用`React`构建交互性良好的前端，给用户方便舒适的体验。
#### 工具后端
| dptb_pilot/tools
基于mcp协议、通过sse等通讯方式提供大语言模型自动化工具，其中将提供许多`DeePTB`的自动化功能，同时提供guardrail自动截断目标工具。

##### 预计功能列表
1. `generate_deeptb_e3_training_config`, `generate_deeptb_sk_training_config` 生成DeePTB-E3和SK的训练输入文件
2. `submit_train` 提交训练任务
3. `train_report` 生成训练报告
4. `band_with_baseline_model` 使用基准模型直接预测结构能带
5. `generate_sk_baseline_model` 根据基准模型自动生成DeePTB-SK模型
6. `sk_test_report` 对SK模型进行测试并输出测试报告
7. `e3_test_report` 对E3模型进行测试并输出测试报告
8. `band_predict` 使用模型对能带进行预测
9. `band_with_sk_model` 使用DeePTB-SK模型预测结构能带

#### 接口后端
| dptb_pilot/server/app.py
基于`FastAPI`提供的与前端交互的接口，并通过`google-adk`实现与大语言模型通信，提供LLM通信、登录与工作区读取、光子收费等功能。

#### 集成前端
| dptb_pilot/web_ui
基于`React`框架制作集成式前端。通过url读取`mcp_tools`，通过`gradio`库生成可交互式的HTML前端，自动生成用户专属Agent，并与`LlmAgent`进行交互。

##### 关键技术目标
1. 每个用户单独的工作空间，并保存历史聊天记录和文件，提供文件的上传下载和预览、分子结构的可视化。
2. 与后端guardrail联合，在执行工具函数时提前发送给用户手动调节参数，过程去黑盒化。
3. 提供多种示例，并且提供不同的执行模式，对于高通量计算提供提交为Bohrium任务的形式。

### 2.2 软件部署
DeePTB-Pilot 将面向不同用户群体提供 本地化部署 与 集中式服务部署 两种运行形态，以兼顾科研开发灵活性与大规模用户使用的稳定性。

#### 本地部署模式
本地部署主要面向具有一定计算与开发经验的科研用户，适用于以下场景：
- 个性化修改 DeePTB 输入模板、训练流程与后处理逻辑；
- 在本地或私有集群上调试新模型、新算符或新工作流；
- 作为二次开发与算法验证平台。
本地部署模式下，用户可完整使用：
- DeePTB-Pilot 后端工具（MCP Server）
- Agent 推理与工具调用逻辑
- 本地或私有 HPC 资源的 dpdispatcher / dflow 调度能力
#### 服务器部署模式
服务器部署模式由开发者统一维护，面向更广泛的用户群体提供即开即用的服务：
- 用户通过 Web 前端直接创建并使用个人专属 DeePTB Agent；
- 统一管理算力资源、任务队列与模型缓存；
- 屏蔽底层复杂配置，降低 DeePTB 的使用门槛。
服务器版本的 DeePTB-Pilot 将作为 Bohrium App 进行部署与发布，充分利用 Bohrium 在任务调度、资源管理与计费方面的成熟生态。

## 三、技术架构与实现方案
### 3.1 系统架构图
本项目采用 基于 MCP（Model Context Protocol）的科研智能体架构，整体系统由四个逻辑层次构成，各层职责清晰、解耦良好：
（1）交互层（Frontend）
基于`React`构建的 Web 界面；
负责用户指令输入、Agent 对话展示、任务状态与结果可视化；
提供文件管理、结构预览、能带结果展示等科研常用功能。
（2）接口层（API）
基于`FastAPI`构建的API接口，提供基于`google-adk`构建的`LLM Agent`；
负责解析用户自然语言需求，将其转化为可执行的 DeePTB 工作流；
动态规划多步任务，按需调用 MCP 工具，并维护上下文状态。
（3）工具层（Tool Layer）
由`DeePTB-Pilot/tools`提供的`MCP Server`；
将`DeePTB`的关键操作封装为原子化、可组合的工具函数，包括：
- 输入文件生成
- 训练与测试提交
- 性质预测与结果分析
- 内置 guardrail 机制，避免高风险或不可控操作。
（4）执行层（Execution Layer）
具体计算执行环境，包括：
- 本地计算资源
- 通过 dpdispatcher / dflow 管理的远程 HPC 与超算集群
负责实际的 DeePTB 训练、测试、能带与输运相关计算任务。
### 3.2 核心技术栈
- 智能体框架：Google ADK
- 通信协议：MCP（Model Context Protocol），SSE
- 前端框架：React
- 后端服务：FastAPI
- 任务调度与工作流：dpdispatcher，dflow
- 科学计算核心：DeePTB
- 后端与工具开发语言：Python
## 四、成功标准与验收指标
- 功能完整性：计划书中列出的所有`DeePTB-Pilot`功能全部实现并可稳定运行。
- 用户体验：非DeePTB专家用户能够通过`DeePTB-Pilot`前端，在30分钟内成功完成一次从数据准备到模型训练和基础性质预测的全流程操作。
- 性能指标：与手动操作相比，使用Agent能将DeePTB的常规工作流准备和提交时间减少70%以上。
- 部署可用性：成功在Bohrium App平台完成部署，并至少支持5个并发用户稳定使用。

## 五、项目仓库链接
`DeePTB-Pilot`: https://github.com/Lonya0/dptb-pilot.git
旧仓库链接：
`DeePTB-agent-tools`: https://github.com/Lonya0/DeePTB-agent-tools
`BetterAIM`: https://github.com/Lonya0/BetterAIM