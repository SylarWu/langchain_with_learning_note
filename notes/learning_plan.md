# LangChain Agent 源码学习路线图

> 目标：按代码依赖顺序，逐层深入理解 LangChain 的 Agent 体系——从核心抽象、经典实现到新架构（LangGraph + Middleware）。

---

## Phase 0: 前置知识与项目结构

**文件导航**

```
libs/core/langchain_core/              # 基础抽象（Schema、Runnable、Messages）
libs/langchain/langchain_classic/      # 经典 Agent 实现（AgentExecutor）
libs/langchain_v1/langchain/           # 新架构（create_agent + LangGraph）
```

**推荐阅读**

- `CLAUDE.md` 中的项目架构说明
- 官方文档 Agents 章节（docs.langchain.com/oss/python/langchain/agents）

---

## Phase 1: 核心 Schema（ langchain-core ）

Agent 的本质是 **Action → Observation → Thought** 的循环。LangChain 用三个核心类来表达这一过程。

### 1.1 AgentAction / AgentFinish / AgentStep

**文件**: `libs/core/langchain_core/agents.py`

- `AgentAction`: LLM 决定调用某个 tool，包含 `tool`, `tool_input`, `log`
- `AgentActionMessageLog`: 增加了 `message_log`，用于 ChatModel 场景
- `AgentStep`: `(AgentAction, observation)` 的封装，代表一次完整的"行动+观察"
- `AgentFinish`: 终止条件，包含 `return_values` 和 `log`

**学习要点**

1. 三个类都继承自 `Serializable`，理解序列化机制
2. `messages` property: 如何将 action/step/finish 转换成 `BaseMessage`（AIMessage / HumanMessage / FunctionMessage）
3. `_convert_agent_action_to_messages()` 和 `_convert_agent_observation_to_messages()`: 这是重构对话历史的关键逻辑

**代码练习**: 手写一个 `AgentAction` 和 `AgentStep`，观察它们的 `messages` property 输出。

---

## Phase 2: 经典 Agent 抽象层（ langchain-classic ）

### 2.1 Agent 基类

**文件**: `libs/langchain/langchain_classic/agents/agent.py`

从第 55 行开始阅读：

- `BaseSingleActionAgent`: 单动作 Agent 抽象
  - `plan()`: 核心决策方法，输入 `intermediate_steps`，输出 `AgentAction | AgentFinish`
  - `aplan()`: 异步版本
  - `return_stopped_response()`: 达到 max_iterations 时的兜底策略
- `BaseMultiActionAgent`: 多动作 Agent 抽象（一次调用多个 tool）

**学习要点**

1. `intermediate_steps: list[tuple[AgentAction, str]]` 是 Agent 的"记忆"——所有历史行动和观察
2. `get_allowed_tools()`: Agent 可以声明自己只能使用哪些 tool

### 2.2 Agent 的具体实现

**文件**: `libs/langchain/langchain_classic/agents/agent.py`（第 704 行起）

- `Agent` 类：基于 `LLMChain` 的通用实现
  - `llm_chain`: 驱动决策的 LLMChain
  - `output_parser`: 解析 LLM 输出为 `AgentAction` 或 `AgentFinish`
  - `_construct_scratchpad()`: 将 `intermediate_steps` 格式化为 prompt 中的 `agent_scratchpad`
  - `plan()`: 调用 `llm_chain.predict()` → `output_parser.parse()`

### 2.3 AgentExecutor——执行引擎

**文件**: `libs/langchain/langchain_classic/agents/agent.py`（第 1012 行起）

`AgentExecutor` 继承自 `Chain`，是经典架构的核心执行器。

**关键属性**

- `agent`: 决策 agent
- `tools`: Agent 可调用的工具集
- `max_iterations`: 最大循环次数（默认 15）
- `max_execution_time`: 最大执行时间
- `early_stopping_method`: `"force"` 或 `"generate"`
- `handle_parsing_errors`: 输出解析失败时的处理策略
- `trim_intermediate_steps`: 控制历史记忆的截断

**关键方法**

- `_should_continue(iterations, time_elapsed)`: 判断是否继续循环
- `_take_next_step()`: 执行一次完整的"决策 → 调用 tool → 得到 observation"循环
- `_iter_next_step()`: 生成器版本，yield 每一步的 action 和 observation

### 2.4 AgentExecutorIterator——逐步迭代

**文件**: `libs/langchain/langchain_classic/agents/agent_iterator.py`

- `__iter__()`: 同步迭代器，逐个 yield `AgentAction` 和 observation
- `__aiter__()`: 异步迭代器
- `reset()`: 重置状态（`intermediate_steps`, `iterations`, `time_elapsed`）
- `make_final_outputs()`: 组装最终输出

**学习要点**

1. 迭代器模式如何让你"看到" Agent 思考的每一步
2. `yield_actions=True` 时可以在 observation 之前先 yield action

---

## Phase 3: 经典 Agent 的具体类型

### 3.1 MRKL / ZeroShotAgent（ReAct 风格）

**文件**: `libs/langchain/langchain_classic/agents/mrkl/base.py`

MRKL（Modular Reasoning, Knowledge and Language）是 LangChain 最经典的 Agent。

- `ZeroShotAgent`: 零样本 ReAct Agent
- `create_prompt()`: 组合 `PREFIX` + `tool descriptions` + `FORMAT_INSTRUCTIONS` + `SUFFIX`
- `_validate_tools()`: 验证 tool 是否为单输入

**Prompt 模板**: `libs/langchain/langchain_classic/agents/mrkl/prompt.py`

### 3.2 ReAct Docstore Agent

**文件**: `libs/langchain/langchain_classic/agents/react/base.py`

基于论文 [ReAct](https://arxiv.org/pdf/2210.03629.pdf) 的实现，用于文档检索场景。

- `ReActDocstoreAgent`
- 专用 prompt: `WIKI_PROMPT`, `TEXTWORLD_PROMPT`
- tool 要求: 必须有 `Search` 和 `Lookup`

### 3.3 Self-Ask With Search

**文件**: `libs/langchain/langchain_classic/agents/self_ask_with_search/base.py`

基于论文 [Self-Ask](https://arxiv.org/abs/2210.03350) 的实现。

- `SelfAskWithSearchAgent`
- 核心思想：将复杂问题拆解为子问题，逐个搜索回答
- 必须且只能有一个 tool

### 3.4 Structured Chat Agent

**文件**: `libs/langchain/langchain_classic/agents/structured_chat/base.py`

支持多参数 tool 的 Agent。

### 3.5 Tool Calling Agent（现代推荐）

**文件**: `libs/langchain/langchain_classic/agents/tool_calling_agent/base.py`

```python
def create_tool_calling_agent(llm, tools, prompt):
    """创建一个使用原生 tool calling API 的 Agent。"""
```

**学习要点**

1. 这是从"文本解析"到"原生 API"的演进
2. 使用 `bind_tools()` 而非 prompt engineering
3. `format_to_tool_messages()`: 将 intermediate steps 格式化为 message 列表

---

## Phase 4: Output Parser——理解 LLM 输出

Agent 的 Output Parser 负责将 LLM 的原始输出解析为 `AgentAction` 或 `AgentFinish`。

### 4.1 ReAct Single Input Parser

**文件**: `libs/langchain/langchain_classic/agents/output_parsers/react_single_input.py`

- 解析格式：
  ```
  Thought: ...
  Action: search
  Action Input: what is the weather?
  ```
- 或终止格式：
  ```
  Thought: ...
  Final Answer: The weather is sunny.
  ```

### 4.2 JSON Parser

**文件**: `libs/langchain/langchain_classic/agents/output_parsers/json.py`

### 4.3 OpenAI Functions / Tools Parser

**文件**:

- `libs/langchain/langchain_classic/agents/output_parsers/openai_functions.py`
- `libs/langchain/langchain_classic/agents/output_parsers/openai_tools.py`

### 4.4 XML Parser

**文件**: `libs/langchain/langchain_classic/agents/output_parsers/xml.py`

**学习要点**

1. 对比不同 Parser 的设计差异：正则表达式 vs JSON vs XML
2. `OutputParserException`: 解析失败时的异常处理
3. `get_format_instructions()`: 如何让 prompt 包含格式说明

---

## Phase 5: Prompt 与 Scratchpad 格式化

### 5.1 Scratchpad 构建

**文件**: `libs/langchain/langchain_classic/agents/schema.py`

- `AgentScratchPadChatPromptTemplate`: 专门为 agent scratchpad 设计的 prompt template
- `_construct_agent_scratchpad()`: 将 `intermediate_steps` 拼接成 thought + observation 字符串

### 5.2 Tool 格式化

**文件**: `libs/langchain/langchain_classic/agents/format_scratchpad/`

- `format_to_tool_messages()`: 将历史步骤转为 message 列表（用于 tool calling agent）
- `format_to_openai_function_messages()`: OpenAI function calling 专用格式

---

## Phase 6: 新架构——LangGraph + Middleware（ langchain v1 ）

这是 LangChain 的下一代 Agent 架构，基于 [LangGraph](https://langchain-ai.github.io/langgraph/) 构建。

### 6.1 入口：create_agent

**文件**: `libs/langchain_v1/langchain/agents/factory.py`（第 696 行起）

```python
def create_agent(
    model: str | BaseChatModel,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    response_format: ResponseFormat | type | dict | None = None,
    ...
) -> CompiledStateGraph:
```

**核心流程**（按代码顺序阅读）

1. **初始化 model**: `init_chat_model(model)` 将字符串转为 ChatModel 实例
2. **处理 response_format**: 将原始 schema 包装为 `AutoStrategy`，后续自动检测最优策略
3. **构建 ToolNode**: `langgraph.prebuilt.tool_node.ToolNode`，执行 tool 调用
4. **收集 middleware hooks**: `before_agent`, `before_model`, `after_model`, `after_agent`, `wrap_model_call`, `wrap_tool_call`
5. **Compose handlers**: 用洋葱模型将多个 middleware 的 handler 链式组合
6. **构建 StateGraph**:
   - `model` node: 调用 LLM
   - `tools` node: 执行 tool（如果有 tool_calls）
   - middleware nodes: 每个 middleware 的 hook 都注册为独立 node
7. **编译图**: `graph.compile()`

### 6.2 Model Request / Response 类型

**文件**: `libs/langchain_v1/langchain/agents/middleware/types.py`（第 89 行起）

- `ModelRequest`: 封装对 LLM 的调用请求
  - `model`, `messages`, `tools`, `system_message`, `response_format`, `state`, `runtime`
- `ModelResponse`: 封装 LLM 的返回
  - `result: list[AIMessage]`, `structured_response`
- `ExtendedModelResponse`: 增加 `command` 字段，允许 middleware 控制图跳转

### 6.3 AgentState

**文件**: `libs/langchain_v1/langchain/agents/middleware/types.py`

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    structured_response: Any | None
```

- `messages`: 完整的对话历史，使用 `add_messages` reducer 自动合并
- `structured_response`: structured output 的结果

### 6.4 Middleware 基类

**文件**: `libs/langchain_v1/langchain/agents/middleware/types.py`

```python
class AgentMiddleware(Protocol):
    def before_agent(self, state: AgentState) -> dict | Command: ...
    def before_model(self, state: AgentState) -> dict | Command: ...
    def wrap_model_call(self, request: ModelRequest, handler: Callable) -> ModelResponse: ...
    def after_model(self, state: AgentState) -> dict | Command: ...
    def after_agent(self, state: AgentState) -> dict | Command: ...
    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable) -> ToolMessage: ...
```

**Hook 执行顺序**

```
before_agent → before_model → wrap_model_call → [LLM invoke] → after_model → tools → after_agent
```

**学习要点**

1. `Command` 对象：可以控制图跳转到指定 node（`jump_to`）
2. `wrap_model_call` 是洋葱模型：middleware 可以包装/拦截/修改 model 调用
3. `wrap_tool_call` 同理，用于拦截 tool 执行

### 6.5 内置 Middleware 示例

按复杂度递增阅读：

| Middleware | 文件 | 功能 |
|-----------|------|------|
| `ModelRetryMiddleware` | `middleware/model_retry.py` | model 调用失败时自动重试 |
| `ToolRetryMiddleware` | `middleware/tool_retry.py` | tool 调用失败时自动重试 |
| `ModelFallbackMiddleware` | `middleware/model_fallback.py` | model 失败时切换到备用 model |
| `ModelCallLimitMiddleware` | `middleware/model_call_limit.py` | 限制 model 调用次数 |
| `ToolCallLimitMiddleware` | `middleware/tool_call_limit.py` | 限制 tool 调用次数 |
| `HumanInTheLoopMiddleware` | `middleware/human_in_the_loop.py` | 人工审批关键操作 |
| `SummarizationMiddleware` | `middleware/summarization.py` | 自动摘要控制上下文长度 |
| `PIIMiddleware` | `middleware/pii.py` | PII 数据脱敏 |
| `ShellToolMiddleware` | `middleware/shell_tool.py` + `_execution.py` | 安全 shell 执行策略 |

### 6.6 Structured Output 体系

**文件**: `libs/langchain_v1/langchain/agents/structured_output.py`

- `ToolStrategy`: 使用 tool calling 实现 structured output（添加一个特殊的 output tool）
- `ProviderStrategy`: 使用 provider 原生 API（如 OpenAI 的 `response_format`）
- `AutoStrategy`: 自动检测 model 能力，选择最优策略
- `OutputToolBinding`: 将 Pydantic/dataclass/TypedDict 转为 tool binding

**代码流程**（`factory.py` 中）

1. 用户传入 `response_format` → 包装为 `AutoStrategy`
2. `_get_bound_model()` 中检测 model 是否支持 provider strategy
3. 支持 → `ProviderStrategy`；不支持 → `ToolStrategy`
4. `_handle_model_output()` 解析输出，提取 `structured_response`

---

## Phase 7: 图执行与状态流转

### 7.1 Model Node

**文件**: `libs/langchain_v1/langchain/agents/factory.py`（第 1316 行起）

```python
def model_node(state, runtime) -> list[Command]:
    request = ModelRequest(model=model, tools=default_tools, ...)
    if wrap_model_call_handler:
        result = wrap_model_call_handler(request, _execute_model_sync)
    else:
        result = _execute_model_sync(request)
    return _build_commands(result)
```

### 7.2 Tools Node

使用 LangGraph 的 `ToolNode`：

```python
if tool_node is not None:
    graph.add_node("tools", tool_node)
```

### 7.3 Edge 定义

```python
graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, ["tools", END])
if tool_node:
    graph.add_edge("tools", "model")
```

---

## Phase 8: 测试驱动的学习

通过阅读测试代码理解使用方式：

**文件**

- `libs/langchain_v1/tests/unit_tests/agents/test_react_agent.py`
- `libs/langchain_v1/tests/unit_tests/agents/test_create_agent_tool_validation.py`
- `libs/langchain_v1/tests/unit_tests/agents/test_return_direct_graph.py`
- `libs/langchain_v1/tests/unit_tests/agents/test_system_message.py`
- `libs/langchain_v1/tests/unit_tests/agents/test_agent_streaming.py`

**经典 Agent 测试**

- `libs/langchain/tests/unit_tests/agents/`

---

## 推荐学习顺序（每日计划建议）

| 天数 | 主题 | 重点文件 |
|-----|------|---------|
| Day 1 | 核心 Schema | `libs/core/langchain_core/agents.py` |
| Day 2 | 经典 Agent 抽象 | `libs/langchain/langchain_classic/agents/agent.py`（前 200 行） |
| Day 3 | AgentExecutor 执行循环 | `libs/langchain/langchain_classic/agents/agent.py`（第 1012 行起） |
| Day 4 | AgentExecutorIterator | `libs/langchain/langchain_classic/agents/agent_iterator.py` |
| Day 5 | MRKL / ZeroShotAgent | `agents/mrkl/`, `agents/react/` |
| Day 6 | Output Parsers | `agents/output_parsers/` |
| Day 7 | Tool Calling Agent（经典） | `agents/tool_calling_agent/base.py` |
| Day 8 | 新架构入口 create_agent | `libs/langchain_v1/langchain/agents/factory.py`（函数签名 + 参数解析） |
| Day 9 | Middleware 类型与协议 | `libs/langchain_v1/langchain/agents/middleware/types.py` |
| Day 10 | 图构建逻辑 | `factory.py`（第 1046 行起：StateGraph 构建） |
| Day 11 | Model Node 执行 | `factory.py`（`_execute_model_sync`, `_get_bound_model`） |
| Day 12 | Structured Output | `structured_output.py` + `factory.py` 相关逻辑 |
| Day 13 | 内置 Middleware（上） | `model_retry.py`, `model_fallback.py`, `tool_retry.py` |
| Day 14 | 内置 Middleware（下） | `summarization.py`, `pii.py`, `human_in_the_loop.py` |
| Day 15 | 实战：写一个自定义 Middleware | 参考 `middleware/types.py` 的 Protocol |

---

## 关键概念对照表

| 经典架构（langchain-classic） | 新架构（langchain v1） |
|---------------------------|---------------------|
| `AgentExecutor` | `CompiledStateGraph`（由 `create_agent` 返回） |
| `intermediate_steps` | `state["messages"]` |
| `plan()` / `aplan()` | `model_node` + `tools_node` 的循环 |
| `output_parser` | `bind_tools()` + `ToolNode`（原生 API） |
| `AgentScratchPadChatPromptTemplate` | `add_messages` reducer 自动管理 |
| `max_iterations` | `ModelCallLimitMiddleware` / `ToolCallLimitMiddleware` |
| `handle_parsing_errors` | middleware 的 `wrap_model_call` 中捕获 |
| `early_stopping_method` | middleware 的 `Command` 跳转控制 |

---

## 思考题

1. 经典架构中，`AgentExecutor` 的 `_take_next_step` 和 `AgentExecutorIterator` 的 `__next__` 有什么关系？为什么要拆成两个类？
2. `ToolCallingAgent` 相比 `ZeroShotAgent` 的优势是什么？为什么前者不需要复杂的 output parser？
3. 新架构中，`wrap_model_call` 的洋葱模型如何实现？如果两个 middleware 都修改了 `request.tools`，最终生效的是谁？
4. `AutoStrategy` 如何决定使用 `ProviderStrategy` 还是 `ToolStrategy`？这个检测逻辑在代码的哪个位置？
5. 如果想在 Agent 每次调用 tool 前加一个"人工确认"，应该在 classic 架构和 v1 架构中分别如何实现？

---

> **提示**: 每读完一个文件，建议用 `git blame` 或 `git log --oneline -10 <file>` 看看最近的改动，了解该模块的演进方向。
