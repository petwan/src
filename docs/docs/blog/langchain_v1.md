---
title: ⚡理解 LangChain 1.0 的工作流
date: 2026-01-05
tags: [LLMs]
description: 一篇 LangChain v1.0 的入门介绍，基于 Google 的 Pergel 模式，以一个例子说明整体的执行过程，粗略介绍中间件的一些基本概念。
draft: false
---

# ⚡理解 LangChain 1.0 的工作流

> 这是一篇入门的文章，仅用于学习，企业级方案不在这个文章里介绍。

AI 应用的开发正变得越来越复杂，单纯调用一次 API 往往解决不了问题。我们需要让 AI 像侦探一样思考，像团队一样协作，执行多步骤、有状态的工作流。

这个文章通过一个 **“破案小组”** 的比喻，来轻松理解 LangChain 中强大的工作流引擎——**LangGraph** 的核心思想。

> 💡 本文基于 LangChain 1.0，`pip install -U langchain` 。

## 1. 核心比喻：一个高效的破案小组

想象一个重大案件，警局成立了一个专项小组：

*   **👮‍♂️ 侦查员 A**：负责梳理嫌疑人的时间线。
*   **👩‍💻 情报员 B**：负责调取和分析监控、通讯记录。
*   **👨‍⚕️ 心理专家 C**：负责分析嫌疑人的行为与心理状态。
*   **👨‍⚖️ 法律顾问 D**：负责评估证据是否构成完整的法律证据链。

他们的工作都围绕**同一个嫌疑人**展开。为了高效协作，他们共享一份**唯一的案件档案**，里面记录了：
1.  目前已掌握的所有证据
2.  嫌疑人的供述
3.  监控/通话记录的关键信息
4.  哪些线索已得到验证
5.  哪些疑点仍需追查

**你作为组长**，负责协调这4名成员。工作按**轮次**进行，每轮中，每位成员只处理自己专业领域的任务，并通过更新**案件档案**来交换信息，最终共同破案。

### 1.1 第一轮侦查
组长下达初始指令，小组成员开始工作并记录发现：
*   **侦查员 A**：梳理时间线 → 发现一处矛盾点 → **写入档案**。
*   **情报员 B**：调取案发地监控 → 发现关键身影 → **写入档案**。
*   **心理专家 C**：观察初步行为 → 记录“有隐瞒可能性” → **写入档案**。
*   **法律顾问 D**：材料不足，暂不行动。

**第一轮结束**：组长汇总所有人的记录，**统一更新案件档案**。

### 1.2 第二轮分析
组长基于更新后的档案，再次分派针对性的任务：
*   **侦查员 A**：结合B提供的监控时间，**修正**时间线。
*   **情报员 B**：根据A修正的时间，调取对应时段的手机通话记录。
*   **心理专家 C**：发现供述与监控时间矛盾，标记“建议进行压力测试”。
*   **法律顾问 D**：开始审视当前证据的关联性。

**第二轮结束**：档案再次被**统一更新**，线索更清晰，矛盾更突出。

### 1.3 第三轮定案
组长发现，证据链已完整闭合，所有供述一致，没有成员提出新的疑点或需要补充的信息。
**案件宣告侦破，工作流结束。**

## 2. 从“破案”到“LangGraph”：核心概念映射

这个生动的例子，完美映射了 LangGraph 的四大核心抽象：

| 破案小组                 | LangGraph 概念                         | 作用                                                |
| :----------------------- | :------------------------------------- | :-------------------------------------------------- |
| **案件档案**             | **State (状态)**                       | 工作流中共享、随时间变化的核心数据。                |
| **小组成员(A, B, C, D)** | **Node (节点)**                        | 执行具体任务的单元（可以是函数、LLM调用、工具等）。 |
| **组长 (你)**            | **Graph (图) / Orchestrator (编排器)** | 定义节点执行顺序和逻辑的蓝图。                      |
| **“轮次”工作模式**       | **Stateful Workflow (有状态工作流)**   | 状态在一轮轮执行中传递和演化，直至达到终点。        |

### 2.1 两个关键的技术挑战与 LangGraph 的解决方案

我们的“破案”流程虽然清晰，但也引出了两个潜在问题：

1.  **档案更新冲突**：如果A和B同时修改了档案的同一部分怎么办？
2.  **流程中断与回溯**：如果中途需要暂停审议，或者想查看之前的推理步骤怎么办？

**LangGraph 提供了优雅的解决方案**：
*   **状态更新策略 (State Reducers)**：精确定义每个信息字段的合并逻辑（如覆盖、追加等），解决冲突。
*   **检查点 (Checkpoints)**：自动保存每一轮结束后的完整状态，实现暂停、恢复和步骤追溯。

## 3. LangGraph 实战：如何构建你的“破案引擎”

理解了比喻，我们来看看代码。构建一个 LangGraph 工作流，就像组建那个破案小组。

### 3.1 定义“案件档案” (State Schema) 和 背景（Context Schema）
首先，我们需要用代码定义我们的“案件档案”里具体记录什么。这里我们用 `TypedDict` 来声明。

```python
from typing import TypedDict, List, Annotated

# 定义一个自定义的“合并策略”：将新日志追加到旧日志后面
def append_log(old_log: List[str], new_log: List[str]) -> List[str]:
    return old_log + new_log

class CaseFileState(TypedDict):
    """
    我们的案件档案 State。
    - `clues`: 默认为“覆盖”策略，新线索列表直接替换旧的。
    - `investigation_log`: 使用“追加”策略，所有分析记录都会保留。
    """
    clues: List[str]  # 线索列表，默认更新策略为“覆盖”
    investigation_log: Annotated[List[str], append_log] # 调查日志，使用“追加”策略

# ======= Context Schema: 不变的"案件背景" =========
class CaseContext(TypedDict):
    """
    案件背景 Context - 固定不变的信息
    - `case_id`: 案件编号，用于标识
    - `jurisdiction`: 法律管辖区域，决定适用的法律
    - `priority_level`: 优先级，可能影响资源分配（但不改变）
    """
    case_id: str
    jurisdiction: str
    priority_level: str
```

**关键点**：`Annotated` 和 `append_log` 函数让我们能精细控制 `investigation_log` 字段的更新方式，这正是 LangGraph 灵活性的体现。

::: info State更新策略
上面的例子中，State更新策略是覆盖，即每次运行节点时，都会将节点的输出结果覆盖到 State 中。
LangGraph 允许为每个状态字段指定 reducer 函数，控制如何合并新值和旧值，通过 TypedDict + Annotated 实现。
- 策略 1：覆盖（Replace） —— 默认就是如此（但可显式声明）
- 策略 2：累加（Accumulate / Append） —— 适用于列表等
- 策略 3：最大值/最小值/自定义逻辑
:::


> 💡 小组成员（Node）运行时，可以查看 Context 对应的内容，但是不能修改。

除了 `State Schema` 和 `Context Schema`， LangGraph 还需要一个额外的概念：**Input Schema** 和 **Output Schema**。

- `Input Schema` 是一开始必须带进系统的信息内容形式，如果没有给定Input Schema，StateGraph默认把State Scheam 作为 Input Schema。

- `Output Schema` 是 Node 运行过程中产生的内容形式，如果没有给定Output Schema，StateGraph默认把State Scheam 作为 Output Schema。


### 3.2 招募“小组成员” (定义 Nodes)
每个节点都是一个普通的 Python 函数，它接收当前的 `State`，并返回要更新到 `State` 中的内容。

```python
def detective_node(state: CaseFileState) -> dict:
    """侦查员节点：发现新线索"""
    new_clue = "嫌疑人在案发时间声称在家，但无证人。"
    return {
        "clues": [new_clue], # 更新clues字段
        "investigation_log": [f"[侦查员] 发现了线索：{new_clue}"]
    }

def analyst_node(state: CaseFileState) -> dict:
    """情报员节点：分析线索并记录"""
    current_clues = state["clues"]
    analysis = f"目前共有 {len(current_clues)} 条线索需要交叉验证。"
    return {
        "investigation_log": [f"[情报员] 分析报告：{analysis}"]
        # 没有返回 `clues`，所以 `clues` 字段将保持不变
    }
```

### 3.3 任命“组长”并制定流程 (构建 Graph)
现在，我们创建图（组长），添加节点（成员），并安排他们的工作顺序（边）。

```python
from langgraph.graph import StateGraph, START, END

# 1. 创建一个图，并告诉它我们档案的格式 (CaseFileState)
workflow_builder = StateGraph(state_schema=CaseFileState, context_schema=CaseContext)

# 2. 将我们的“小组成员”（节点函数）添加到图中
workflow_builder.add_node("detective", detective_node)
workflow_builder.add_node("analyst", analyst_node)

# 3. 制定工作流程
workflow_builder.add_edge(START, "detective")  # 开始 -> 先让侦查员上
workflow_builder.add_edge("detective", "analyst")  # 侦查员完成后 -> 情报员分析
workflow_builder.add_edge("analyst", END)  # 分析完成后 -> 结束

# 4. 编译成可执行的“工作流引擎”
investigation_workflow = workflow_builder.compile()

for key in workflow_builder.__dict__.keys():
    print(f"{key} >| {workflow_builder.__dict__[key]}")
```

### 3.4 启动调查！
现在，我们可以用一个初始的“空档案”来启动这个工作流。

```python
initial_state: CaseFileState = {"clues": [], "investigation_log": ["案件启动"]}

# 执行工作流！
final_state = investigation_workflow.invoke(initial_state)

print("最终线索：", final_state["clues"])
print("\n完整调查日志：")
for log in final_state["investigation_log"]:
    print(" -", log)
```

**输出将会是：**
```
最终线索： ['嫌疑人在案发时间声称在家，但无证人。']

完整调查日志：
 - 案件启动
 - [侦查员] 发现了线索：嫌疑人在案发时间声称在家，但无证人。
 - [情报员] 分析报告：目前共有 1 条线索需要交叉验证。
```

看！日志被完美地**累积**了下来，而线索被**更新**了。这就是我们定义的 State Reducers 在起作用。

### 3.5 高级能力——流程“中断”与“存档”
现实破案中，组长可能需要中途喊停，让大家重新审视证据。LangGraph 通过 **`Checkpointer`** 和 **`Interrupts`** 支持这一场景。

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# 引入一个"档案管理员"（记忆存储器），负责保存每一轮的快照
memory = MemorySaver()

# 重新编译工作流，并指定在 ‘analyst’ 节点执行前必须中断，等待指令
workflow_builder = StateGraph(CaseFileState)
workflow_builder.add_node("detective", detective_node)
workflow_builder.add_node("analyst", analyst_node)
workflow_builder.add_edge(START, "detective")
workflow_builder.add_edge("detective", "analyst")
workflow_builder.add_edge("analyst", END)

# 关键：配置 checkpoint 和 interrupt
investigation_workflow = workflow_builder.compile(
    checkpointer=memory, interrupt_before=["analyst"]  # 指定在 ‘analyst’ 节点前中断
)

# 执行时，需要提供一个 thread_id，类似于"案件编号"
config = {"configurable": {"thread_id": "case-001"}}
initial_state = {"clues": [], "investigation_log": []}

# 第一轮执行，会在 analyst 执行前自动暂停
result = investigation_workflow.invoke(initial_state, config=config)
print("流程已暂停在 analyst 节点前。")
print("当前状态：", investigation_workflow.get_state(config).values)

# 模拟组长审阅后，决定继续
user_input = input("\n是否批准情报员开始分析？ (yes/no): ")
if user_input.lower() == "yes":
    # 传入 None 表示从当前中断处继续执行
    final_state = investigation_workflow.invoke(None, config=config)
    print("\n调查完成。最终状态：", final_state)
else:
    print("\n调查暂停。")
```

此外，你还可以随时查看这个案件（`thread_id`）的所有历史记录：
```python
for snapshot in investigation_workflow.get_state_history(config):
    print(“-” * 20)
    print(“步骤快照：”, snapshot.values)
    print(“下一步执行：”, snapshot.next)
```

> 💡 可以把它想象thread_id成会话 ID，但范围更广：它不仅限于用户聊天。它定义了 LangGraph 中一个完整的逻辑流程。如果更改此 ID thread_id，LangGraph 会启动一个全新的线程——不会记住之前的线程。

thread_id 可以实现：
1. 控制内存范围
2. 恢复已中断流程的执行
3. 可以设计互不冲突的并行任务

常见的 thread_id 的策略：
- Chat session: user-{id}-chat-{timestamp}
- Document task: file-{file_id}-run_{uuid}
- Learning agent: user-{id}-topic-{topic_id}
- Multi-agent: task-{tieket_id}

对于生产环境，使用由数据库作为checkpoints的存储
```bash
pip install langgraph-checkpoint-postgres
```

示例代码：
```python
from langchain.agents import create_agent

from langgraph.checkpoint.postgres import PostgresSaver  


DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    checkpointer.setup() # auto create tables in PostgresSql
    agent = create_agent(
        "gpt-5",
        tools=[get_user_info],
        checkpointer=checkpointer,  
    )
```

## 4. 进阶：构建可循环的工作流
LangGraph 不仅支持线性流程，还能轻松实现带循环的复杂工作流。

```python {61-63}
from typing import TypedDict, Literal
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START, END
from datetime import datetime


class Task(TypedDict):
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed"]
    created_at: str
    updated_at: str | None


class TaskManagerState(TypedDict):
    #  使用自定义 reducer：新任务列表直接替换旧列表
    tasks: Annotated[list[Task], lambda old, new: new]


def add_init_tasks(state):
    return {
        "tasks": [
            {
                "id": f"task_{i}",
                "description": f"Sample task {i}",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": None,
            }
            for i in range(1, 4)
        ]
    }


def process_next_task(state):
    tasks = state["tasks"]
    now = datetime.now().isoformat()
    for i, task in enumerate(tasks):
        if task["status"] == "pending":
            updated_task = {**task, "status": "completed", "updated_at": now}
            # 构造新任务列表（替换该任务）
            new_tasks = tasks[:i] + [updated_task] + tasks[i + 1 :]
            return {"tasks": new_tasks}
    return {}  # 无待办任务，不更新状态


def should_continue(state):
    """判断是否继续循环"""
    has_pending = any(t["status"] == "pending" for t in state["tasks"])
    return "process" if has_pending else END


# 构建状态图
builder = StateGraph(TaskManagerState)
builder.add_node("add", add_init_tasks)
builder.add_node("process", process_next_task)

builder.add_edge(START, "add")
builder.add_edge("add", "process")

builder.add_conditional_edges(
    "process", should_continue, ["process", END]  # 可能的下一节点：继续处理或结束
)
graph = builder.compile()

result = graph.invoke({"tasks": []})

for task in result["tasks"]:
    print(f"✅ {task['id']}: {task['description']} → {task['status']}")
```


## 5. create_agent
LangChain 给出的官方示例，提供了一个创建智能代理的函数：`create_agent`

```python
from httpx import request
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI  # 使用 OpenAI 兼容客户端
import requests
from langgraph.checkpoint.memory import MemorySaver


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    response = requests.get(f"https://wttr.in/{city}?format=j1")
    return response.json()


# 配置硅基流动的模型（例如 DeepSeek、Qwen、Llama 等）
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",  # 或其他 SiliconFlow 支持的模型
    api_key="api_key",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.0,
)
memory = MemorySaver()  # [!code highlight]

checkpointer = memory
config = {"configurable": {"thread_id": "test-thread"}}

agent = create_agent(
    model=llm,
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
    checkpointer=memory,
)

# Run the agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in Beijing"}]},
    config=config,
)
for snapshot in agent.get_state_history(config):
    print(snapshot)

```
::: warning
执行代码需要安装 langchain-openai 依赖，`pip install langchain-openai`
:::

> 📦 本质：`create_agent` 是 LangGraph 的高级封装——它自动构建了一个包含 “LLM → Tool → LLM” 循环的状态图，并内置了消息管理、工具路由和 checkpoint 支持。

我们还可以通过使用 `ToolStrategy` 来定义输出的结构，其实就是在调用LLM的时候，加入了格式的指导，同时对模型的输出（可能是比较好处理的结构，也可以不是很好处理的结构）进行解析，返回一个结构化的结果。
```python
from typing import Optional, List
from pydantic import BaseModel, Field
from pydantic import BaseModel
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI


class PartialContact(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    found_fields: List[str] = Field(default_factory=list)


# 允许部分提取
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",  # 或其他 SiliconFlow 支持的模型
    api_key="your api key",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.0,
)


agent = create_agent(
    model=llm,
    system_prompt="You are a helpful assistant",
    response_format=ToolStrategy(PartialContact),
)

# 调用代理
result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "提取联系信息：张三，zhangsan@example.com，13800138000",
            }
        ]
    }
)

# 获取结构化响应
contact = result["structured_response"]
print(contact)
# ContactInfo(name='张三', email='zhangsan@example.com', phone='13800138000')

print(result["messages"][-1])
```

> 💡 这里就需要注意，因为LLM的输出不确定，所以最终的结构化结果如果不符合预期，部分原因可能在LLM的输出，另一部分原因可能在 `ToolStrategy` 对结果的解析，`ToolStrategy`的结果被保存在了 `structured_response` 的 key 中。

除了 `ToolStrategy` 之外，还有 `AutoToolStrategy` 以及 `ProviderStrategy`。

- ProviderStrategy 是针对不同模型提供商（如 OpenAI、Anthropic、Google 等）的特性进行优化的策略。不同的模型提供商可能有不同的结构化输出方式，ProviderStrategy 会利用各提供商的原生特性。

- AutoToolStrategy 是一个自动选择 ToolStrategy 的策略。它会自动选择最合适的 ToolStrategy，并使用它来处理问题。

## 6. 多轮问答机器人
从上面的例子中，可以看到，model 节点实际上是一个比较特殊的Node而已（在LangChain的设计中，这些可以运行的Node都属于 Runnable ），它接受一个消息列表（`List[BaseMessage]`），输出一个 `AIMessage`。而用户输入也可以被封装为 `HumanMessage` —— 这意味着，“用户”完全可以被视为一个特殊的“工具”或“外部节点”，与 LLM 在状态图中交替交互。

下面的例子完全去掉了builder，直接使用 while 循环实现多轮问答：
```python
from httpx import request
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI  # 使用 OpenAI 兼容客户端
from langchain.messages import SystemMessage, AIMessage, HumanMessage

# 配置硅基流动的模型（例如 DeepSeek、Qwen、Llama 等）
model = ChatOpenAI(
    model="Qwen/Qwen3-8B",  # 或其他 SiliconFlow 支持的模型
    api_key="your api key",
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.0,
)

system_message = SystemMessage(
    content="你叫小花，是一名乐于助人的智能助手，请在对话中保持友好的态度。"
)

messages = [system_message]

while True:
    user_input = input("用户：")
    if user_input.lower() in {"exit", "quit"}:
        print("结束对话")
        break

    # 追加用户消息
    messages.append(HumanMessage(content=user_input))

    # 获取模型生成的回复
    print("模型：", end="", flush=True)
    full_reply = ""

    for chunk in model.stream(messages):
        if chunk.content:
            print(chunk.content, end="", flush=True)
            full_reply += chunk.content

    print("\n" + "-" * 40)

    messages.append(AIMessage(content=full_reply))

    # 仅保留最新50条消息
    messages = messages[-50:]
```

> 💡 执行前记得更新 api key

## 7. 中间件
官方参考文档：[内置的中间件链接](https://docs.langchain.com/oss/python/langchain/middleware/built-in)

<Image 
src='./assets/langchain_middleware.png'
width='50%'
/>

### 7.1 动态模型
在学习到这里的时候，我们应该对 LangChain 的基本概念有了一定的了解，其整体的链路应该是ReAct模式，如下图所示：

<Image
  src="./assets/langchain_v1_1.png"
  alt="langchain ReAct模式"
  width="50%"
  align="center"
  :card="false"
/>

这里的model是静态给定的，我们也可以根据实际的业务在运行过程中动态选择模型，比如根据用户的输入动态选择模型，或者根据用户的输入动态选择模型参数，或者根据用户的输入动态选择模型输入参数等等。

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-3.5-turbo")
advanced_model = ChatOpenAI(model="gpt-4")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
```

### 7.2 自定义工具错误处理方式
可以使用 `wrap_tool_call` 装饰器来处理工具调用的错误，实际上就是把工具的错误处理逻辑封装成中间件，然后添加到 `middleware` 列表中。
```python
def wrap_tool_call(
    func: _CallableReturningToolResponse | None = None,
    *,
    tools: list[BaseTool] | None = None,
    name: str | None = None,
) -> (
    Callable[
        [_CallableReturningToolResponse],
        AgentMiddleware,
    ]
    | AgentMiddleware
)
```
这个中间件的wrap_tool_call函数，会被添加到 ToolNode 的实例化的过程中，因此对所有 Client-side 的Tool，都会添加这个中间件。

> 💡 Client-side 意味着是在用户的本地环境要执行，对于LLM，则是Server side，通过API访问的方式获取结果，并不是在用户本地环境执行推理。

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

agent = create_agent(
    model="gpt-4o",
    tools=[search, get_weather],
    middleware=[handle_tool_errors]
)
```

### 7.3 自定义 StateSchema
LangChain 1.0 做了大量的封装，虽然还保留了 state_schema 这个选项（主要是为了做兼容），但官方推荐使用中间件进行定义。

LangChain 中已经定义了一个给 Agent 的State，我们仅需要在其上面添加自己的 schema 即可。

```python
# langchain v1.0
class AgentState(TypedDict, Generic[ResponseT]):
    """State schema for the agent."""

    messages: Required[Annotated[list[AnyMessage], add_messages]]
    jump_to: NotRequired[Annotated[JumpTo | None, EphemeralValue, PrivateStateAttr]]
    structured_response: NotRequired[Annotated[ResponseT, OmitFromInput]]
```

### 7.3 修剪消息
大多数 LLM 都有最大支持的上下文窗口（以标记为单位）。可以在 before_model 的时候，对消息进行修剪。

```python
@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    if len(messages) <= 3:
        return None  # No changes needed

    first_msg = messages[0]
    recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
    new_messages = [first_msg] + recent_messages

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }
# create_agent 中把 trim_messages 添加到 middleware 中
```

可以在恰当的时机进行指定历史消息的删除，使用 langchain 的 RemoveMessage 方法

```python
from langchain.messages import RemoveMessage

@after_model
def delete_old_messages(state: AgentState, runtime: Runtime) -> dict | None:
    """Remove old messages to keep conversation manageable."""
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
    return None
```
### 7.4 SummarizationMiddleware
如果单纯地删除消息，可能会因为消息队列的清理而丢失信息。相比之前，更合适的方法是对之前的消息进行汇总、提炼。不过既然要汇总，那么就需要借用LLM模型了，因此需要额外配置对应的model。
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig


checkpointer = InMemorySaver()

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens", 4000),
            keep=("messages", 20)
        )
    ],
    checkpointer=checkpointer,
)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
agent.invoke({"messages": "hi, my name is bob"}, config)
agent.invoke({"messages": "write a short poem about cats"}, config)
agent.invoke({"messages": "now do the same but for dogs"}, config)
final_response = agent.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
```

### 7.5 自定义中间件
```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
```

Node风格：
- `@before_agent` - 在代理启动前运行（每次调用运行一次）
- `@before_model` - 在每次模型调用之前运行
- `@after_model` - 在每次模型响应后运行
- `@after_agent` - 在代理程序完成后运行（每次调用一次）
wrap风格：
- `@wrap_model_call` - 使用自定义逻辑包装每个模型调用
- `@wrap_tool_call` - 使用自定义逻辑包装每个工具调用
其他：
- `@dynamic_prompt` - 生成动态系统提示


## 8. Tool Runtime
工具可以通过ToolRuntime访问运行时，该参数提供：

- 状态- 在执行过程中流动的可变数据（例如，消息、计数器、自定义字段）
- 上下文- 不可变配置，例如用户 ID、会话详细信息或应用程序特定配置
- 存储- 跨对话的持久长期记忆
- 流写入器- 工具执行时流式自定义更新
- 配置-RunnableConfig用于执行
- 工具调用 ID - 当前工具调用的 ID


<Image 
  src='./assets/tool_runtime.svg'
  width='100%'
/>

如果要调用之前我们创建的 custom state，可以通过 runtime 进行访问
```python
# Access custom state fields
@tool
def get_user_preference(
    pref_name: str,
    runtime: ToolRuntime  # ToolRuntime parameter is not visible to the model
) -> str:
    """Get a user preference value."""
    preferences = runtime.state.get("user_preferences", {})
    return preferences.get(pref_name, "Not set")
```

## 总结
langchain 是通过大量的封装，构建一个看着比较简单的graph，基于Google的Pregel 模型进行实现，实现的重点是在于与其他函数的集成。

## 下一步
写一个基于 LangChain 实现的 RAG piplines。


