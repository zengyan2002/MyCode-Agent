# 团队成员忙碌期间消息处理修复 Plan

审批状态：用户已于 2026-09-13 批准并要求开始实现。实现中的必要衔接修正见下文。

## 架构概览

采用“当前一轮结束先读取邮箱，空闲时等待后端通知并定期检查唤醒消息”的方案。邮箱里尚未确认且 wake 为 true 的消息本身就是待处理通知，不另外建立通知文件或进程间服务。

TeamMailbox 继续负责消息持久化和消费位置。TeammateHost 负责串行执行轮次、等待和成功后的批量确认。InProcessBackend 负责消费并清除内存 Event。Supervisor、tmux 和 iTerm2 保留现有控制方式；忙碌时不向终端发送输入，避免输入影响成员正在运行的命令。

空闲 Host 每 200 毫秒检查一次未读唤醒消息，约每秒读取 5 次邮箱。该机制解决独立进程发送方没有本地唤醒回调，以及运行状态检查与进入等待之间的竞态。它会增加本地文件读取开销，不适合据此宣称支持大量成员或跨机器通信；当前目标是现有本机团队。已有后端通知仍能让等待提前结束，200 毫秒不是包含磁盘和调度耗时的硬延迟保证。

## 核心数据结构与接口

### 复用现有消息与游标

- MailboxMessage：沿用 message_id、wake、summary、body 等字段，无新增持久化字段。
- MailboxCursor：沿用 byte_offset 和 last_message_id，byte_offset 表示已确认批次最后一条完整消息后的字节位置。
- WakeWaiter：保留 Callable[[], Awaitable[None]]，一次返回表示消费一次后端通知；不表示一定存在模型输入。

### TeamMailbox

- read_unread(actor: TeamActorContext) -> tuple[MailboxMessage, ...]：接口和完整行读取规则保持不变。
- acknowledge(actor: TeamActorContext, messages: Sequence[MailboxMessage]) -> MailboxCursor：接收实际处理成功的一批消息，从当前消费位置扫描至 messages[-1].message_id 对应的完整行，用二进制读取位置保存新的字节偏移。不使用文件总长度推断本批终点。
- drain_for_agent(actor: TeamActorContext) -> tuple[UserMessage, ...]：保持现有接口及 Lead 的消费时机，只受益于共用的精确批次终点修复。

### TeammateHost

- __call__(launch: TeammateLaunch, wait_for_wake: WakeWaiter) -> None：公开接口保持不变。
- 模块常量 _MAILBOX_POLL_SECONDS = 0.2：固定空闲检查周期，不新增配置项或仅供测试的构造参数。
- wake_task: asyncio.Task[None] | None：在一次 Host 调用内保存尚未完成的后端等待。每个 Host 同时最多一个。邮箱通知触发工作时继续保留这个等待，后续空闲时复用；后端等待已结束则取得其结果并清空引用。

### InProcessBackend

- start(launch: TeammateLaunch) -> BackendHandle：在现有方法内定义 wait_for_wake() -> None，先 await event.wait()，再 event.clear()，把该函数传入 Host。
- wait 返回与 clear 之间没有 await，避免异步发送方在二者之间插入新的 set 后被清除。
- wake、stop、probe、wake_event 的现有接口保持不变。

## 模块设计

### 1. 消费确认（F3，N1）

Host 在 runtime.run 成功后一次调用 acknowledge(actor, messages)，替换当前逐条传入单个对象的错误调用。

acknowledge 在短暂的收件人邮箱锁内读取当前游标，以二进制逐行扫描至本批最后一个消息 ID，保存该行结束处的偏移。遇到目标前的记录沿用现有严格解码行为；找不到目标则报错，不用文件末尾代替。空批次维持原行为，不推进游标。

锁只保护读取确认终点及写游标，不覆盖模型调用。确认后追加的消息不受影响；确认前已经追加的新消息也不会被越过。既有按字节保存的游标无需迁移。当前每个收件人由自身消费者顺序确认，不引入多消费者事务。

### 2. Host 的轮次与等待（F1、F2、F3、F4、F5）

保持现有租约校验、运行环境加载和启动握手顺序。初始化成功后：

1. 读取当前未读消息，与尚未处理的首次提示组成输入。每个成功轮次结束后返回这一步，因此工作期间到达的消息会进入下一轮。
2. 有输入则标记 RUNNING 并 await runtime.run，不能同时开启另一轮。成功后清除首次提示并确认本批消息；失败沿用 FAILED 和关闭运行环境的流程，不确认本批。
3. 没有输入则写入 IDLE，创建或复用 wake_task，进入空闲等待循环。
4. 用 asyncio.wait 等待同一个 wake_task，timeout 为 0.2 秒。超时不会取消等待任务，不使用反复 wait_for 后取消标准输入读取的方式。
5. 后端通知完成时调用 task.result() 传播后端错误，清空引用，重新读取邮箱；没有输入就再次等待，不调用模型。
6. 超时时检查邮箱。只有发现未读 wake=true 消息才退出空闲等待，并将同批未读消息一起交给模型；只有 wake=false 消息时继续等待。已进入空闲等待的成员不会仅因普通积压消息而运行。
7. 邮箱触发一轮时若后端等待还没完成，继续持有并复用它。Host 退出时取消并回收这个异步等待任务，然后关闭 Runtime。标准输入的线程读取不因每次检查新建；Python 取消线程包装任务不能强行停止底层 readline，这沿用外部 Host 原有进程退出边界，不宣称修复所有终端关闭行为。

“无消息检查”和“开始等待”之间到达的 wake=true 消息会被下一次检查发现，不需要再来一次通知。邮箱触发工作的同时若后端信号也到达，至多多一次空邮箱检查，不重复运行同一批消息。

### 3. 后端通知与生产入口（F5、F6）

同进程后端消费 Event 后清除，避免设置一次后持续立即返回。不增加新的后端接口或测试专用实现参数。

Supervisor 保持只对 IDLE/SUSPENDED 发送后端通知；RUNNING 成员由未确认的唤醒消息保留通知，不靠修改其状态判断实现可靠性。这也避免改变任务扫描的实际唤醒成员集合。

独立 Host 的 SendMessage 仍可使用没有 wake_handler 的 TeamMailbox：它保存的 wake 字段会由接收者主动检查，因此不必把另一个进程的内存 Event 暴露给发送者，也不新增跨进程回调转发。测试必须覆盖这一真实装配形态，而非给所有发送方都装上测试回调。

tmux/iTerm2 保留现有标准输入通知。Host 对一个未结束的标准输入等待只创建一个任务；空闲检查作为第二个触发来源。已结束成员没有运行中的 Host，不会由这条路径重新创建。

## 模块交互

```text
A 的 SendMessage
  → TeamMailbox.send：消息含 wake 标记并落盘
  → 可选后端通知：空闲成员可以提前结束等待

B 正在工作：
  runtime.run 返回成功 → 确认实际处理批次 → 再读邮箱 → 执行下一轮

B 正在等待：
  后端通知到达，或 200ms 检查发现 wake=true 未读消息
  → 读取本批消息 → runtime.run → 确认本批终点 → 继续检查/等待
```

## 文件组织

| 操作 | 文件 | 内容 |
|---|---|---|
| 修改 | src/mycode/teams/mailbox.py | 按已处理批次的实际终点更新游标和注释 |
| 修改 | src/mycode/teams/host.py | 等待前检查、空闲定时检查、复用等待任务、批量确认 |
| 修改 | src/mycode/teams/backends/in_process.py | 消费 Event 后清除 |
| 修改 | src/mycode/teams/backends/base.py | 明确 WakeWaiter 的一次通知消费语义 |
| 修改 | tests/unit/teams/test_mailbox.py | 新消息到达期间确认、UTF-8 字节偏移、既有调用回归 |
| 修改 | tests/unit/teams/test_backends.py | Event 消费、重复唤醒及既有后端命令验证 |
| 新建 | tests/unit/teams/test_host.py | 轮次、等待边界、失败和结束清理 |
| 新建 | tests/integration/test_agent_team_messaging.py | 真实 Store/邮箱/Host/同进程后端组成的通信流程 |
| 修改 | docs/specs/team-message-wakeup/checklist.md | 实现后的逐项实际验证证据 |

cli.py 和外部后端的实现不修改，但其装配和控制行为纳入验证。

实现期间发现的必要衔接修正（2026-09-13）：Host 初始化后无输入会立即进入 IDLE，而 Supervisor._await_host_handshake 原先只接受 RUNNING，导致空会话恢复超时。将握手成功条件扩展为 RUNNING 或 IDLE；两种状态均由 Host 完成运行环境加载后写入。新增真实 Host 与 Supervisor 的空闲握手测试，不改变恢复策略和失败状态判断。

## 技术决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 忙碌通知如何保留 | 复用邮箱中未确认的 wake=true 消息 | 已有持久化字段能表达真实需求，无需维护第二份通知状态 |
| 跨进程到达检查 | 空闲时每 200ms 检查，保留后端通知 | 覆盖没有发送回调的独立成员；成本是定期本地文件读取 |
| 是否给忙碌终端发送输入 | 不发送 | 避免干扰工具标准输入，也不改任务扫描语义 |
| 消费位置 | 扫描到本批最后一个消息 ID | 保持数据模型和调用接口，避免文件末尾跳过新消息 |
| 后端等待生命周期 | 一个 Host 最多一个，跨轮复用 | 标准输入读取不因轮询超时反复创建线程 |
| 测试替换 | 测试侧 Runtime、屏障和 monkeypatch | 不增加只为测试使用的生产抽象 |
| 平台验证 | 本机真实同进程链路，外部命令契约测试 | 当前 Windows 未找到 tmux/it2，不能承诺真实终端验收 |

## spec 覆盖自检

F1/F2 由逐轮检查与空闲检查满足；F3 由批量确认和实际终点满足；F4 由空闲检查 wake 标记满足；F5 由 Event 清除、复用等待和空输入不调用模型满足；F6 由共用 Host、无回调发送测试和原后端控制语义满足。持久化格式不变，没有新模块间循环依赖。
