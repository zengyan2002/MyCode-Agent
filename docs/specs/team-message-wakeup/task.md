# 团队成员忙碌期间消息处理修复 Tasks

审批状态：用户已于 2026-09-13 统一批准剩余文档并要求开始实现。实际进度见文末。

## 文件清单

| 操作 | 文件 | 职责 |
|---|---|---|
| 修改 | src/mycode/teams/mailbox.py | 精确批次确认 |
| 修改 | src/mycode/teams/host.py | 轮次和等待衔接 |
| 修改 | src/mycode/teams/backends/in_process.py | 消费后清除 Event |
| 修改 | src/mycode/teams/backends/base.py | 等待接口说明 |
| 修改 | tests/unit/teams/test_mailbox.py | 确认与消息格式回归 |
| 修改 | tests/unit/teams/test_backends.py | 通知消费与终端命令契约 |
| 新建 | tests/unit/teams/test_host.py | Host 行为测试 |
| 新建 | tests/integration/test_agent_team_messaging.py | 通信集成测试 |
| 修改 | docs/specs/team-message-wakeup/checklist.md | 验收状态与证据 |

## 执行约定

- 以下实现任务须在剩余三份文档得到批准后开始；每个任务约 2—5 分钟的聚焦改动，验证命令运行时间另计。
- 在工作区根目录执行命令，使用 python -m pytest。当前已发现 Python 3.12 和 pytest；实际依赖是否齐备由第一次验证确认。
- 测试先复现对应缺陷属于预期红灯；明确标注后随对应实现修复，成组变绿再提交。
- .gitignore 当前忽略 docs/。提交时只对本次四份文档使用 git add -f 的明确路径，不修改忽略规则，不添加其他文档或产物。
- 只提交本次改动，保留用户后来产生的无关修改。每组提交前看差异并执行 git diff --check。

## T1：建立验收基线

**文件：** tests/unit/teams/、tests/integration/test_agent_team*.py（读取和运行）。
**依赖：** 文档批准。
**步骤：** 检查工作区状态；运行团队单元测试与既有团队集成测试；记录失败是否已存在；记录 tmux/it2 可用性。
**验证：** python -m pytest tests/unit/teams tests/integration/test_agent_team.py tests/integration/test_agent_team_worktrees.py tests/integration/test_agent_team_recovery.py tests/integration/test_agent_team_cleanup.py；保存实际结果，不能把既有失败记为本次已修复。

## T2：添加消费位置回归场景

**文件：** tests/unit/teams/test_mailbox.py。
**依赖：** T1。
**步骤：** 读取第一批后追加第二批再确认第一批；加入中文正文以检查字节偏移；验证第二批仍未读、空批次不推进，以及 drain_for_agent 既有调用仍可用。
**验证：** python -m pytest tests/unit/teams/test_mailbox.py；新消息被跳过的测试在修复前应失败，记录失败断言。

## T3：修复批次确认

**文件：** src/mycode/teams/mailbox.py。
**依赖：** T2。
**步骤：** 保留 acknowledge 的序列参数；在收件人锁内从当前游标二进制扫描到本批最后消息的完整行；用实际结束偏移保存游标；目标不存在时报错；更新注释。
**验证：** python -m pytest tests/unit/teams/test_mailbox.py tests/unit/teams/test_tools.py 全部通过；查看 diff 确认 JSON 格式未改变。完成第一组提交。

## T4：添加同进程通知消费测试

**文件：** tests/unit/teams/test_backends.py。
**依赖：** T1。
**步骤：** 通过真实 InProcessBackend.start 获得传入 Host 的等待函数；验证一次唤醒结束一次等待，第二次等待阻塞；再次唤醒后第二次等待完成；多次 set 可合并。
**验证：** python -m pytest tests/unit/teams/test_backends.py；记录修复前第二次等待意外返回的失败。

## T5：消费并清除 Event

**文件：** src/mycode/teams/backends/in_process.py、src/mycode/teams/backends/base.py。
**依赖：** T4。
**步骤：** start 内增加生产使用的局部 wait_for_wake 函数，wait 后无中间 await 地 clear；传给 Host；明确 WakeWaiter 消费一次通知的语义。
**验证：** python -m pytest tests/unit/teams/test_backends.py 全部通过。完成第二组提交。

## T6：添加 Host 轮次和确认测试

**文件：** tests/unit/teams/test_host.py。
**依赖：** T3、T5。
**步骤：** 用真实 Store/邮箱和测试侧 Runtime 建立成员租约；用 Event 屏障停住第一轮；轮内发送消息后放行；验证两轮串行、第二轮自动收到消息；覆盖失败不确认和批量确认不抛类型错误。
**验证：** python -m pytest tests/unit/teams/test_host.py；记录现有 Host 停在等待或确认报错的证据。

## T7：调整 Host 的每轮处理顺序

**文件：** src/mycode/teams/host.py。
**依赖：** T6。
**步骤：** 每轮先读邮箱；有输入才执行；成功后一次确认整批；下一轮先读积压消息；保留初始提示清除、异常状态和退出处理顺序。
**验证：** python -m pytest tests/unit/teams/test_host.py 中 T6 场景通过；不新增模型调用并发。

## T8：补充等待边界与默认不唤醒测试

**文件：** tests/unit/teams/test_host.py。
**依赖：** T7。
**步骤：** 用受控的后端等待和测试侧读取屏障覆盖检查后到达、已等待时到达、wake=false 不触发、重复通知后重新阻塞；使用没有 wake_handler 的发送邮箱；检查跨消息轮次最多一个未结束后端等待。
**验证：** python -m pytest tests/unit/teams/test_host.py；等待回归场景先失败；超时只作测试卡死上限，不以 sleep 推测并发顺序。

## T9：实现空闲消息检查和等待任务复用

**文件：** src/mycode/teams/host.py。
**依赖：** T8。
**步骤：** 引入 _MAILBOX_POLL_SECONDS；保存一个 wake_task；用 asyncio.wait 的超时检查 wake=true 消息；后端完成时取得结果并清空引用；邮箱触发时复用未完成等待；退出回收等待任务；空输入继续等待。
**验证：** python -m pytest tests/unit/teams/test_host.py tests/unit/teams/test_backends.py 全部通过，包括错误传播和取消清理。完成第三组提交。

## T10：验证通信集成和外部后端契约

**文件：** tests/integration/test_agent_team_messaging.py、tests/unit/teams/test_backends.py。
**依赖：** T9。
**步骤：** 组合真实 TeamMailbox、TeamStateStore、TeammateHost、InProcessBackend，经过 SendMessageTool 验证 A 发给忙碌及空闲 B；发送侧另建无回调邮箱覆盖独立成员装配形态；执行一个真实子进程投递用例，验证文件通知能跨进程；用测试侧 monkeypatch 核对 tmux/it2 命令目标及复用标准输入等待；验证已结束成员不被重建。
**验证：** python -m pytest tests/integration/test_agent_team_messaging.py tests/unit/teams/test_backends.py；模型执行替换为受控测试 Runtime，报告为通信链路集成，不宣称真实模型或终端验收。完成第四组提交。

## T11：回归验收与交付

**文件：** checklist.md 及本次实现和测试文件。
**依赖：** T10。
**步骤：** 逐项执行 checklist；先通过团队回归，再运行一次全套测试；记录失败及平台限制；检查测试替换没有扩散到生产接口；按实际证据勾选；提交验收记录。
**验证：** 下列明确命令，另加 git diff --check；若失败则修复重跑受影响范围，基线失败单独列出。

```powershell
python -m pytest tests/unit/teams tests/integration/test_agent_team.py tests/integration/test_agent_team_worktrees.py tests/integration/test_agent_team_recovery.py tests/integration/test_agent_team_cleanup.py tests/integration/test_agent_team_messaging.py
python -m pytest
git diff --check
```

T1 的基线命令使用上述第一条并去掉尚未创建的 test_agent_team_messaging.py。

## 执行顺序

T1 → T2 → T3 → T4 → T5 → T6 → T7 → T8 → T9 → T10 → T11。

## 自检

Mailbox、Host、后端等待和生产装配均有对应任务。接口名称与 plan.md 一致，依赖无环，每项有具体验证。测试失败复现和修复后通过明确区分；实际执行状态只依据测试证据填写。

## 执行进度与必要补充

- T1：完成，团队基线 42 passed。
- T2—T3：完成，先复现 2 项失败，修复后邮箱/工具 9 passed；提交 4f8b6fa。
- T4—T5：完成，先复现通知被重复消费，修复后后端 3 passed；提交 ee74255。
- T6—T9：完成，先复现轮次等待、确认类型错误和空闲通知超时，修复后 Host/后端 10 passed；提交 02533e4。
- T10：完成，通信及后端 9 passed，含真实子进程发送；提交 3c57466。
- T11：完成，最终团队回归 59 passed；全套 1114 passed、4 skipped；差异检查通过。验收证据与真实终端限制已写入 checklist.md。
- 补充任务：已完成，修改 src/mycode/teams/supervisor.py 的 _await_host_handshake，接受初始化完毕的 IDLE；在 tests/integration/test_agent_team_messaging.py 增加真实 Host 的空闲握手回归。先复现握手超时，修复后团队回归 59 passed；提交 f652e5c。原因见 plan.md。
- 提交完整性补充：tests/ 原本全部被忽略，新增测试依赖既有 support.py 及测试包 __init__.py。将 tests/unit/teams/support.py、tests/__init__.py、tests/unit/__init__.py、tests/unit/teams/__init__.py、tests/integration/__init__.py 一并纳入本次明确路径提交，避免新检出仓库缺少测试依赖；不修改这些文件的已有内容，也不提交其他被忽略测试。
