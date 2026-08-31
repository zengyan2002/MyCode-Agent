"""应用级固定限制与协议常量。"""

ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_MAX_TOKENS = 8192
ANTHROPIC_THINKING_BUDGET = 4096

HTTP_CONNECT_TIMEOUT_SECONDS = 10.0
HTTP_WRITE_TIMEOUT_SECONDS = 30.0
HTTP_POOL_TIMEOUT_SECONDS = 10.0
ERROR_BODY_LIMIT_BYTES = 2048

#git命令查看状态的超时时间
GIT_STATUS_TIMEOUT_SECONDS = 2.0

# 模型一次请求默认可容纳的输入与输出 Token 总数
DEFAULT_CONTEXT_WINDOW_TOKENS = 200_000
# 生成上下文摘要时，思考草稿和正式摘要合计最多使用的输出 Token 数
DEFAULT_COMPACTION_OUTPUT_TOKENS = 20_000
# 单个工具结果超过这个字符数时，把完整正文保存到工作区 artifact，对话中只保留文件位置和内容预览
DEFAULT_TOOL_RESULT_SPILL_CHARS = 50_000
# 同一轮工具结果中，未单独超限的正文合计超过这个字符数时，从最大的结果开始落盘，直到剩余正文不再超过该值
DEFAULT_TOOL_BATCH_SPILL_CHARS = 200_000
# 自动压缩时额外预留的 Token 数。请求估算达到“上下文窗口－摘要输出上限－该余量”时触发自动压缩
AUTO_COMPACTION_MARGIN_TOKENS = 13_000
# 用户手动压缩上下文时额外预留的 Token 数
MANUAL_COMPACTION_MARGIN_TOKENS = 3_000
# 压缩较早对话后，至少原样保留末尾多少个完整消息组
RECENT_MESSAGE_GROUPS = 5
# 如果末尾消息组不足这个 Token 数，继续向前保留完整消息组，直到达到该目标或已经没有更早的消息
RECENT_MESSAGE_TOKENS = 10_000
# 工具结果落盘后，开头和结尾各最多保留多少字符作为模型可见预览
TOOL_RESULT_PREVIEW_CHARS = 2_000
# 一次上下文压缩最多请求 Provider 的次数；连续失败达到该次数后，停止继续自动压缩，等待手动重试或重置
MAX_COMPACTION_ATTEMPTS = 3
# read_file 单次分段读取最多允许请求 48,000 字节；不指定分段大小时不受此限制
READ_FILE_CHUNK_LIMIT_BYTES = 48_000

# 项目指令文件最多递归展开 5 层 @include。
INSTRUCTION_INCLUDE_MAX_DEPTH = 5

# 会话列表从第一条用户消息生成标题时最多保留的字符数。
SESSION_TITLE_MAX_CHARS = 60
# 恢复的会话超过 24 小时未活跃时，提醒模型重新确认项目状态。
SESSION_GAP_REMINDER_HOURS = 24
# 启动时删除超过 30 天未活跃的会话文件。
SESSION_RETENTION_DAYS = 30
# 恢复超长会话时最多完成的上下文压缩轮数。
SESSION_RESTORE_MAX_COMPACTIONS = 3
# 每份长期记忆索引允许的最大行数和 UTF-8 字节数。
MEMORY_INDEX_MAX_LINES = 200
MEMORY_INDEX_MAX_BYTES = 25 * 1024
# 正常退出时等待后台记忆提取完成的最长时间。
MEMORY_SHUTDOWN_TIMEOUT_SECONDS = 10.0

# 项目本地 Hook 只服务当前工作区，不应提交到版本库。
LOCAL_HOOK_CONFIG_RELATIVE_PATH = ".mycode/config.local.yaml"
# Hook 命令、HTTP 响应和拒绝文案最多保留的字符数。
HOOK_OUTPUT_LIMIT_CHARS = 8_000
# Hook HTTP 请求等待连接和响应的最长时间。
HOOK_HTTP_TIMEOUT_SECONDS = 10.0
# 会话或应用关闭时等待异步 Hook 自行结束的最长时间。
HOOK_SHUTDOWN_TIMEOUT_SECONDS = 2.0

# 以下是产品级资源边界，不允许由模型或 Provider 配置控制。
TOOL_TIMEOUT_SECONDS = 30.0

# 最大模型调用次数限制一条用户请求内的真实 Provider 请求；读并发只限制同一个连续
# READ 批次。两者都可通过 AgentRunOptions 为可信调用方进一步收紧。
DEFAULT_MAX_MODEL_CALLS = 50
DEFAULT_MAX_READ_CONCURRENCY = 8
