---
name: commit
description: 分析当前 Git 变更，逐文件暂存并生成规范的提交
allowedTools:
  - execute_command
  - read_file
  - search_code
mode: inline
---

# 提交当前变更

按照以下流程处理当前工作区的 Git 变更：

1. 先运行 `git status --short` 了解工作区全貌。
2. 分别查看 staged 和 unstaged diff；不要把两者混在一起判断。
3. 根据变更内容决定本次提交应包含哪些文件。逐个执行 `git add <文件>`，不得使用 `git add -A`。
4. 不得暂存 `.env`、密钥、临时脚本、构建产物或其他明显不应提交的文件。发现可疑文件时先向用户说明并停止提交。
5. 生成 Conventional Commits 格式的消息：`type(scope): description`。type 从 feat、fix、docs、refactor、test 中选择。
6. 如果变更内容跨度太大，先建议拆分提交，不要勉强塞进一个 commit。
7. 暂存完成后再次检查 staged diff，确认无误再执行 `git commit`。

用户补充要求：

$ARGUMENTS
