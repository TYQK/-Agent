# -Agent
扫描仓库 → 规则审查 → 可选 LLM 审查 → 自动安全修复 → 运行测试/校验 → 生成 Markdown/JSON 报告 → 可导出 git diff patch。
运行方式

把文件放到你的项目根目录，然后执行：

python repo_review_agent.py --repo . --out agent_review_report.md --json-out agent_review_report.json --validate

如果想让它自动做安全修复，比如去除行尾空格、补文件末尾换行、把 Python 的 except: 改成 except Exception:：

python repo_review_agent.py --repo . --fix --validate --out agent_review_report.md --patch-out agent_fix.patch

如果你想启用 LLM 审查：

export OPENAI_API_KEY="你的 API Key"
python repo_review_agent.py --repo . --llm --validate --out agent_review_report.md

也可以指定模型和接口：

python repo_review_agent.py \
  --repo . \
  --llm \
  --model 你的模型名 \
  --base-url https://api.openai.com/v1/chat/completions \
  --fix \
  --validate
