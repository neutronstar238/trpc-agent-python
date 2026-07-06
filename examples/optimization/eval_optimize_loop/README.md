# Evaluation + Optimization Loop Example

This example demonstrates an auditable evaluation and prompt-optimization loop.
It is intentionally offline-first: `fake` and `trace` modes need no API key.

## Modes

```bash
python examples/optimization/eval_optimize_loop/run_pipeline.py --mode fake
python examples/optimization/eval_optimize_loop/run_pipeline.py --mode trace
python examples/optimization/eval_optimize_loop/run_pipeline.py --mode online
```

`fake` mode evaluates deterministic fixture outputs for baseline and three
candidates through `AgentEvaluator`. It should select `candidate_local_patch`,
reject a no-op candidate, and reject an overfit candidate that improves training
while regressing a critical validation case.

`trace` mode materializes recorded conversations for baseline and every
candidate, then runs the same `AgentEvaluator` summary path with
`eval_mode="trace"`. This proves the replay path works without model inference.

`online` mode uses `AgentOptimizer.optimize(...)`, `TargetPrompt`, and
`optimizer.json`. It first prints only whether required environment variables
are present. It requires:

```bash
TRPC_AGENT_API_KEY
TRPC_AGENT_BASE_URL
TRPC_AGENT_MODEL_NAME
```

The online smoke path is opt-in because it performs real optimizer and
revalidation calls. The default optimizer config is bounded for this example,
but real-provider latency is not held to the fake/trace three-minute
deterministic expectation. A native optimizer `SUCCEEDED` status is recorded as
an artifact only; the product decision is always the report's
`gate_decision.accepted`.

## Outputs

Each run writes to `runs/<run_id>/` by default:

- `optimization_report.json`: issue-facing machine-readable summary.
- `optimization_report.md`: concise human-readable report.
- `trace_evalset.json` and `trace_metrics.json`: trace-mode replay inputs.
- `online/result.json`, `online/summary.txt`, `online/rounds/`,
  `online/baseline_prompts/`, `online/best_prompts/`, and
  `online/config.snapshot.json`: native optimizer artifacts in online mode.

The top-level report always includes `run_id`, `mode`, `seed`, `baseline`,
`candidates`, `delta`, `gate_decision`, `failure_attribution`, `cost`,
`duration_seconds`, `config_snapshot`, `environment_snapshot`, and `artifacts`.
`optimization_report.schema.json` is the contract used by the CLI before it
writes `optimization_report.json`; schema validation failures stop report
generation instead of producing a partial artifact.

A compact sample output is checked in at `fixtures/optimization_report.sample.json`.

## CLI Inputs

```bash
python examples/optimization/eval_optimize_loop/run_pipeline.py \
  --mode fake \
  --train-evalset train.evalset.json \
  --optimizer-dev-evalset optimizer_dev.evalset.json \
  --val-evalset val.evalset.json \
  --optimizer-config optimizer.json \
  --gate-config gate.json \
  --output-dir runs \
  --run-id demo
```

All path arguments have defaults pointing at this example. `--system-prompt`
and `--router-prompt` are used by online mode and are still recorded in offline
configuration snapshots. Gate config may override validation delta, hard-fail,
critical-regression, cost, duration, and required-metric checks. By default the
gate inherits `optimize.stop.required_metrics` from `optimizer.json`.

The deterministic metric in this example is `route_tool_args_score`: it parses
the final JSON response and scores only `route`, `tool.name`, and
`tool.arguments`. Reason wording and safety are handled by the rubric metric, so
a harmless explanation rewrite does not zero out an otherwise correct route.

`environment_snapshot` records the git commit, dirty flag, Python version, SDK
version when installed, model name, redacted base URL host, seed, command, and
optimizer config path. It never records API keys. Known provider/runtime noise
from DeepSeek schema downgrades and SSE decoder shutdown is isolated from the
online smoke output.

`optimizer_dev.evalset.json` is the optimizer-internal holdout passed to
`AgentOptimizer.optimize(..., validation_dataset_path=...)`. `val.evalset.json`
is the final validation set and is only used for baseline scoring and final
candidate gate scoring.

## Design Notes

本示例把“评测 - 失败归因 - prompt 候选 - 验证 gate - 审计报告”做成一个可复现的最小闭环，而不是把所有逻辑塞进 `AgentOptimizer` 核心模块。原因是 SDK 已经提供 `AgentEvaluator`、`AgentOptimizer`、`TargetPrompt` 和 GEPA 产物持久化，issue 的关键缺口是把这些能力组织成一个业务可复制的 pipeline：离线时能稳定证明 gate 行为，在线时能切到真实模型和原生优化器，失败时能解释为什么拒绝候选。

默认 `fake` 模式使用固定 fixture 输出，避免 API key、模型随机性和成本影响 CI；fixture 只作为 agent output，所有分数、pass/fail 和 metric 明细都来自 `AgentEvaluator`。它同时构造三类候选：`candidate_local_patch` 修复退款和人工升级路由且不破坏 FAQ；`candidate_noop` 没有验证集提升，因此被拒绝；`candidate_overfit` 训练集满分但把关键物流政策 case 错误升级为人工，因此被 hard-fail 和 critical-regression gate 拒绝。`trace` 模式会为 baseline 和每个 candidate 生成 `eval_mode="trace"` 数据，再调用 `AgentEvaluator` 回放，验证“不推理也能评测”的离线路径。`online` 模式只在显式选择时读取模型环境变量，把 `optimizer_dev.evalset.json` 传给原生优化器，最终再用 `val.evalset.json` 重评 baseline 和 best prompt；顶层报告基于原生 `OptimizeResult` 以及 best prompt 的重新验证结果生成。

报告采用 JSON 优先：机器读取 `optimization_report.json` 做 gate、审计和 CI 判断；人读 `optimization_report.md` 只看 baseline、winner、case delta、失败归因和接受/拒绝理由。成本审计会区分 optimizer 调用和最终重评调用；如果供应商价格未知，报告不会把未知成本写成 0 并通过成本预算。所有运行时产物只写入 `runs/` 或调用方指定的输出目录，避免污染源 prompt 和示例数据。

## Verification

```bash
pytest tests/evaluation/test_eval_optimize_loop_example.py -q
python examples/optimization/eval_optimize_loop/run_pipeline.py --mode fake
python examples/optimization/eval_optimize_loop/run_pipeline.py --mode trace
```

Full repository pytest includes optional integration suites. In an environment
without optional extras such as Cube/E2B, Mempalace, A2A, AG-UI, Claude Agent
SDK, or OpenClaw dependencies, `tests/conftest.py` ignores those suites during
collection and prints the missing dependency list in the pytest header.
