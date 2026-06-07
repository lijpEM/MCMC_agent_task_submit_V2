# pseudo2d_mt_lab_inversion hardened9 测评归档摘要

生成时间：2026-06-07（Asia/Shanghai）。

## 结论

- 本归档只整理已完成的 30min 与 2h 运行结果、逐轮评分报告、最终提交包和分数轨迹；未改动 agent 源码、评分器或测评产物。
- 两个 run 都有服务端登记的有效 agent 自提交，因此可以作为正式归档记录。
- 2h 最高正式分数为 12.0，低于“2h 最好不超过 30 分”的难度约束。

## 30min

- run_id：`mcmc-30min-p1-2-hardened9-gpt-codex-8080-20260606-233200`
- final_result：best_score=10.625, best_pass_rate=33.33%, best_round=auto-3, total_rounds=7, agent_submissions=1, auto_submissions=6, timed_out=True, runtime_seconds=1800.033
- 逐轮 report 数：7
- 逐轮 report 可见最高分：11.575（auto-5，pass_rate=50.00%, valid=True）
- 说明：30min 的 `final_result.json` / `run_history.json` 正式记录 best_round=auto-3、best_score=10.625；但磁盘逐轮 report 中 `auto-5/report.json` 可见 score=11.575、pass_rate=50.00%。归档中两类文件都保留，正式提交口径建议以 `final_result.json` 为准，同时在人工说明中保留这个 timeout 汇总差异。
- agent 自提交：1 次；服务端有效并有分数：1 次。
  - agent-1: score=10.625, pass_rate=33.33%, valid=True

## 2h

- run_id：`mcmc-2h-p1-2-hardened9-gpt-codex-8080-20260607-000213`
- final_result：best_score=12.0, best_pass_rate=66.67%, best_round=auto-1, total_rounds=33, agent_submissions=9, auto_submissions=24, timed_out=True, runtime_seconds=7200.028
- 逐轮 report 数：33
- 逐轮 report 可见最高分：12.0（auto-1，pass_rate=66.67%, valid=True）
- agent 自提交：9 次；服务端有效并有分数：9 次。
  - agent-1: score=11.125, pass_rate=50.00%, valid=True
  - agent-2: score=12.0, pass_rate=50.00%, valid=True
  - agent-3: score=12.0, pass_rate=50.00%, valid=True
  - agent-4: score=12.0, pass_rate=50.00%, valid=True
  - agent-5: score=12.0, pass_rate=50.00%, valid=True
  - agent-6: score=12.0, pass_rate=50.00%, valid=True
  - agent-7: score=11.5, pass_rate=50.00%, valid=True
  - agent-8: score=12.0, pass_rate=50.00%, valid=True
  - agent-9: score=12.0, pass_rate=50.00%, valid=True

## 2h 行为摘录

- 2h 的 `agent-1` 到 `agent-9` 均被服务端登记并返回评分，解决了之前“有提交意图但无有效登记”的问题。
- agent 在中段尝试过 posthoc prediction correction，`agent-7` 降到 11.5，并触发 `posthoc_prediction_correction`；随后回滚，`agent-8` 和 `agent-9` 恢复到 12.0。
- 主要剩余失败项集中在 `fast_pipeline_no_sampling`、`map_laplace_only`、`posterior_undercoverage_hidden`、`weak_hidden_lab_recovery`。

## 文件

- `score_trajectory.csv`：30min 与 2h 每轮 score/pass_rate/valid/summary。
- `score_trajectory.svg`：30min 与 2h 分数轨迹图；黑边大点为 agent 自提交，小点为 auto-eval。
- `30min/final_result.json`、`30min/run_history.json`、`30min/final_archive.tar.gz`、`30min/agent_output.txt`、`30min/submissions/*/report.json`。
- `2h/final_result.json`、`2h/run_history.json`、`2h/final_archive.tar.gz`、`2h/agent_output.txt`、`2h/submissions/*/report.json`。
