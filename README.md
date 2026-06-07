# 拟二维海洋 MT 贝叶斯 LAB 反演

本目录是 SE-Bench 任务 `pseudo2d_mt_lab_inversion` 的 hardened9 最终可提交题包。包内包含任务说明、agent 起始环境、公开数据、starter 代码、评分器、judge 评分器副本、校准说明和 30min/2h GPT-Codex 实跑证据。

## 目录结构

```text
docs/           任务说明、评分说明、校准记录、hardened9 分数轨迹和二阶段说明材料。
environment/    Dockerfile、Python 依赖、agent 可见任务文件和 judge 评分器副本。
scorer/         评分脚本和本地 score wrapper。
tasks/          SE-Bench 任务 JSON。
```

## 主要任务文件

```text
docs/task_spec.md
environment/task_files/data/mt_profile_20_public.npz
environment/task_files/data/serpent_mt/SERPENT_fullMTdataSet.txt
environment/task_files/data/serpent_mt/maxrho_1300C_MPT.txt
environment/task_files/starter/README.md
environment/task_files/starter/load_profile_data.py
environment/task_files/starter/mt1d_forward.py
environment/task_files/starter/pseudo2d_model.py
environment/task_files/starter/example_forward_one_station.py
environment/task_files/starter/run_baseline_map.py
scorer/evaluate.py
scorer/score.sh
environment/judge_files/evaluate.py
tasks/pseudo2d_mt_lab_inversion.json
```

## 评分器版本

当前提交包内四份评分器同步，SHA256 为：

```text
b5ca7146c7681d11ac59da1a44ee74759d1c126da6883e994fa9b2bfb819d4b2
```

对应文件：

```text
scorer/evaluate.py
environment/judge_files/evaluate.py
../judge_files/evaluate.py
../judge_files/evaluate_scorer_copy.py
```

评分方向为 `maximize`。评分器返回 0-100 连续分数，并按 A-F 组件、结构性失败、hidden 泛化质量和后验可信度综合评分。

## Hardened9 实跑证据

本提交包使用 hardened9 评分器和 GPT/Codex agent 在 8080 judge route 完成 30min 与 2h 复跑。两个 run 都有服务端登记的有效 agent 自提交，因此可作为正式归档记录。

```text
30min: mcmc-30min-p1-2-hardened9-gpt-codex-8080-20260606-233200
       best_score=10.625
       best_pass_rate=0.3333333333333333
       best_round=auto-3
       total_rounds=7
       agent_submissions=1
       auto_submissions=6
       timed_out=true
       runtime_seconds=1800.032591342926
       archive_size_bytes=44736
       agent-1 score=10.625, pass_rate=0.3333333333333333, valid=true

2h:    mcmc-2h-p1-2-hardened9-gpt-codex-8080-20260607-000213
       best_score=12.0
       best_pass_rate=0.6666666666666666
       best_round=auto-1
       total_rounds=33
       agent_submissions=9
       auto_submissions=24
       timed_out=true
       runtime_seconds=7200.027633428574
       archive_size_bytes=264193
       agent-1 through agent-9: all service-side valid and scored
```

2h 最高正式分数为 12.0，低于“2h 最好不超过 30 分”的难度校准目标。30min 的 `final_result.json` / `run_history.json` 记录 `best_round=auto-3`、`best_score=10.625`；逐轮 report 中 `auto-5/report.json` 可见 `score=11.575`、`pass_rate=0.5`，这是 timeout 收尾时的汇总差异。正式 benchmark 口径以 `final_result.json` 为准，同时在 `docs/hardened9_results/SUMMARY.md` 中保留该说明。

## 二阶段说明材料

```text
docs/task_authenticity_statement_zh.txt
docs/workload_time_statement_zh.txt
docs/human_reference_result_zh.txt
docs/contamination_prevention_statement_zh.txt
docs/environment_dry_run_statement_zh.txt
docs/baseline_results_zh.txt
docs/baseline/starter_baseline_run_20260605.log
docs/baseline/starter_baseline_eval_20260605.txt
```

## Hardened9 归档材料

```text
docs/hardened9_results/SUMMARY.md
docs/hardened9_results/score_trajectory.csv
docs/hardened9_results/score_trajectory.svg
docs/hardened9_results/30min_final_result.json
docs/hardened9_results/30min_run_history.json
docs/hardened9_results/30min_agent1_report.json
docs/hardened9_results/30min_auto5_report_timeout_late.json
docs/hardened9_results/2h_final_result.json
docs/hardened9_results/2h_run_history.json
docs/hardened9_results/2h_agent9_report.json
```

完整外部归档包保存在工作区：

```text
/home/workspace/mcmc_agent_qc0605/archive_mcmc_hardened9_20260607.tar.gz
```

## 本次修复覆盖的质检问题

本提交包修复并加固了 P1-2 质检指出的评分链路问题：

- 结构乘数封顶：无结构性失败时 structural penalty 可达到 `1.0`；hidden 质量失败不再直接造成结构乘数崩塌。
- `map_laplace_only` 假阳性：真实 adaptive MCMC / DE-MCMC / ensemble / SMC / VI 等采样证据不会仅因出现 optimizer、covariance、Hessian、Laplace 初始化等词被误判。
- 隐藏泛化执行链路：hidden rerun 会复制 helper 模块/目录，并支持 CLI 与环境变量输入输出路径。
- 隐藏真值泄露：hidden 输入会剥离真值字段，避免 agent 直接读取 hidden answer。
- hidden/public 分支不一致：公开 MCMC、hidden 切换 canonical/Laplace/VI 分支的行为会被检测并限分。
- 样本诊断漏洞：评分器会读取 posterior samples 自行估算样本数量、活跃参数、split-Rhat、ESS 和链顺序可信度，防止只靠 summary 虚报诊断或打乱样本顺序获得高分。

## 提交状态

当前 hardened9 包可作为提交候选。最终打包前应清理 `__pycache__/` 和 `*.pyc`，并记录 zip SHA256。
