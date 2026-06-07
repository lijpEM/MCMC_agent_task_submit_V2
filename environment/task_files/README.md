# Pseudo-2D Marine MT Bayesian LAB Inversion Task

This directory contains the files needed to run the agent test for the
`pseudo2d_mt_lab_inversion` task.

Directory layout:

```text
data/           Public 20-station MT profile and original SERPENT MT files.
docs/           Task specification and reference papers.
environment/    Dockerfile and Python requirements.
scorer/         Evaluation script and local score wrapper.
starter/        Lightweight starter code plus full SERPENT reference port.
tasks/          SE-Bench task JSON.
design/         Design notes and earlier task drafts.
inbox/          Raw incoming materials kept for provenance.
```

Primary task files:

```text
docs/task_spec.md
data/mt_profile_20_public.npz
starter/README.md
starter/load_profile_data.py
starter/mt1d_forward.py
starter/pseudo2d_model.py
starter/example_forward_one_station.py
starter/run_baseline_map.py
scorer/evaluate.py
tasks/pseudo2d_mt_lab_inversion.json
```

Local smoke-test commands:

```bash
cd /home/workspace/pseudo2d_mt_lab
python starter/example_forward_one_station.py
python starter/run_baseline_map.py
```
