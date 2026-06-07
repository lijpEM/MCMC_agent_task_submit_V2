FROM python:3.11-slim

COPY task_files /opt/mcmc_task_files
COPY judge_files /opt/mcmc_judge_files

RUN mkdir -p /home/workspace/pseudo2d_mt_lab /opt/evaluator \
    && cp /opt/mcmc_judge_files/evaluate.py /opt/evaluator/evaluate.py \
    && chmod -R 555 /opt/evaluator

RUN pip install --no-cache-dir -i https://mirrors.ivolces.com/pypi/simple/ \
    "numpy>=1.24" \
    "scipy>=1.10"
