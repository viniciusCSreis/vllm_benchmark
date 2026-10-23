# Minimal example

Install the CLI on your host and start Docker. The service uses Python's standard
library and the provider mounts `vllm_bench`; the image has no CLI installation.
From this directory:

```bash
vllm_bench run_scenarios --benchmark example --scenario baseline
```

Results and reports appear in `results/local-example/run001/`.
For a real vLLM workload, select your vLLM image, replace `startup.py` with your
server launcher, and replace `benchmark.py` with your workload client.
