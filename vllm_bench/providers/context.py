"""The script context is the same on every provider; only addresses differ."""

from .paths import CONTAINER_TRIAL


def service_instances(config, trial, *, shared_network=False):
    scenario = config.get_scenario(trial.benchmark, trial.scenario)
    instances = {}
    for group in scenario.services:
        service = config.services[group]
        for replica in range(1, service.replicas + 1):
            name = f"{group}-{replica}"
            port = 8000 + len(instances) if shared_network else service.health_port
            host = "127.0.0.1" if shared_network else name
            instances[name] = {
                "name": name,
                "group": group,
                "role": service.role,
                "replica": replica,
                "url": f"http://{host}:{port}",
                "health_path": service.health_path,
                "metrics_path": service.metrics_path,
            }
    return instances


def script_context(config, trial, project, instances):
    scenario = config.get_scenario(trial.benchmark, trial.scenario)
    return {
        **trial.model_dump(),
        "sample_interval": config.sample_interval,
        "global_args": config.get_global_args(trial.benchmark),
        "service_args": {
            group: config.services[group].service_args for group in scenario.services
        },
        "scenario_args": scenario.args,
        "benchmark_args": config.benchmarks[trial.benchmark].script_args,
        "project_dir": str(project),
        "service_project_dir": str(project),
        "trial_dir": str(CONTAINER_TRIAL),
        "services": list(instances.values()),
        "internal_services": list(instances.values()),
    }
