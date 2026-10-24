"""Connect benchmark scripts to services owned by the user."""

from urllib.parse import urlsplit

from .json_io import write_json
from .providers.context import script_context


def service_endpoints(config, trial, values):
    required = config.get_scenario(trial.benchmark, trial.scenario).services
    urls = {}
    for value in values:
        name, separator, url = value.partition("=")
        if not separator or not name or not url:
            raise ValueError(
                "Use --service-url NAME=URL (e.g. leader=http://host:8000)"
            )
        if name not in required:
            raise ValueError(
                f"Service '{name}' is not required by this benchmark/scenario"
            )
        if name in urls:
            raise ValueError(f"Repeated service URL: {name}")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.query
            or parsed.fragment
            or (parsed.port is not None and parsed.port == 0)
        ):
            raise ValueError(f"Invalid service URL for {name}: use an HTTP(S) base URL")
        urls[name] = url.rstrip("/")
    if missing := set(required) - urls.keys():
        raise ValueError(f"Missing service URLs: {', '.join(sorted(missing))}")
    return [
        {
            "name": f"{name}-1",
            "group": name,
            "role": config.services[name].role,
            "replica": 1,
            "url": urls[name],
            "health_path": config.services[name].health_path,
            "metrics_path": config.services[name].metrics_path,
        }
        for name in required
    ]


def prepare_context(
    config, trial, root, directory, endpoints, *, service_project_dir=None
):
    instances = {item["name"]: item for item in endpoints}
    context = script_context(config, trial, root, instances)
    context.update(
        external_services=True,
        service_project_dir=str(service_project_dir or root),
        trial_dir=str(directory),
        output_dir=str(directory),
    )
    path = directory / "client.json"
    write_json(path, context)
    write_json(directory / "endpoints.json", endpoints)
    return path
