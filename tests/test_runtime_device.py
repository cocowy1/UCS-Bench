from directme.perception.runtime import resolve_runtime_device


def test_explicit_cpu_is_stable():
    assert resolve_runtime_device("cpu") == "cpu"


def test_auto_returns_supported_runtime_string():
    assert resolve_runtime_device("auto") in {"cpu", "cuda", "mps"}
