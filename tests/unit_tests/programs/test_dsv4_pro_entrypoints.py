"""Tests for the lightweight DSV4 Pro entry points and visualization."""

from types import SimpleNamespace

import pytest

import rooflang.programs.models.dsv4_pro as dsv4_pro
from rooflang.language.graph import ComputeGraph
from rooflang.language.hardware.component import Compute
from rooflang.language.kernels.forward import Nop
from rooflang.language.kernels.identity import Spawn
from rooflang.programs.experiments import main as main_module
from rooflang.programs.experiments import simulation, visualization
from rooflang.programs.models import MODEL_NAMES, load_model


def test_package_lazy_visualization_attribute():
    assert dsv4_pro.visualize_layer is visualization.visualize_layer
    with pytest.raises(AttributeError, match="has no attribute 'missing'"):
        dsv4_pro.__getattr__("missing")


def test_model_loader_exposes_registered_models():
    assert MODEL_NAMES == ("dsv4_pro",)
    assert load_model("dsv4_pro") is dsv4_pro
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("missing")


def test_simulate_runs_and_exports(monkeypatch):
    result = object()
    run = lambda: result
    simulator = SimpleNamespace(run=run)
    constructor_calls = []
    monkeypatch.setattr(
        simulation, "Simulator",
        lambda *args, **kwargs: constructor_calls.append((args, kwargs))
        or simulator,
    )
    exports = []
    monkeypatch.setattr(
        simulation, "export_trace",
        lambda value, path: exports.append((value, path)),
    )

    assert simulation.simulate(
        "graph", "placement", "hardware", "trace.json",
        measurement_start="read",
    ) is result
    assert constructor_calls == [
        (("graph", "placement", "hardware"), {"measurement_start": "read"})
    ]
    assert exports == [(result, "trace.json")]


class _Meta:
    pass


def test_collect_kernels_handles_attributes_sequences_and_cycles():
    first, second = Nop(), Nop()
    meta = _Meta()
    nested = _Meta()
    meta.value = [first, (nested, object())]
    nested.kernel = second
    nested.parent = meta

    assert visualization._collect_kernels(meta) == {first, second}
    assert visualization._collect_kernels(first) == {first}


def test_visualize_layer_adds_adjacent_structural_kernels(monkeypatch):
    graph = ComputeGraph()
    seed, spawn, adjacent, isolated = Nop(), Spawn(world=1), Nop(), Spawn(world=1)
    for kernel in (seed, spawn, adjacent, isolated):
        graph.add_kernel(kernel)
    graph.add_control_edge(seed, spawn)
    graph.add_control_edge(spawn, adjacent)

    layer = _Meta()
    layer.seed = seed
    calls = []
    monkeypatch.setattr(
        visualization, "export_graph",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    visualization.visualize_layer(
        graph, layer, extra_seeds={adjacent}, path="layer.svg")

    args, kwargs = calls[0]
    assert args == (graph, "layer.svg")
    assert kwargs["kernels"] == {seed, spawn, adjacent}
    assert kwargs["figsize"] == (24, 16)
    assert isolated not in kwargs["kernels"]


def test_visualize_layer_uses_large_figure(monkeypatch):
    graph = ComputeGraph()
    kernels = [Nop() for _ in range(51)]
    for kernel in kernels:
        graph.add_kernel(kernel)
    monkeypatch.setattr(
        visualization, "_collect_kernels", lambda _: set(kernels))
    calls = []
    monkeypatch.setattr(
        visualization, "export_graph",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    visualization.visualize_layer(graph, object())

    assert calls[0]["figsize"] == (48, 32)


class _Superchip:
    pass


def test_main_covers_cluster_prefill_decode_and_superchip(monkeypatch,
                                                          capsys):
    cluster = SimpleNamespace(nodes=[Compute("gpu0", kind="gpu")])
    superchip = _Superchip()
    monkeypatch.setattr(main_module, "B300SuperChip", _Superchip)
    monkeypatch.setattr(main_module, "H200SuperChip", _Superchip)
    monkeypatch.setattr(main_module, "HARDWARE_MAP", {
        "Cluster": lambda: cluster,
        "Superchip": lambda: superchip,
    })

    declarations = []
    declared = SimpleNamespace(
        graph="graph", layers=["layer"], emb="emb", read="read",
        kv=["kv"], output="output",
    )

    def declare_model(**kwargs):
        declarations.append(kwargs)
        return (
            declared.graph, declared.layers, declared.emb, declared.read,
            declared.kv, declared.output,
        )

    calls = []
    selected_models = []
    fake_model = SimpleNamespace(
        N_LAYERS=61,
        declare_model=declare_model,
        optimize_model_cluster_prefill=lambda *args, **kwargs:
        calls.append(("prefill", args, kwargs))
        or ("prefill_graph", "prefill_placement"),
        optimize_model_cluster_decode=lambda *args, **kwargs:
        calls.append(("decode", args, kwargs))
        or ("decode_graph", "decode_placement"),
        optimize_model_superchip=lambda *args:
        calls.append(("superchip", args, {}))
        or ("super_graph", "super_placement"),
    )
    monkeypatch.setattr(
        main_module, "load_model",
        lambda name: selected_models.append(name) or fake_model,
    )
    visualizations = []
    monkeypatch.setattr(
        visualization, "visualize_layer",
        lambda *args, **kwargs: visualizations.append((args, kwargs)),
    )
    simulations = []
    simulation_results = iter([
        SimpleNamespace(measured_time_us=1000.0, measurement_start_us=0.0),
        SimpleNamespace(measured_time_us=2000.0, measurement_start_us=500.0),
        SimpleNamespace(measured_time_us=3000.0, measurement_start_us=0.0),
    ])
    monkeypatch.setattr(
        main_module, "simulate",
        lambda *args, **kwargs: simulations.append((args, kwargs))
        or next(simulation_results),
    )

    monkeypatch.setattr("sys.argv", [
        "dsv4", "--hardware", "Cluster", "--stage", "prefill",
        "--batch-size", "16", "--cp", "1", "--dp", "1",
        "--ep", "1", "--pp-partition", "61",
        "--measurement_start", "none", "--visualization",
    ])
    main_module.main()

    monkeypatch.setattr("sys.argv", [
        "dsv4", "--hardware", "Cluster", "--stage", "decode",
    ])
    main_module.main()

    monkeypatch.setattr("sys.argv", [
        "dsv4", "--hardware", "Superchip", "--stage", "prefill",
    ])
    main_module.main()

    assert declarations[0]["batch_size"] == 16
    assert "batch_size" not in declarations[1]
    assert declarations[1]["decode"] is True
    assert [call[0] for call in calls] == ["prefill", "decode", "superchip"]
    assert selected_models == ["dsv4_pro"] * 3
    assert visualizations == [
        (("graph", "layer"), {"extra_seeds": {"emb", "read", "kv"}})
    ]
    assert simulations[0][1]["measurement_start"] is None
    assert simulations[1][1]["measurement_start"] == "read"
    output = capsys.readouterr().out
    assert "prefill (Cluster)" in output
    assert "decode (Cluster)" in output
    assert "KV preload: 500.0 us" in output


def test_main_visualization_without_a_layer_is_a_noop(monkeypatch):
    cluster = SimpleNamespace(nodes=[Compute("gpu0", kind="gpu")])
    monkeypatch.setattr(main_module, "HARDWARE_MAP", {
        "Cluster": lambda: cluster,
    })
    monkeypatch.setattr(main_module, "load_model", lambda name: SimpleNamespace(
        N_LAYERS=61,
        declare_model=lambda **kwargs:
        ("graph", [], None, None, [], "output"),
        optimize_model_cluster_prefill=lambda *args, **kwargs:
        ("graph", "placement"),
    ))
    monkeypatch.setattr(
        main_module, "simulate",
        lambda *args, **kwargs: SimpleNamespace(
            measured_time_us=1.0, measurement_start_us=0.0),
    )
    monkeypatch.setattr(
        visualization, "visualize_layer",
        lambda *args, **kwargs: pytest.fail("must not visualize"),
    )
    monkeypatch.setattr("sys.argv", [
        "dsv4", "--hardware", "Cluster", "--stage", "prefill",
        "--visualization",
    ])

    main_module.main()
