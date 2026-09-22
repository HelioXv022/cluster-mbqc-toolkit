import pytest

import cluster_mbqc_toolkit.devices as devices_mod
from cluster_mbqc_toolkit import make_device


def test_make_device_passthrough_for_default_qubit():
    dev = make_device("default.qubit", 3)
    assert dev.wires.tolist() == [0, 1, 2]


def test_make_device_wraps_missing_quimb_with_a_fix(monkeypatch):
    def fake_qml_device(name, wires, **kwargs):
        raise ImportError("This feature requires quimb, ... pip install quimb")

    monkeypatch.setattr(devices_mod.qml, "device", fake_qml_device)

    with pytest.raises(ImportError) as excinfo:
        make_device("default.tensor", 4, method="mps")

    assert "autoray==0.7.2" in str(excinfo.value)
    assert "cotengra==0.7.5" in str(excinfo.value)


def test_make_device_reraises_unrelated_import_errors(monkeypatch):
    def fake_qml_device(name, wires, **kwargs):
        raise ImportError("totally unrelated failure")

    monkeypatch.setattr(devices_mod.qml, "device", fake_qml_device)

    with pytest.raises(ImportError, match="totally unrelated failure"):
        make_device("default.tensor", 4)
