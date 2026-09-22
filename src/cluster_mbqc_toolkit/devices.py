"""
Device factory. Its one piece of logic: turning default.tensor's generic
"requires quimb" ImportError into a message that names the PennyLane /
quimb version conflict and its fix. See the README section "Known issue:
PennyLane <= 0.42 and quimb".
"""
from __future__ import annotations

import pennylane as qml


def make_device(name: str, num_wires: int, **kwargs):
    """``qml.device(name, wires=num_wires, **kwargs)``, with a specific
    error message for ``default.tensor`` when quimb is missing or broken.

    PennyLane <= 0.42 pins ``autoray<0.8``; current quimb pulls in a
    cotengra that requires ``autoray>=0.8``. PennyLane >= 0.43 pins
    ``autoray==0.8.x`` and has no conflict."""
    try:
        return qml.device(name, wires=num_wires, **kwargs)
    except ImportError as exc:
        if name == "default.tensor" and "quimb" in str(exc):
            raise ImportError(
                "default.tensor needs quimb. On PennyLane <= 0.42, installing "
                "current quimb breaks PennyLane's import (autoray/cotengra "
                "AttributeError): PennyLane <= 0.42 pins autoray<0.8, current "
                "cotengra requires autoray>=0.8. Fix with either:\n"
                '    pip install "pennylane>=0.43" quimb\n'
                "or, to stay on PennyLane <= 0.42, pin both:\n"
                '    pip install "autoray==0.7.2" "cotengra==0.7.5"'
            ) from exc
        raise
