"""
A thin device factory. The one piece of real logic here is turning
default.tensor's generic "requires quimb" ImportError into a message that
also names the actual version-conflict trap this repo's author hit while
building it -- see the README's "lessons learned" section for the full
debugging story.
"""
from __future__ import annotations

import pennylane as qml


def make_device(name: str, num_wires: int, **kwargs):
    """``qml.device(name, wires=num_wires, **kwargs)`` with a friendlier
    error for the one failure mode that turned out to be worth explaining:
    ``default.tensor`` needing quimb, whose own dependency (cotengra)
    conflicts with PennyLane's pinned ``autoray`` version."""
    try:
        return qml.device(name, wires=num_wires, **kwargs)
    except ImportError as exc:
        if name == "default.tensor" and "quimb" in str(exc):
            raise ImportError(
                "default.tensor needs quimb. If `pip install quimb` alone "
                "still leaves PennyLane failing to import afterwards (an "
                "autoray/cotengra AttributeError or ImportError), it is "
                "very likely this specific conflict: PennyLane pins "
                "autoray<0.8, but quimb's cotengra dependency needs "
                "autoray>=0.8's get_namespace(). Fix with, e.g.:\n"
                '    pip install "autoray==0.7.2" "cotengra==0.7.5"\n'
                "(pin BOTH -- pinning only autoray leaves cotengra broken, "
                "and vice versa)."
            ) from exc
        raise
