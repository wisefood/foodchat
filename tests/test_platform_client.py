"""The WiseFood client the code was written against is the one requirements.txt installs.

Production went down on every session creation with
``Client.__init__() got an unexpected keyword argument 'telemetry'``: the code
passed a kwarg that exists from wisefood 0.0.26, and requirements.txt pinned
0.0.25 — which was also the newest version published, so the image could not
have had it. This test fails in any environment where the installed client does
not accept what ``_create_client`` passes, which is the drift that shipped.
"""

from __future__ import annotations

import inspect


def test_the_installed_client_accepts_what_create_client_passes():
    from wisefood import Client

    params = inspect.signature(Client.__init__).parameters
    assert "telemetry" in params, (
        "backend.platform passes telemetry=False; the installed wisefood client "
        f"does not take it (accepts: {sorted(params)})"
    )


def test_the_pin_is_at_least_the_first_version_with_telemetry():
    """The signature check above passes on a dev machine with an editable
    install; this one reads what the IMAGE will get."""
    import re
    from pathlib import Path

    req = Path(__file__).resolve().parents[1] / "requirements.txt"
    m = re.search(r"^wisefood==(\d+)\.(\d+)\.(\d+)", req.read_text(), re.M)
    assert m, "wisefood must be pinned exactly — an unpinned client is how versions drift"
    assert tuple(int(x) for x in m.groups()) >= (0, 0, 26)
