"""Контракт кнопки «Исправить прикус» (шаг «Сканы», до монтажа; docs/handoff/bite-correction.md).

Ядро — bite.correct_bite (tests/test_bite.py: отказы без опоры и сверх предела). Реализовано 14.10 (ведущий,
пока Codex недоступен): tests/test_bite_correction_ui.py.
"""

import inspect

import pytest


def test_session_and_api_contract():
    from casedesigner.app import server
    from casedesigner.app.session import Session

    for name in ("bite_correct", "bite_accept", "bite_reject", "bite_reset"):
        assert callable(getattr(Session, name, None)), name
    src = server.routes_source()
    for action in ("correct", "accept", "reject", "reset"):  # POST /api/bite/<action>
        assert f'["bite", "{action}"]' in src, action
    assert "bite_fix" in inspect.getsource(Session.save_case) and "bite_fix" in inspect.getsource(Session.open_case)
    assert "bite_fix" in inspect.getsource(Session._update_bite_motion)  # прикус сканов = поправка ∘ прикус сканера
