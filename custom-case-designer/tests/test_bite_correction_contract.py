"""Контракт кнопки «Исправить прикус» (шаг «Сканы», до монтажа; docs/handoff/bite-correction.md).

Ядро — bite.correct_bite (tests/test_bite.py: отказы без опоры и сверх предела). Тесты с xfail(strict=True) —
задачи ведомого: после реализации они проходят, и пометку нужно снять.
"""

import inspect

import pytest


@pytest.mark.xfail(strict=True, reason="ведомый: методы сессии, маршруты API, поправка в файле кейса")
def test_session_and_api_contract():
    from casedesigner.app import server
    from casedesigner.app.session import Session

    for name in ("bite_correct", "bite_accept", "bite_reject", "bite_reset"):
        assert callable(getattr(Session, name, None)), name
    src = server.routes_source()
    for route in ("/api/bite/correct", "/api/bite/accept", "/api/bite/reject", "/api/bite/reset"):
        assert route in src, route
    assert "bite_fix" in inspect.getsource(Session.save_case) and "bite_fix" in inspect.getsource(Session.open_case)
    assert "bite_fix" in inspect.getsource(Session._update_bite_motion)  # прикус сканов = поправка ∘ прикус сканера
