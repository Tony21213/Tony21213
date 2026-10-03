"""Память совмещений: программа учится на том, где пользователь оставил скан.

Если автоматическое совмещение не устроило, пользователь поправляет
положение скана вручную. После принятия кейса запоминается, что предложила
программа и где оказался скан в итоге. Итоговое положение считается верным.

Выучивается то, что повторяется от кейса к кейсу на одном аппарате КТ:

* **сдвиг границы эмали.** Размытие и артефакты у каждого аппарата свои:
  граница зуба на КТ лежит чуть снаружи или внутри настоящей поверхности, и
  скан садится выше или ниже, чем надо. Для каждого принятого кейса
  известно, какой сдвиг объясняет итоговое положение (Registration.edge_shift):
  оценённый вместе с положением, если пользователь ничего не менял, или
  восстановленный из ручного положения, если менял. Медиана по аппарату
  становится априорной поправкой, и чем больше принятых кейсов, тем сильнее
  она удерживает следующие совмещения (CaseCT(edge_prior=..., prior_weight=...));
* **как часто и насколько пользователь поправляет** автоматику на этом
  аппарате — показатель, где её надо улучшать.

Учимся только на принятых кейсах и только на хорошо легших сканах — иначе
программа закрепляла бы ошибки. В память не пишутся имена файлов и пациентов:
только аппарат, матрицы и показатели точности.
"""

import json
import os
import time

import numpy as np

from .register import apply

# Поправка начинает действовать, когда по аппарату накопилось столько принятых кейсов.
MIN_CASES = 3
# Кейс годится для обучения, если скан хорошо лёг на коронки.
GOOD_FIT = {"min_matched_fraction": 0.2, "max_p90_mm": 0.3}
# Сдвиги больше этого — не свойство аппарата, а ошибка; такие кейсы не учитываются.
MAX_BIAS_MM = 0.3
# Вес одного принятого кейса в априорной поправке, в «точках скана», и предел накопления.
CASE_WEIGHT = 300.0
MAX_WEIGHT_CASES = 30


def correction_mm(scan_vertices: np.ndarray, auto: np.ndarray, final: np.ndarray) -> float:
    """Насколько пользователь сдвинул скан относительно предложенного программой (максимум по вершинам)."""
    return float(np.linalg.norm(apply(final, scan_vertices) - apply(auto, scan_vertices), axis=1).max())


class AlignmentMemory:
    """Принятые совмещения в файле JSON Lines (по строке на кейс)."""

    def __init__(self, path: str):
        self.path = path

    def records(self) -> list[dict]:
        if not os.path.isfile(self.path):
            return []
        with open(self.path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def record(self, device: str, auto, final) -> dict:
        """Запомнить принятое совмещение.

        auto — что предложила программа (CaseCT.register), final — на чём
        остановился пользователь: тот же объект, если он ничего не менял, или
        CaseCT.evaluate / CaseCT.register(start=...) для поправленного положения.
        """
        stats = final.stats
        rec = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": device or "unknown",
            "jaw": final.jaw,
            "corrected_mm": round(correction_mm(final.scan.vertices, auto.transform, final.transform), 4),
            "edge_shift_mm": round(float(final.edge_shift), 4),
            "fit": stats,
            "auto_transform": np.round(auto.transform, 9).tolist(),
            "final_transform": np.round(final.transform, 9).tolist(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def _usable(self, device: str) -> list[dict]:
        out = []
        for r in self.records():
            fit = r.get("fit", {})
            if (r.get("device") == (device or "unknown")
                    and fit.get("matched_fraction", 0) >= GOOD_FIT["min_matched_fraction"]
                    and fit.get("p90_mm", np.inf) <= GOOD_FIT["max_p90_mm"]
                    and abs(r.get("edge_shift_mm", np.inf)) <= MAX_BIAS_MM):
                out.append(r)
        return out

    def prior(self, device: str) -> tuple[float, float]:
        """Выученный сдвиг границы эмали для аппарата (мм) и его вес; (0, 0) — пока мало кейсов."""
        usable = self._usable(device)
        if len(usable) < MIN_CASES:
            return 0.0, 0.0
        shift = float(np.median([r["edge_shift_mm"] for r in usable]))
        return shift, CASE_WEIGHT * min(len(usable), MAX_WEIGHT_CASES)

    def summary(self) -> dict:
        """По каждому аппарату: сколько кейсов, выученная поправка, как часто правили вручную."""
        out = {}
        for device in sorted({r.get("device", "unknown") for r in self.records()}):
            recs = [r for r in self.records() if r.get("device") == device]
            corrected = [r["corrected_mm"] for r in recs]
            out[device] = {
                "cases": len(recs),
                "usable_for_learning": len(self._usable(device)),
                "edge_shift_mm": round(self.prior(device)[0], 4),
                "corrected_share": round(float(np.mean([c > 0.05 for c in corrected])), 3) if recs else 0.0,
                "median_correction_mm": round(float(np.median(corrected)), 4) if recs else 0.0,
            }
        return out
