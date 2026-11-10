# -*- coding: utf-8 -*-
"""Ядро экспертной системы на имитационном принципе.

Реализует метод из лекции «Построение модели в виде оценочной функции»
(курс «Моделирование систем»): линейная модель Yj = a1j·x1 + … + anj·xn,
обучение по принципу «кнута и пряника» (поощрение столбца верного класса,
наказание остальных), экспертиза по принципу демона («кто громче крикнет,
тот и прав»), нормализация в гиперкуб [-1; 1], шаг обучения h
(коэффициент темперамента), дообучение при ошибках «мира».

Модуль не зависит от интерфейса: его использует и Tkinter-приложение,
и автотесты.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np

TIE_EPS = 1e-9          # равные голоса демонов → объект не опознан
UNKNOWN_VALUE = 0.5     # значение неопределённого признака (следствие 2 лекции)

# ---------------------------------------------------------------------------
# База знаний по умолчанию: музыкальные инструменты (л.р. 2)
# ---------------------------------------------------------------------------

DEFAULT_FEATURES = [
    "Струнный", "Смычковый", "Щипковый", "Духовой",
    "Клавишный", "Ударный", "Электрический", "Деревянный",
]

DEFAULT_CLASSES = [
    "Скрипка", "Гитара", "Электрогитара", "Флейта",
    "Барабан", "Ксилофон", "Пианино", "Синтезатор",
]

DEFAULT_EXAMPLES = [
    ("Скрипка",       [1, 1, 0, 0, 0, 0, 0, 1]),
    ("Гитара",        [1, 0, 1, 0, 0, 0, 0, 1]),
    ("Электрогитара", [1, 0, 1, 0, 0, 0, 1, 0]),
    ("Флейта",        [0, 0, 0, 1, 0, 0, 0, 0]),
    ("Барабан",       [0, 0, 0, 0, 0, 1, 0, 0]),
    ("Ксилофон",      [0, 0, 0, 0, 0, 1, 0, 1]),
    ("Пианино",       [1, 0, 0, 0, 1, 1, 0, 1]),
    ("Синтезатор",    [0, 0, 0, 0, 1, 0, 1, 0]),
]

# Объекты, которых система не видела при обучении — задача прогноза.
UNKNOWN_OBJECTS = {
    "Виолончель":    [1, 1, 0, 0, 0, 0, 0, 1],
    "Укулеле":       [1, 0, 1, 0, 0, 0, 0, 1],
    "Бас-гитара":    [1, 0, 1, 0, 0, 0, 1, 0],
    "Электроскрипка": [1, 1, 0, 0, 0, 0, 1, 0],
    "Клавесин":      [1, 0, 1, 0, 1, 0, 0, 1],
    "Орга́н":         [0, 0, 0, 1, 1, 0, 0, 1],
    "Саксофон":      [0, 0, 0, 1, 0, 0, 0, 0],
    "Литавры":       [0, 0, 0, 0, 0, 1, 0, 0],
    "Электроорган":  [0, 0, 0, 0, 1, 0, 1, 0],
    "Арфа":          [1, 0, 1, 0, 0, 0, 0, 1],
}

UPDATE_RULES = {
    "on_error": "только при ошибке (лекция: «в успешном случае модель остаётся неизменной»)",
    "always": "при каждом предъявлении (как в Colab-наработке)",
}


def fmt(v, digits: int = 2) -> str:
    """Компактная запись числа: 4 → «4», 0.5 → «0.5», -1.25 → «-1.25»."""
    v = float(v)
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    s = ("%." + str(digits) + "f") % v
    return s.rstrip("0").rstrip(".")


def vec_str(vec) -> str:
    """Вектор признаков в записи лекции: (1,1,0,0,0,0,0,1)."""
    return "(" + ",".join(fmt(v) for v in vec) + ")"


# ---------------------------------------------------------------------------
# Записи истории обучения
# ---------------------------------------------------------------------------

@dataclass
class Presentation:
    """Один акт обучения: «мир» предъявил пример, система среагировала,
    «мир» поощрил/наказал."""
    epoch: int
    index: int                      # номер примера в эпохе, с 1
    class_index: int
    class_name: str
    vector: list
    demons_before: list             # реакция системы ДО коррекции
    predicted: int | None           # индекс класса-ответа или None
    predicted_name: str
    ok: bool
    updated: bool                   # была ли коррекция матрицы
    clipped: bool                   # сработала ли нормализация
    matrix_before: np.ndarray
    matrix_after: np.ndarray


@dataclass
class EpochRecord:
    number: int                     # сквозной номер эпохи (итерации)
    presentations: list = field(default_factory=list)
    errors_during: int = 0          # ошибок реакции в ходе эпохи
    check: list = field(default_factory=list)   # экспертиза после эпохи
    errors_after: int = 0           # ошибок экспертизы после эпохи
    matrix_after: np.ndarray | None = None

    @property
    def total(self) -> int:
        return len(self.presentations)


@dataclass
class TrainReport:
    kind: str                       # scratch | continue | feedback
    epochs: list = field(default_factory=list)
    converged: bool = False
    reason: str = ""
    params: dict = field(default_factory=dict)
    added_example: tuple | None = None

    @property
    def n_epochs(self) -> int:
        return len(self.epochs)

    @property
    def curve_during(self) -> list:
        return [e.errors_during for e in self.epochs]

    @property
    def curve_after(self) -> list:
        return [e.errors_after for e in self.epochs]

    @property
    def final_errors(self) -> int:
        return self.epochs[-1].errors_after if self.epochs else 0


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class ExpertModel:
    """Линейная экспертная система: матрица aij (строки — признаки,
    столбцы — классы), обучаемая поощрением/наказанием."""

    def __init__(self, features=None, classes=None, examples=None, *,
                 step: float = 1.0, normalize: bool = False,
                 update_rule: str = "on_error", max_epochs: int = 30,
                 auto_stop: bool = True):
        self.features = list(features or DEFAULT_FEATURES)
        self.classes = list(classes or DEFAULT_CLASSES)
        self.examples = [(c, list(map(float, v)))
                         for c, v in (examples or DEFAULT_EXAMPLES)]
        self.step = float(step)
        self.normalize = bool(normalize)
        self.update_rule = update_rule
        self.max_epochs = int(max_epochs)
        self.auto_stop = bool(auto_stop)
        self.matrix = np.zeros((len(self.features), len(self.classes)))
        self.history: list[EpochRecord] = []
        self.reports: list[TrainReport] = []
        self._validate()

    # ------------------------------------------------------------------
    # служебное
    # ------------------------------------------------------------------

    @property
    def n_features(self) -> int:
        return len(self.features)

    @property
    def n_classes(self) -> int:
        return len(self.classes)

    @property
    def trained(self) -> bool:
        return bool(np.any(self.matrix != 0))

    @property
    def epochs_done(self) -> int:
        return len(self.history)

    def _validate(self):
        if len(set(self.features)) != len(self.features):
            raise ValueError("Признаки повторяются")
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("Классы повторяются")
        for cls, vec in self.examples:
            if cls not in self.classes:
                raise ValueError("Пример ссылается на неизвестный класс «%s»" % cls)
            if len(vec) != self.n_features:
                raise ValueError("Пример «%s»: длина вектора %d, признаков %d"
                                 % (cls, len(vec), self.n_features))

    def params(self) -> dict:
        return {"step": self.step, "normalize": self.normalize,
                "update_rule": self.update_rule, "max_epochs": self.max_epochs,
                "auto_stop": self.auto_stop}

    def reset(self):
        """Чистая матрица: система ещё ничего не знает."""
        self.matrix = np.zeros((self.n_features, self.n_classes))
        self.history = []
        self.reports = []

    # ------------------------------------------------------------------
    # экспертиза
    # ------------------------------------------------------------------

    def demons(self, x) -> np.ndarray:
        """Вектор демонов: X · A."""
        return np.asarray(x, dtype=float) @ self.matrix

    def decide(self, demons) -> tuple[int | None, float]:
        """Принцип демона: побеждает максимальный голос.
        Возвращает (индекс победителя или None при ничьей, отрыв от второго)."""
        d = np.asarray(demons, dtype=float)
        if d.size == 0:
            return None, 0.0
        order = np.argsort(-d, kind="stable")
        best = int(order[0])
        gap = float(d[best] - d[order[1]]) if d.size > 1 else float("inf")
        if d.size > 1 and gap <= TIE_EPS:
            return None, 0.0
        return best, gap

    def contributions(self, x) -> np.ndarray:
        """Слагаемые aij·xi (строки — признаки, столбцы — классы)."""
        return np.asarray(x, dtype=float)[:, None] * self.matrix

    def expertise(self, x, name: str = "объект") -> dict:
        """Режим экспертизы для одного объекта. Возвращает словарь с
        демонами, победителем, отрывом, ранжированием и слагаемыми."""
        x = [float(v) for v in x]
        if len(x) != self.n_features:
            raise ValueError("Ожидается %d признаков, получено %d"
                             % (self.n_features, len(x)))
        d = self.demons(x)
        winner, gap = self.decide(d)
        order = [int(i) for i in np.argsort(-d, kind="stable")]
        ranking = [(self.classes[i], float(d[i])) for i in order]
        runner_up = None
        if winner is not None and len(order) > 1:
            second = order[1] if order[0] == winner else order[0]
            runner_up = (self.classes[second], float(d[second]))
        known = self.match_example(x)
        return {
            "name": name,
            "input": x,
            "demons": [float(v) for v in d],
            "winner": winner,
            "winner_name": self.classes[winner] if winner is not None else "не опознан",
            "recognized": winner is not None,
            "runner_up": runner_up,
            "gap": gap,
            "ranking": ranking,
            "contributions": self.contributions(x),
            "known_as": known,          # имя класса, если вектор совпал с эталоном
            "plane_check": [(self.classes[j], float(d[j]),
                             "на плоскости" if abs(d[j]) <= TIE_EPS else
                             ("положительное полупространство" if d[j] > 0
                              else "отрицательное полупространство"))
                            for j in range(self.n_classes)],
        }

    def match_example(self, x) -> str | None:
        x = np.asarray(x, dtype=float)
        for cls, vec in self.examples:
            if np.allclose(x, vec):
                return cls
        return None

    def check_all(self) -> list[dict]:
        """Экспертиза всех примеров базы на текущей матрице."""
        rows = []
        for cls, vec in self.examples:
            d = self.demons(vec)
            winner, gap = self.decide(d)
            rows.append({
                "class": cls,
                "vector": list(vec),
                "predicted": self.classes[winner] if winner is not None else "не опознан",
                "ok": winner is not None and self.classes[winner] == cls,
                "demons": [float(v) for v in d],
                "gap": gap,
            })
        return rows

    def errors_now(self) -> int:
        return sum(1 for r in self.check_all() if not r["ok"])

    # ------------------------------------------------------------------
    # обучение
    # ------------------------------------------------------------------

    def present(self, class_index: int, vector, epoch: int = 0,
                index: int = 0) -> Presentation:
        """Один акт обучения: реакция → оценка «мира» → поощрение/наказание."""
        x = np.asarray(vector, dtype=float)
        before = self.matrix.copy()
        d = x @ self.matrix
        predicted, _ = self.decide(d)
        ok = predicted == class_index
        updated = self.update_rule == "always" or not ok
        clipped = False
        if updated:
            delta = self.step * x
            sign = -np.ones(self.n_classes)
            sign[class_index] = 1.0              # пряник верному, кнут остальным
            self.matrix = self.matrix + delta[:, None] * sign[None, :]
            if self.normalize:
                clipped_m = np.clip(self.matrix, -1.0, 1.0)
                clipped = bool(np.any(clipped_m != self.matrix))
                self.matrix = clipped_m
        return Presentation(
            epoch=epoch, index=index, class_index=class_index,
            class_name=self.classes[class_index], vector=[float(v) for v in x],
            demons_before=[float(v) for v in d], predicted=predicted,
            predicted_name=(self.classes[predicted] if predicted is not None
                            else "не опознан"),
            ok=ok, updated=updated, clipped=clipped,
            matrix_before=before, matrix_after=self.matrix.copy(),
        )

    def run_epoch(self) -> EpochRecord:
        """Одна эпоха: предъявить все примеры базы, затем экспертиза всех."""
        number = self.epochs_done + 1
        rec = EpochRecord(number=number)
        for i, (cls, vec) in enumerate(self.examples, start=1):
            p = self.present(self.classes.index(cls), vec, epoch=number, index=i)
            rec.presentations.append(p)
            if not p.ok:
                rec.errors_during += 1
        rec.check = self.check_all()
        rec.errors_after = sum(1 for r in rec.check if not r["ok"])
        rec.matrix_after = self.matrix.copy()
        self.history.append(rec)
        return rec

    def _train_loop(self, kind: str, max_epochs: int, auto_stop: bool,
                    min_epochs: int = 1) -> TrainReport:
        if not self.examples:
            raise ValueError("База примеров пуста — учить нечему")
        report = TrainReport(kind=kind, params=self.params())
        for _ in range(max_epochs):
            rec = self.run_epoch()
            report.epochs.append(rec)
            if auto_stop and rec.errors_after == 0 and len(report.epochs) >= min_epochs:
                report.converged = True
                report.reason = ("все %d примеров опознаются верно после %d эпох"
                                 % (len(self.examples), report.n_epochs))
                break
        else:
            report.converged = report.final_errors == 0
            report.reason = ("достигнут предел %d эпох, ошибок %d/%d"
                             % (max_epochs, report.final_errors, len(self.examples)))
        self.reports.append(report)
        return report

    def train(self, max_epochs: int | None = None,
              auto_stop: bool | None = None) -> TrainReport:
        """Обучение с нуля до сходимости (или до предела эпох)."""
        self._validate()
        self.reset()
        return self._train_loop(
            "scratch",
            self.max_epochs if max_epochs is None else int(max_epochs),
            self.auto_stop if auto_stop is None else bool(auto_stop))

    def train_continue(self, epochs: int = 1) -> TrainReport:
        """Дообучение: ещё `epochs` проходов по базе поверх текущей матрицы."""
        self._validate()
        return self._train_loop("continue", int(epochs), auto_stop=False)

    def learn_feedback(self, vector, true_class: str) -> TrainReport:
        """«Мир» наказал за неверный ответ: запоминаем пример (если он новый)
        и доучиваем систему, пока все примеры не опознаются (предел эпох)."""
        vector = [float(v) for v in vector]
        if true_class not in self.classes:
            self.add_class(true_class)
        added = None
        if not any(c == true_class and np.allclose(v, vector)
                   for c, v in self.examples):
            self.examples.append((true_class, vector))
            added = (true_class, vector)
        report = self._train_loop("feedback", self.max_epochs, auto_stop=True)
        report.added_example = added
        return report

    # ------------------------------------------------------------------
    # редактирование базы знаний
    # ------------------------------------------------------------------

    def add_feature(self, name: str, default: float = 0.0):
        name = name.strip()
        if not name:
            raise ValueError("Имя признака пустое")
        if name in self.features:
            raise ValueError("Признак «%s» уже есть" % name)
        self.features.append(name)
        self.examples = [(c, v + [float(default)]) for c, v in self.examples]
        self.matrix = np.vstack([self.matrix, np.zeros((1, self.n_classes))])

    def remove_feature(self, index: int) -> str:
        if self.n_features <= 1:
            raise ValueError("Нельзя удалить последний признак")
        name = self.features.pop(index)
        self.examples = [(c, v[:index] + v[index + 1:]) for c, v in self.examples]
        self.matrix = np.delete(self.matrix, index, axis=0)
        return name

    def add_class(self, name: str):
        name = name.strip()
        if not name:
            raise ValueError("Имя класса пустое")
        if name in self.classes:
            raise ValueError("Класс «%s» уже есть" % name)
        self.classes.append(name)
        self.matrix = np.hstack([self.matrix, np.zeros((self.n_features, 1))])

    def remove_class(self, index: int) -> str:
        if self.n_classes <= 1:
            raise ValueError("Нельзя удалить последний класс")
        name = self.classes.pop(index)
        self.examples = [(c, v) for c, v in self.examples if c != name]
        self.matrix = np.delete(self.matrix, index, axis=1)
        return name

    def add_example(self, cls: str, vector):
        if cls not in self.classes:
            raise ValueError("Неизвестный класс «%s»" % cls)
        vector = [float(v) for v in vector]
        if len(vector) != self.n_features:
            raise ValueError("Ожидается %d признаков" % self.n_features)
        self.examples.append((cls, vector))

    def set_example(self, index: int, cls: str, vector):
        if cls not in self.classes:
            raise ValueError("Неизвестный класс «%s»" % cls)
        self.examples[index] = (cls, [float(v) for v in vector])

    def remove_example(self, index: int) -> tuple:
        return self.examples.pop(index)

    def problems(self) -> list[str]:
        """Замечания к базе: неразличимые классы, неинформативные признаки,
        классы без примеров."""
        notes = []
        have = {c for c, _ in self.examples}
        for c in self.classes:
            if c not in have:
                notes.append("Класс «%s» без примеров — система его не выучит." % c)
        seen: dict[tuple, str] = {}
        for c, v in self.examples:
            key = tuple(v)
            if key in seen and seen[key] != c:
                notes.append("Классы «%s» и «%s» имеют одинаковые признаки %s — "
                             "они неразличимы, нужен дополнительный признак."
                             % (seen[key], c, vec_str(v)))
            seen.setdefault(key, c)
        if self.examples:
            arr = np.array([v for _, v in self.examples])
            for i, f in enumerate(self.features):
                col = arr[:, i]
                if np.allclose(col, col[0]):
                    notes.append("Признак «%s» одинаков у всех примеров (=%s) — "
                                 "он неинформативен." % (f, fmt(col[0])))
        return notes

    # ------------------------------------------------------------------
    # анализ модели
    # ------------------------------------------------------------------

    def equation(self, j: int, zero_terms: bool = False) -> str:
        """Уравнение класса j: y1 = -8·x1 + 4·x2 … (Скрипка)."""
        terms = []
        for i in range(self.n_features):
            a = self.matrix[i, j]
            if abs(a) <= TIE_EPS and not zero_terms:
                continue
            sign = "-" if a < 0 else "+"
            mag = fmt(abs(a))
            coef = "" if mag == "1" else mag + "·"
            terms.append((sign, "%sx%d" % (coef, i + 1)))
        if not terms:
            body = "0"
        else:
            first_sign, first = terms[0]
            body = ("-" if first_sign == "-" else "") + first
            for s, t in terms[1:]:
                body += " %s %s" % (s, t)
        return "y%d = %s   (%s)" % (j + 1, body, self.classes[j])

    def equations(self) -> list[str]:
        return [self.equation(j) for j in range(self.n_classes)]

    def prototypes(self) -> dict:
        """Эталон класса: средний вектор его примеров."""
        proto = {}
        for c in self.classes:
            vs = [v for cc, v in self.examples if cc == c]
            if vs:
                proto[c] = np.mean(np.array(vs, dtype=float), axis=0)
        return proto

    def world_distances(self) -> dict:
        """Хэмминговы расстояния между эталонами («мир как есть»)."""
        proto = self.prototypes()
        names = [c for c in self.classes if c in proto]
        out = {}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                out[(names[a], names[b])] = float(
                    np.abs(proto[names[a]] - proto[names[b]]).sum())
        return out

    def mind_distances(self) -> dict:
        """Различия столбцов матрицы («мир в сознании системы»)."""
        out = {}
        for a in range(self.n_classes):
            for b in range(a + 1, self.n_classes):
                out[(self.classes[a], self.classes[b])] = float(
                    np.abs(self.matrix[:, a] - self.matrix[:, b]).sum())
        return out

    def feature_importance(self) -> list[tuple[str, float]]:
        """Размах коэффициента признака по классам: 0 → признак ни на что
        не влияет (следствие 1 лекции)."""
        out = []
        for i, f in enumerate(self.features):
            row = self.matrix[i]
            out.append((f, float(row.max() - row.min()) if row.size else 0.0))
        return out

    # ------------------------------------------------------------------
    # объяснение «почему?»
    # ------------------------------------------------------------------

    def explain(self, x, name: str = "объект") -> dict:
        """Экспертиза + подробное объяснение по шагам.
        Возвращает результат expertise() c полями sections (список
        (заголовок, текст)) и text (всё одной строкой)."""
        res = self.expertise(x, name)
        x = res["input"]
        d = res["demons"]
        sections: list[tuple[str, str]] = []

        # 1. входной вектор
        present = [(self.features[i], x[i]) for i in range(self.n_features) if x[i] != 0]
        lines = ["X = %s" % vec_str(x)]
        if present:
            lines.append("Есть признаки: " + ", ".join(
                "%s%s" % (f, "" if v == 1 else " (=%s, неизвестно)" % fmt(v))
                for f, v in present) + ".")
        else:
            lines.append("Ни один признак не задан — все xi = 0.")
        if res["known_as"]:
            lines.append("Такой вектор совпадает с эталоном «%s» из базы — "
                         "система этот пример видела при обучении." % res["known_as"])
        else:
            lines.append("Такого примера в базе не было — это задача прогноза: "
                         "система ищет наиболее похожий из известных ей классов.")
        sections.append(("Шаг 1. Входной вектор признаков", "\n".join(lines)))

        # 2. произведение X·A
        lines = ["Голос демона j-го класса: Yj = Σ aij·xi "
                 "(столбец j матрицы, умноженный на X)."]
        for j, cls in enumerate(self.classes):
            terms = ["%s·(%s)" % (fmt(x[i]), fmt(self.matrix[i, j]))
                     for i in range(self.n_features) if x[i] != 0]
            body = " + ".join(terms) if terms else "0"
            lines.append("Y%d (%s) = %s = %s" % (j + 1, cls, body, fmt(d[j])))
        sections.append(("Шаг 2. Умножаем X на матрицу коэффициентов", "\n".join(lines)))

        # 3. выбор
        lines = ["Вектор демонов: %s." % vec_str(d)]
        if res["recognized"]:
            w = res["winner"]
            lines.append("Максимальный голос Y%d = %s → объект опознан как «%s» "
                         "(«кто громче крикнет, тот и прав»)."
                         % (w + 1, fmt(d[w]), self.classes[w]))
            ru = res["runner_up"]
            lines.append("Второе место: «%s» с голосом %s. Отрыв Δ = %s — это "
                         "уверенность системы (расстояние до разделяющей плоскости)."
                         % (ru[0], fmt(ru[1]), fmt(res["gap"])))
            if res["gap"] <= 0.5:
                lines.append("Отрыв мал: объект лежит вблизи границы классов и может "
                             "быть перепутан (следствие 4 лекции).")
        else:
            top = max(d) if d else 0.0
            tied = [self.classes[j] for j in range(self.n_classes)
                    if abs(d[j] - top) <= TIE_EPS]
            lines.append("Максимальный голос %s делят классы: %s. Выбрать один "
                         "невозможно — объект НЕ опознан. Нужен различающий признак "
                         "или дообучение." % (fmt(top), ", ".join(tied)))
        sections.append(("Шаг 3. Принцип демона: выбираем максимум", "\n".join(lines)))

        # 4. решающие признаки
        if res["recognized"] and res["runner_up"]:
            w = res["winner"]
            s = self.classes.index(res["runner_up"][0])
            diffs = []
            for i in range(self.n_features):
                if x[i] == 0:
                    continue
                dv = (self.matrix[i, w] - self.matrix[i, s]) * x[i]
                diffs.append((self.features[i], dv, self.matrix[i, w] * x[i],
                              self.matrix[i, s] * x[i]))
            diffs.sort(key=lambda t: -t[1])
            lines = ["Сравниваем вклад каждого заданного признака в голос «%s» и "
                     "в голос «%s»:" % (self.classes[w], self.classes[s])]
            for f, dv, cw, cs in diffs:
                verdict = ("за «%s»" % self.classes[w] if dv > TIE_EPS else
                           ("за «%s»" % self.classes[s] if dv < -TIE_EPS
                            else "не различает"))
                lines.append("  %-16s %s → %s | %s → %s | разница %+s  (%s)"
                             % (f, self.classes[w][:12], fmt(cw),
                                self.classes[s][:12], fmt(cs), fmt(dv), verdict))
            pos = [f for f, dv, _, _ in diffs if dv > TIE_EPS]
            if pos:
                lines.append("Решающие признаки: %s — именно они дали «%s» перевес."
                             % (", ".join(pos), self.classes[w]))
            neg = [f for f, dv, _, _ in diffs if dv < -TIE_EPS]
            if neg:
                lines.append("Признаки «против»: %s — они тянули ответ к «%s»."
                             % (", ".join(neg), self.classes[s]))
            zero = [self.features[i] for i in range(self.n_features) if x[i] == 0]
            if zero:
                lines.append("Признаки со значением 0 на результат не влияют: %s."
                             % ", ".join(zero))
            sections.append(("Шаг 4. Какие признаки решили исход", "\n".join(lines)))

        # 5. геометрия
        lines = ["Каждое уравнение Yj = 0 — гиперплоскость в %d-мерном пространстве "
                 "признаков. Положение точки X относительно плоскостей:" % self.n_features]
        for cls, val, where in res["plane_check"]:
            mark = "  ← область точки" if res["recognized"] and cls == res["winner_name"] else ""
            lines.append("  %-16s Y = %7s  %s%s" % (cls, fmt(val), where, mark))
        if res["recognized"]:
            lines.append("Точка лежит в области класса «%s»: его плоскость даёт "
                         "наибольшее значение." % res["winner_name"])
        sections.append(("Шаг 5. Геометрическая интерпретация", "\n".join(lines)))

        # 6. похожесть в мире
        proto = self.prototypes()
        if proto:
            xs = np.asarray(x)
            dist = sorted(((float(np.abs(xs - v).sum()), c) for c, v in proto.items()))
            lines = ["Хэммингово расстояние от X до эталонов классов («мир как есть»):"]
            lines += ["  %-16s %s" % (c, fmt(dv)) for dv, c in dist[:5]]
            nearest = dist[0][1]
            if res["recognized"] and nearest != res["winner_name"]:
                lines.append("Ближайший эталон в мире — «%s», а система выбрала «%s»: "
                             "модель в «сознании» системы подобна миру, но не совпадает "
                             "с ним. Если это ошибка — дообучите систему."
                             % (nearest, res["winner_name"]))
            else:
                lines.append("Ближайший эталон «%s» — ответ системы согласуется с "
                             "расположением объектов в мире." % nearest)
            sections.append(("Шаг 6. Проверка по расстояниям в мире", "\n".join(lines)))

        res["sections"] = sections
        res["text"] = "\n\n".join("%s\n%s" % (t, b) for t, b in sections)
        res["summary"] = (
            "Опознан: %s (отрыв %s)" % (res["winner_name"], fmt(res["gap"]))
            if res["recognized"] else
            "Объект не опознан: равные максимальные голоса")
        return res

    def describe_presentation(self, p: Presentation) -> str:
        """Текст акта обучения в стиле лекции."""
        others = [c for k, c in enumerate(self.classes) if k != p.class_index]
        s = ["Предъявляем пример «%s»: X = %s." % (p.class_name, vec_str(p.vector))]
        s.append("Реакция системы (демоны до коррекции): %s → %s%s."
                 % (vec_str(p.demons_before), p.predicted_name,
                    " ✓" if p.ok else " ✗ (ошибка)"))
        if p.updated:
            s.append("Вектор %s%s прибавляется (поощрение!) к столбцу «%s» и "
                     "вычитается (наказание!) из столбцов %s."
                     % ("h·" if self.step != 1 else "", vec_str(p.vector),
                        p.class_name, ", ".join("«%s»" % c for c in others)))
            if p.clipped:
                s.append("Матрица вышла за гиперкуб [-1; 1] — нормализована.")
        else:
            s.append("Ответ верный — матрица не меняется.")
        return "\n".join(s)

    # ------------------------------------------------------------------
    # сохранение
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "features": list(self.features),
            "classes": list(self.classes),
            "examples": [[c, list(v)] for c, v in self.examples],
            "matrix": self.matrix.tolist(),
            "params": self.params(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExpertModel":
        params = data.get("params", {})
        m = cls(data["features"], data["classes"],
                [(c, v) for c, v in data["examples"]],
                step=params.get("step", 1.0),
                normalize=params.get("normalize", False),
                update_rule=params.get("update_rule", "always"),
                max_epochs=params.get("max_epochs", 30),
                auto_stop=params.get("auto_stop", True))
        if "matrix" in data:
            mat = np.array(data["matrix"], dtype=float)
            if mat.shape == (m.n_features, m.n_classes):
                m.matrix = mat
        return m

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "ExpertModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def clone(self) -> "ExpertModel":
        return copy.deepcopy(self)


# ---------------------------------------------------------------------------
# исследование шага обучения
# ---------------------------------------------------------------------------

def step_study(model: ExpertModel, steps=(0.1, 0.25, 0.5, 1.0, 2.0),
               normalize: bool = True, max_epochs: int = 40) -> list[dict]:
    """Обучить с нуля при разных шагах h и собрать кривые обучения.
    Без нормализации шаг на решение не влияет (argmax инвариантен к масштабу
    матрицы), поэтому исследование по умолчанию ведётся в гиперкубе [-1;1]."""
    out = []
    for h in steps:
        m = model.clone()
        m.step = float(h)
        m.normalize = normalize
        rep = m.train(max_epochs=max_epochs, auto_stop=True)
        out.append({"step": float(h), "epochs": rep.n_epochs,
                    "converged": rep.converged,
                    "curve_after": rep.curve_after,
                    "curve_during": rep.curve_during,
                    "final_errors": rep.final_errors,
                    "matrix": m.matrix.copy()})
    return out
