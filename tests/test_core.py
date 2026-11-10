# -*- coding: utf-8 -*-
"""Автотесты ядра экспертной системы (pytest)."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import expert_core as ec  # noqa: E402
from expert_core import ExpertModel  # noqa: E402


# ---------------------------------------------------------------------------
# пример из лекции: птица / самолёт / планер
# ---------------------------------------------------------------------------

LECTURE_FEATURES = ["крылья", "хвост", "клюв", "двигатель", "оперение", "шасси"]
LECTURE_CLASSES = ["птица", "самолет", "планер"]
LECTURE_EXAMPLES = [
    ("птица",   [1, 1, 1, 0, 1, 0]),
    ("самолет", [1, 1, 0, 1, 0, 1]),
    ("планер",  [1, 1, 0, 0, 0, 0]),
]


def lecture_model(**kw):
    return ExpertModel(LECTURE_FEATURES, LECTURE_CLASSES, LECTURE_EXAMPLES, **kw)


def test_lecture_iterations_reproduce_matrices():
    """Итерации 1–4 лекции: матрицы после каждого предъявления."""
    m = lecture_model()
    p1 = m.present(0, [1, 1, 1, 0, 1, 0])
    assert p1.predicted is None and not p1.ok and p1.updated
    assert m.matrix[:, 0].tolist() == [1, 1, 1, 0, 1, 0]
    assert m.matrix[:, 1].tolist() == [-1, -1, -1, 0, -1, 0]
    assert m.demons([1, 1, 1, 0, 1, 0]).tolist() == [4, -4, -4]

    m.present(1, [1, 1, 0, 1, 0, 1])
    expected2 = np.array([[0, 0, -2], [0, 0, -2], [1, -1, -1],
                          [-1, 1, -1], [1, -1, -1], [-1, 1, -1]], dtype=float)
    assert np.array_equal(m.matrix, expected2)
    assert m.demons([1, 1, 1, 0, 1, 0]).tolist() == [2, -2, -6]
    assert m.demons([1, 1, 0, 1, 0, 1]).tolist() == [-2, 2, -6]

    m.present(2, [1, 1, 0, 0, 0, 0])
    expected3 = np.array([[-1, -1, -1], [-1, -1, -1], [1, -1, -1],
                          [-1, 1, -1], [1, -1, -1], [-1, 1, -1]], dtype=float)
    assert np.array_equal(m.matrix, expected3)
    # планер (−2, −2, −2) — не опознан
    winner, gap = m.decide(m.demons([1, 1, 0, 0, 0, 0]))
    assert winner is None

    # итерация 4: повторение примера + нормализация
    m.present(2, [1, 1, 0, 0, 0, 0])
    assert m.matrix[0].tolist() == [-2, -2, 0]
    m.matrix = np.clip(m.matrix, -1, 1)
    assert m.demons([1, 1, 1, 0, 1, 0]).tolist() == [0, -4, -2]
    assert m.demons([1, 1, 0, 1, 0, 1]).tolist() == [-4, 0, -2]
    assert m.demons([1, 1, 0, 0, 0, 0]).tolist() == [-2, -2, 0]
    assert m.errors_now() == 0
    # прогноз: ракета опознаётся как самолёт
    r = m.expertise([0, 1, 0, 1, 0, 0], "ракета")
    assert r["winner_name"] == "самолет"
    assert r["demons"] == [-2, 0, -1]
    # расстояния «мир как есть» и «в сознании»
    assert m.world_distances()[("птица", "самолет")] == 4
    assert m.mind_distances()[("птица", "самолет")] == 8


def test_lecture_train_converges_on_errors_only():
    m = lecture_model()
    rep = m.train()
    assert rep.converged
    assert m.errors_now() == 0
    # в успешном случае ответа модель не меняется
    before = m.matrix.copy()
    m.present(0, [1, 1, 1, 0, 1, 0])
    assert np.array_equal(before, m.matrix)


# ---------------------------------------------------------------------------
# база лабораторной: 8 инструментов
# ---------------------------------------------------------------------------

def test_default_training_converges():
    m = ExpertModel()
    rep = m.train()
    assert rep.converged and rep.n_epochs == 4
    assert rep.curve_after == [2, 4, 1, 0]
    assert all(r["ok"] for r in m.check_all())
    assert m.matrix.shape == (8, 8)


def test_always_rule_reproduces_colab_matrix():
    m = ExpertModel(update_rule="always")
    m.train(max_epochs=4, auto_stop=False)
    assert m.matrix[0].tolist() == [-8, -8, -8, -16, -16, -16, -8, -16]
    assert m.matrix[1].tolist() == [4, -4, -4, -4, -4, -4, -4, -4]
    assert m.equation(0).startswith("y1 = -8·x1 + 4·x2 - 8·x3")


def test_tie_is_unrecognized():
    m = ExpertModel()
    r = m.expertise([0] * 8)
    assert not r["recognized"] and r["winner"] is None
    assert "не опознан" in r["winner_name"]


def test_unknown_objects_prognosis():
    m = ExpertModel()
    m.train()
    assert m.expertise(ec.UNKNOWN_OBJECTS["Виолончель"])["winner_name"] == "Скрипка"
    assert m.expertise(ec.UNKNOWN_OBJECTS["Саксофон"])["winner_name"] == "Флейта"
    r = m.expertise([1, 0, 1, 0, 0, 0, 0, 1], "Укулеле")
    assert r["known_as"] == "Гитара"


def test_unknown_feature_value_half():
    m = ExpertModel()
    m.train()
    r = m.expertise([1, ec.UNKNOWN_VALUE, 0, 0, 0, 0, 0, 1])
    assert r["recognized"]


def test_feedback_adds_example_and_relearns():
    m = ExpertModel()
    m.train()
    vec = [1, 0, 1, 0, 1, 0, 0, 1]
    assert not m.expertise(vec)["recognized"]
    rep = m.learn_feedback(vec, "Клавесин")
    assert rep.kind == "feedback" and rep.converged
    assert rep.added_example == ("Клавесин", [float(v) for v in vec])
    assert "Клавесин" in m.classes
    assert m.expertise(vec)["winner_name"] == "Клавесин"
    assert m.errors_now() == 0


def test_explain_sections_and_decisive_feature():
    m = ExpertModel()
    m.train()
    r = m.explain([1, 1, 0, 0, 0, 0, 0, 1], "Скрипка")
    titles = [t for t, _ in r["sections"]]
    assert len(titles) >= 5 and titles[0].startswith("Шаг 1")
    step4 = dict(r["sections"])["Шаг 4. Какие признаки решили исход"]
    assert "Смычковый" in step4 and "Решающие признаки" in step4
    assert "Скрипка" in r["summary"]
    assert r["text"]


def test_edit_base_and_roundtrip(tmp_path):
    m = ExpertModel()
    m.add_feature("Медный")
    m.add_class("Труба")
    m.add_example("Труба", [0, 0, 0, 1, 0, 0, 0, 0, 1])
    assert m.matrix.shape == (9, 9)
    rep = m.train()
    assert rep.converged and m.errors_now() == 0
    path = tmp_path / "base.json"
    m.save_json(str(path))
    m2 = ExpertModel.load_json(str(path))
    assert m2.features == m.features and m2.classes == m.classes
    assert np.array_equal(m2.matrix, m.matrix)
    assert m2.errors_now() == 0
    m2.remove_feature(8)
    m2.remove_class(8)
    assert m2.matrix.shape == (8, 8) and len(m2.examples) == 8
    with pytest.raises(ValueError):
        m2.add_class("Гитара")


def test_problems_detects_indistinguishable_and_uninformative():
    m = ExpertModel(["a", "b"], ["K1", "K2"], [("K1", [1, 0]), ("K2", [1, 0])])
    notes = " ".join(m.problems())
    assert "неразличимы" in notes and "неинформативен" in notes
    rep = m.train(max_epochs=5)
    assert not rep.converged


def test_step_is_irrelevant_without_normalization():
    curves = []
    for h in (0.1, 1.0, 5.0):
        m = ExpertModel(step=h)
        curves.append(m.train().curve_after)
    assert curves[0] == curves[1] == curves[2]


def test_step_study_with_normalization():
    rows = ec.step_study(ExpertModel(), steps=(0.1, 0.25, 1.0, 2.0), max_epochs=40)
    by_step = {r["step"]: r for r in rows}
    assert by_step[0.1]["converged"] and by_step[0.25]["converged"]
    assert not by_step[1.0]["converged"] and not by_step[2.0]["converged"]


def test_normalize_clips_and_flags():
    m = ExpertModel(step=2.0, normalize=True)
    p = m.present(0, [1, 1, 0, 0, 0, 0, 0, 1])
    assert p.clipped and np.abs(m.matrix).max() <= 1.0


def test_fmt():
    assert ec.fmt(4.0) == "4" and ec.fmt(-0.5) == "-0.5" and ec.fmt(1.25) == "1.25"
    assert ec.vec_str([1, 0.5, -2]) == "(1,0.5,-2)"
