# -*- coding: utf-8 -*-
"""Вкладка «📖 Объяснение»: пошаговый разбор того, как система пришла к
ответу — текст пригоден для вставки в отчёт по лабораторной работе."""
from __future__ import annotations

import datetime as _dt
import re
import tkinter as tk
from tkinter import ttk, filedialog

import expert_core as ec
import ui_theme as th
from ui_theme import COLORS, FONTS

KIND_TEXT = {"scratch": "обучение с нуля", "continue": "дообучение поверх матрицы",
             "feedback": "дообучение по обратной связи «мира»"}
RULE_TEXT = {"on_error": "только при ошибке", "always": "при каждом предъявлении"}
STEP_NOTE = ("Без нормализации шаг h не влияет на решение (argmax инвариантен к масштабу "
             "матрицы); при нормализации в гиперкуб [-1; 1] малый шаг — консерватизм, "
             "большой — «разболтанность» (см. вкладку Графики → Исследование шага).")


def _chain(values) -> str:
    return " → ".join(str(v) for v in values)


class ExplainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self._key = None
        self._stamp = ""
        card = th.Card(self, "Подробное объяснение",
                       "Как система получила ответ и откуда взялись её коэффициенты — "
                       "текст можно скопировать или сохранить для отчёта.")
        card.pack(fill="both", expand=True)
        bar = tk.Frame(card.body, background=COLORS["card"])
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="Скопировать текст", command=self.copy).pack(side="left")
        ttk.Button(bar, text="Сохранить в файл…", command=self.save).pack(side="left", padx=8)
        self.info = th.label(bar, "", "small", "muted")
        self.info.pack(side="right")
        th.hsep(card.body, 2)
        self.text = th.RichText(card.body, padx=16, pady=10)
        vsb = ttk.Scrollbar(card.body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)
        self.text.tag_configure("summary", font=FONTS["bold"], background=COLORS["accent_bg"],
                                lmargin1=10, lmargin2=10, rmargin=10, spacing1=6, spacing3=6)
        self.text.tag_configure("bullet", lmargin1=14, lmargin2=28, spacing3=4)
        self.text.tag_configure("gap", font=FONTS["mono_small"], spacing1=0, spacing3=0)
        self.refresh()

    # ------------------------------------------------------------------
    # действия
    # ------------------------------------------------------------------

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.text.text())
        self.info.configure(text="✔ текст скопирован в буфер обмена")
        self.after(2500, self._update_info)

    def save(self):
        res = self.app.last_result
        base = re.sub(r"[^\w-]+", "_", res["name"]) if res else "модель"
        path = filedialog.asksaveasfilename(
            parent=self, title="Сохранить объяснение", defaultextension=".txt",
            initialfile="объяснение_%s.txt" % base,
            filetypes=[("Текстовый файл", "*.txt"), ("Все файлы", "*.*")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.text.text() + "\n")
        self.app.log("Объяснение сохранено в файл: %s" % path, "ok")

    def _update_info(self):
        res = self.app.last_result
        self.info.configure(text=("объект «%s» · %s" % (res["name"], self._stamp))
                            if res else "экспертиза ещё не выполнена")

    # ------------------------------------------------------------------
    # построение текста
    # ------------------------------------------------------------------

    def refresh(self):
        m, t = self.app.model, self.text
        res = self.app.last_result
        if res is not None and len(res["input"]) != m.n_features:
            res = None
        key = id(res)
        same = key == self._key
        if not same:
            self._key = key
            self._stamp = _dt.datetime.now().strftime("%H:%M:%S")
        top = t.yview()[0] if same else 0.0
        t.clear()
        if res is None:
            self._build_empty()
        else:
            self._build_result(res)
        if m.trained:
            self._build_training()
        if res is not None:
            self._build_world(res)
            self._build_conclusions(res)
        t.yview_moveto(top)
        self._update_info()

    def _build_empty(self):
        t = self.text
        t.heading("Объяснение появится после экспертизы", 1)
        t.para("На вкладке «Экспертиза» задайте признаки объекта и нажмите «Определить "
               "инструмент». Здесь появится разбор по шагам: входной вектор X, умножение "
               "на матрицу коэффициентов, голоса демонов, решающие признаки, геометрия "
               "и проверка по расстояниям в мире.")
        t.configure(state="normal")
        btn = ttk.Button(t, text="Перейти к экспертизе", style="Accent.TButton",
                         command=lambda: self.app.select_tab("expertise"))
        t.window_create("end", window=btn, padx=14, pady=8)
        t.insert("end", "\n")
        t.configure(state="disabled")

    def _build_result(self, res):
        t = self.text
        if res["recognized"]:
            t.heading("Почему система ответила «%s»?" % res["winner_name"], 1)
        else:
            t.heading("Почему объект не опознан?", 1)
        t.add("%s  ·  объект «%s», X = %s" % (res["summary"], res["name"],
                                              ec.vec_str(res["input"])), "summary")
        for title, body in res["sections"]:
            t.heading(title, 2)
            t.mono_block(body)

    def _build_training(self):
        t, m = self.text, self.app.model
        rep = self.app.last_report
        t.heading("Как система получила эти коэффициенты (режим обучения)", 2)
        lines = ["Параметры: шаг h = %s; нормализация в гиперкуб [-1; 1]: %s; коррекция: %s; "
                 "эпох за всё время: %d."
                 % (ec.fmt(m.step), "включена" if m.normalize else "выключена",
                    RULE_TEXT.get(m.update_rule, m.update_rule), m.epochs_done)]
        if rep is not None:
            lines.append("Последний запуск — %s: %d эп., %s (%s). Ошибок после каждой эпохи: %s."
                         % (KIND_TEXT.get(rep.kind, rep.kind), rep.n_epochs,
                            "сошлось" if rep.converged else "не сошлось", rep.reason,
                            _chain(rep.curve_after)))
        shown = sum(len(e.presentations) for e in m.history)
        fixes = sum(1 for e in m.history for p in e.presentations if p.updated)
        lines.append("Матрица корректировалась %d раз из %d предъявлений примеров "
                     "(в успешном случае модель остаётся неизменной)." % (fixes, shown))
        t.para("\n".join(lines))
        t.heading("Первые акты обучения (как в лекции)", 3)
        if not m.history:
            t.para("История обучения пуста: матрица загружена из файла или ещё не обучалась.",
                   "muted")
        else:
            for p in m.history[0].presentations[:3]:
                t.para(m.describe_presentation(p))
                t.mono_block(self._matrix_text(p.matrix_after))
                t.add("", "gap")
        t.heading("Итоговая система уравнений", 3)
        t.mono_block("\n".join(m.equations()))

    def _matrix_text(self, mat) -> str:
        m = self.app.model
        labels = ["x%d %s" % (i + 1, f) for i, f in enumerate(m.features)]
        w = max(len(s) for s in labels)
        head = [" " * w + " |"] * 2
        for j, c in enumerate(m.classes):
            head[0] += " %7s" % c[:7]
            head[1] += " %7s" % ("Y%d" % (j + 1))
        rows = ["%-*s |" % (w, lab) + "".join(" %7s" % ec.fmt(v) for v in mat[i])
                for i, lab in enumerate(labels)]
        return "\n".join(head + rows)

    def _build_world(self, res):
        t, m = self.text, self.app.model
        focus = ([res["winner_name"]] if res["recognized"]
                 else [n for n, _ in res["ranking"][:3]])
        world, mind = m.world_distances(), m.mind_distances()
        pairs = []
        for (a, b), dw in world.items():
            if a in focus or b in focus:
                dm = mind.get((a, b), mind.get((b, a), 0.0))
                pairs.append((dw, "%s — %s" % (a, b), dm))
        pairs.sort(key=lambda r: (r[0], r[1]))
        t.heading("Мир как есть и мир в сознании системы", 2)
        t.para("Расстояния от %s до остальных классов (первые %d пар по близости в мире)."
               % ("класса «%s»" % focus[0] if len(focus) == 1
                  else "лидирующих классов «%s»" % "», «".join(focus), min(8, len(pairs))))
        if pairs:
            w = max(len(r[1]) for r in pairs[:8] + [(0, "Пара классов", 0)])
            lines = ["%-*s  %-18s  %s" % (w, "Пара классов", "в мире (Хэмминг)",
                                          "в сознании (|Δ столбцов|)")]
            lines += ["%-*s  %-18s  %s" % (w, name, ec.fmt(dw), ec.fmt(dm))
                      for dw, name, dm in pairs[:8]]
            t.mono_block("\n".join(lines))
        t.para("Расстояния подобны, но не совпадают: модель — подобие мира, а не его копия. "
               "Повторение — мать учения.")

    def _bullet(self, *parts):
        t = self.text
        t.add("•  ", "bullet", newline=False)
        for text, tag in parts[:-1]:
            t.add(text, tag, newline=False)
        t.add(parts[-1][0], parts[-1][1])

    def _build_conclusions(self, res):
        t, m = self.text, self.app.model
        t.heading("Выводы", 2)
        total, bad = len(m.examples), m.errors_now()
        if not m.trained:
            self._bullet(("Модель не обучена: матрица пуста, все голоса равны нулю.", "err"))
        elif bad == 0:
            self._bullet(("Адекватность: ", "bold"),
                         ("система верно опознаёт %d из %d примеров базы." % (total, total), "ok"))
        else:
            self._bullet(("Адекватность: ", "bold"),
                         ("верно %d из %d, ошибок %d — нужно дообучение." % (total - bad, total, bad), "err"))
        curve = [e.errors_after for e in m.history]
        if curve:
            self._bullet(("Кривая обучения (ошибок после эпохи): ", "bold"),
                         (_chain(curve) + ".", "body"))
        if m.trained:
            imp = m.feature_importance()
            dead = [f for f, r in imp if r <= ec.TIE_EPS]
            if dead:
                self._bullet(("Неинформативные признаки: ", "bold"),
                             ("%s — ни на что не влияют (одинаковые коэффициенты у всех классов)."
                              % ", ".join(dead), "body"))
            strong = sorted((x for x in imp if x[1] > ec.TIE_EPS), key=lambda r: -r[1])[:2]
            if strong:
                self._bullet(("Наиболее влиятельные признаки: ", "bold"),
                             (", ".join("%s (размах %s)" % (f, ec.fmt(r)) for f, r in strong) + ".", "body"))
            mind = m.mind_distances()
            if mind:
                (a, b), d = min(mind.items(), key=lambda kv: (kv[1], kv[0]))
                self._bullet(("Ближе всего в «сознании» системы классы «%s» и «%s» " % (a, b), "bold"),
                             ("(|Δ столбцов| = %s) — их легче всего перепутать." % ec.fmt(d), "body"))
        self._bullet(("Шаг обучения. ", "bold"), (STEP_NOTE, "body"))
