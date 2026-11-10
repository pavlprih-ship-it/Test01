# -*- coding: utf-8 -*-
"""Вкладка «🔍 Экспертиза»: ввод признаков объекта, ответ системы по
принципу демона, обратная связь «мира» и проверка всех примеров базы."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import expert_core as ec
import ui_theme as th
from ui_theme import COLORS, FONTS

DEFAULT_NAME = "объект"
CHOICES = (("есть", 1.0), ("нет", 0.0), ("не знаю", ec.UNKNOWN_VALUE))
DEBOUNCE_MS = 150
POLL_MS = 400
LEFT_W = 440
HERO_W = 300


def _num(v) -> str:
    return "∞" if v == float("inf") else ec.fmt(v)


def _epochs(n: int) -> str:
    if n % 10 == 1 and n % 100 != 11:
        word = "эпоху"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "эпохи"
    else:
        word = "эпох"
    return "%d %s" % (n, word)


class ExpertiseTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.vars: list[tk.DoubleVar] = []
        self._name_labels: list[tk.Label] = []
        self._feature_key = None
        self._seen = None            # (вход, имя) результата, уже показанного в полях
        self._job = self._poll_job = None
        self._rendered = None
        self._syncing = False
        self._bar_items: list = []
        self._bar_base = 0.0
        self._bar_highlight = None
        self._bar_width = 0
        self.name_var = tk.StringVar(value=DEFAULT_NAME)
        self.auto_var = tk.BooleanVar(value=True)
        self.known_var = tk.StringVar()
        self.unknown_var = tk.StringVar()
        self.truth_var = tk.StringVar()

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        left = th.ScrollFrame(self)
        left.canvas.configure(width=LEFT_W)
        left.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        right = th.ScrollFrame(self)
        right.grid(row=0, column=1, sticky="nsew")
        self._build_input(left.inner)
        self._build_quick(left.inner)
        self._build_answer(right.inner)
        self._build_votes(right.inner)
        self._build_feedback(right.inner)
        self._build_check(right.inner)
        self.refresh()
        self._poll_job = self.after(POLL_MS, self._poll)

    # ------------------------------------------------------------------
    # построение интерфейса
    # ------------------------------------------------------------------

    def _build_input(self, parent):
        card = th.Card(parent, "Задайте критерии инструмента",
                       "Для каждого признака отметьте: есть, нет или не знаю.")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        body = card.body
        row = tk.Frame(body, background=COLORS["card"])
        row.pack(fill="x", pady=(0, 6))
        th.label(row, "Название объекта").pack(side="left")
        entry = ttk.Entry(row, textvariable=self.name_var)
        entry.pack(side="left", fill="x", expand=True, padx=(10, 0))
        entry.bind("<Return>", lambda _e: self._run())
        th.hsep(body, 4)
        self.rows = tk.Frame(body, background=COLORS["card"])
        self.rows.pack(fill="x")
        th.hsep(body, 8)
        auto = tk.Frame(body, background=COLORS["card"])
        auto.pack(fill="x", pady=(0, 8))
        check = ttk.Checkbutton(auto, text="", style="Card.TCheckbutton",
                                variable=self.auto_var, command=self._on_auto)
        check.pack(side="left")
        txt = th.label(auto, "Определять автоматически при изменении признаков",
                       wraplength=LEFT_W - 100, cursor="hand2")
        txt.pack(side="left")
        txt.bind("<Button-1>", lambda _e: check.invoke())
        btns = tk.Frame(body, background=COLORS["card"])
        btns.pack(fill="x")
        ttk.Button(btns, text="🔍 Определить инструмент", style="Accent.TButton",
                   command=self._run).pack(side="left", fill="x", expand=True, ipady=6)
        ttk.Button(btns, text="Сбросить", command=self._reset
                   ).pack(side="left", padx=(8, 0), ipady=6)

    def _build_quick(self, parent):
        card = th.Card(parent, "Быстрая подстановка")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        body = card.body
        th.label(body, "Известный класс из базы", "bold").pack(anchor="w")
        self.known_cb = ttk.Combobox(body, textvariable=self.known_var, state="readonly")
        self.known_cb.pack(fill="x", pady=(2, 10))
        self.known_cb.bind("<<ComboboxSelected>>", self._pick_known)
        th.label(body, "Неизвестный объект (прогноз)", "bold").pack(anchor="w")
        self.unknown_cb = ttk.Combobox(body, textvariable=self.unknown_var, state="readonly")
        self.unknown_cb.pack(fill="x", pady=(2, 4))
        self.unknown_cb.bind("<<ComboboxSelected>>", self._pick_unknown)
        th.label(body, "Система этих объектов не видела — она ищет наиболее "
                 "похожий класс.", "small", "muted", wraplength=LEFT_W - 70
                 ).pack(anchor="w")

    def _build_answer(self, parent):
        self.train_banner = th.StatusBanner(parent)
        card = th.Card(parent, "Ответ системы",
                       "Побеждает демон с максимальным голосом — «кто громче крикнет, тот и прав».")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        self.answer_card = card
        body = card.body
        self.placeholder = th.label(body, "Задайте признаки слева — ответ появится здесь",
                                    "base", "muted", pady=26)
        self.result_frame = tk.Frame(body, background=COLORS["card"])
        rf = self.result_frame
        rf.columnconfigure(1, weight=1)
        hero = tk.Frame(rf, background=COLORS["card"])
        hero.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        self.stripe = tk.Frame(hero, width=5, background=COLORS["primary"])
        self.stripe.pack(side="left", fill="y")
        box = tk.Frame(hero, background=COLORS["card"])
        box.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.obj_lbl = th.label(box, "", "small", "muted", wraplength=HERO_W)
        self.obj_lbl.pack(anchor="w")
        self.big_lbl = th.label(box, "", "big", "primary", wraplength=HERO_W)
        self.big_lbl.pack(anchor="w", pady=(2, 4))
        self.conf_lbl = th.label(box, "", wraplength=HERO_W)
        self.conf_lbl.pack(anchor="w", pady=(0, 8))
        self.known_badge = th.Badge(box)
        self.known_badge.pack(anchor="w")
        self.stale_badge = th.Badge(box, "", "warning", wraplength=HERO_W, justify="left")
        self.explain_btn = ttk.Button(box, text="📖 Подробное объяснение",
                                      command=lambda: self.app.select_tab("explain"))
        self.explain_btn.pack(anchor="w", pady=(10, 0))
        col = tk.Frame(rf, background=COLORS["card"])
        col.grid(row=0, column=1, sticky="nsew")
        th.label(col, "Голоса демонов Yj", "h3").pack(anchor="w")
        th.label(col, "длина полоски — голос относительно самого «тихого» демона",
                 "small", "muted").pack(anchor="w")
        self.bars = th.bar_canvas(col)
        self.bars.pack(fill="x", pady=(4, 0))
        self.bars.bind("<Configure>", self._draw_bars)

    def _build_votes(self, parent):
        card = th.Card(parent, "Таблица голосов демонов",
                       "Yj = Σ aij·xi — столбец матрицы, умноженный на вектор признаков.")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        frame, self.votes = th.make_table(
            card.body, ["№", "Класс", "Голос Yj", "Отрыв от лидера", "Итог"],
            [50, 200, 110, 140, 180], 8, ["center", "w", "center", "center", "w"])
        frame.pack(fill="x")

    def _build_feedback(self, parent):
        card = th.Card(parent, "Обратная связь «мира»",
                       "Ошибку системы исправляет «мир»: пример запоминается и система "
                       "дообучается, пока не станет опознавать все примеры.")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        body = card.body
        row = tk.Frame(body, background=COLORS["card"])
        row.pack(fill="x", pady=(0, 8))
        th.label(row, "Ответ верный?", "bold").pack(side="left", padx=(0, 12))
        self.ok_btn = ttk.Button(row, text="👍 Верно", command=self._confirm)
        self.ok_btn.pack(side="left")
        row = tk.Frame(body, background=COLORS["card"])
        row.pack(fill="x")
        th.label(row, "👎 Неверно, на самом деле это:", "bold").pack(side="left")
        self.truth_cb = ttk.Combobox(row, textvariable=self.truth_var, width=22)
        self.truth_cb.pack(side="left", padx=8)
        self.truth_cb.bind("<Return>", lambda _e: self._retrain())
        self.retrain_btn = ttk.Button(row, text="Дообучить", style="Accent.TButton",
                                      command=self._retrain)
        self.retrain_btn.pack(side="left")
        th.label(body, "Выберите класс из списка или введите новое название — система "
                 "добавит класс и пример для него.", "small", "muted", wraplength=640
                 ).pack(anchor="w", pady=(6, 0))
        self.fb_banner = th.StatusBanner(body)

    def _build_check(self, parent):
        card = th.Card(parent, "Экспертиза всех известных классов",
                       "Проверка адекватности: каждый пример базы предъявляется системе, "
                       "её ответ сравнивается с эталоном.")
        card.pack(fill="x", pady=(0, 10), padx=(0, 6))
        body = card.body
        row = tk.Frame(body, background=COLORS["card"])
        row.pack(fill="x", pady=(0, 8))
        ttk.Button(row, text="Проверить все примеры базы", command=self._check_clicked
                   ).pack(side="left")
        self.check_banner = th.StatusBanner(body)
        self.check_banner.pack(fill="x", pady=(0, 8))
        frame, self.check = th.make_table(
            body, ["Класс", "Вектор", "Ответ системы", "Отрыв Δ", "Статус"],
            [140, 200, 150, 90, 110], 8, ["w", "center", "w", "center", "w"])
        frame.pack(fill="x")

    # ------------------------------------------------------------------
    # признаки и поля ввода
    # ------------------------------------------------------------------

    def _vector(self) -> list:
        out = []
        for v in self.vars:
            try:
                out.append(float(v.get()))
            except tk.TclError:
                out.append(0.0)
        return out

    def _set_vector(self, vec):
        self._syncing = True
        try:
            for var, x in zip(self.vars, vec):
                x = float(x)
                var.set(1.0 if x >= 0.99 else 0.0 if abs(x) < 0.01 else ec.UNKNOWN_VALUE)
        finally:
            self._syncing = False
        self._style_rows()

    def _sync_features(self):
        key = tuple(self.app.model.features)
        if key == self._feature_key:
            return
        old = dict(zip(self._feature_key or (), self._vector()))
        for w in self.rows.winfo_children():
            w.destroy()
        self.vars, self._name_labels = [], []
        for i, name in enumerate(key):
            var = tk.DoubleVar(value=old.get(name, 0.0))
            row = tk.Frame(self.rows, background=COLORS["card"])
            row.pack(fill="x")
            for text, val in reversed(CHOICES):      # радио первыми — их не обрежет длинное имя
                ttk.Radiobutton(row, text=text, value=val, variable=var,
                                style="Card.TRadiobutton", command=self._on_change
                                ).pack(side="right", padx=(4, 0), pady=1)
            th.label(row, "x%d" % (i + 1), "mono_small", "muted", width=3).pack(side="left")
            lbl = th.label(row, name)
            lbl.pack(side="left", fill="x", expand=True)
            tk.Frame(self.rows, height=1, background=COLORS["row_alt"]).pack(fill="x")
            self.vars.append(var)
            self._name_labels.append(lbl)
        self._feature_key = key
        self._seen = None
        self._style_rows()

    def _style_rows(self):
        for var, lbl in zip(self.vars, self._name_labels):
            try:
                v = float(var.get())
            except tk.TclError:
                v = 0.0
            if v >= 0.99:
                lbl.configure(font=FONTS["bold"], foreground=COLORS["primary"])
            elif v <= 0.01:
                lbl.configure(font=FONTS["base"], foreground=COLORS["muted"])
            else:
                lbl.configure(font=FONTS["bold"], foreground=COLORS["accent"])

    def _fill_combos(self):
        m = self.app.model
        self._known = ["%s   %s" % (c, ec.vec_str(v)) for c, v in m.examples]
        self.known_cb.configure(values=self._known)
        self._unknown = [(n, v) for n, v in ec.UNKNOWN_OBJECTS.items()
                         if len(v) == m.n_features]
        self.unknown_cb.configure(values=[n for n, _ in self._unknown])
        self.truth_cb.configure(values=list(m.classes))

    def _sync_inputs(self, res):
        """Поля ввода следуют за результатом, полученным не на этой вкладке."""
        if res is None:
            self._seen = None
            return
        key = (tuple(res["input"]), res["name"])
        if key == self._seen:
            return
        self._seen = key
        self._set_vector(res["input"])
        self.name_var.set(res["name"])

    # ------------------------------------------------------------------
    # действия пользователя
    # ------------------------------------------------------------------

    def _on_change(self):
        if self._syncing:
            return
        self.known_var.set("")
        self.unknown_var.set("")
        self._style_rows()
        if self.auto_var.get():
            self._schedule()
        else:
            self._update_stale(self._current_result())

    def _on_auto(self):
        if self.auto_var.get():
            res = self._current_result()
            if res is None or list(res["input"]) != self._vector():
                self._schedule()
        else:
            self._update_stale(self._current_result())

    def _schedule(self):
        self._cancel_job()
        self._job = self.after(DEBOUNCE_MS, self._run)

    def _cancel_job(self):
        if self._job is not None:
            self.after_cancel(self._job)
            self._job = None

    def _poll(self):
        # run_expertise() обновляет не все вкладки — подхватываем результат, полученный извне
        if self.app.last_result is not self._rendered:
            self.refresh()
        self._poll_job = self.after(POLL_MS, self._poll)

    def destroy(self):
        for job in (self._job, self._poll_job):
            if job is not None:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self._job = self._poll_job = None
        super().destroy()

    def _run(self):
        self._job = None
        self._sync_features()
        name = self.name_var.get().strip() or DEFAULT_NAME
        res = self.app.run_expertise(self._vector(), name)
        self._seen = (tuple(res["input"]), res["name"])
        self.fb_banner.pack_forget()
        self.refresh()

    def _reset(self):
        self._cancel_job()
        self._set_vector([0.0] * len(self.vars))
        self.name_var.set(DEFAULT_NAME)
        for var in (self.known_var, self.unknown_var, self.truth_var):
            var.set("")
        self.fb_banner.pack_forget()
        self._seen = None
        self.app.last_result = None
        self.app.refresh_all()

    def _pick_known(self, _event=None):
        idx = self.known_cb.current()
        if not 0 <= idx < len(self.app.model.examples):
            return
        cls, vec = self.app.model.examples[idx]
        self.unknown_var.set("")
        self._load_and_run(cls, vec)

    def _pick_unknown(self, _event=None):
        idx = self.unknown_cb.current()
        if not 0 <= idx < len(self._unknown):
            return
        name, vec = self._unknown[idx]
        self.known_var.set("")
        self._load_and_run(name, vec)

    def _load_and_run(self, name, vec):
        self._cancel_job()
        self._sync_features()
        self._set_vector(vec)
        self.name_var.set(name)
        self._run()

    def _current_result(self):
        res = self.app.last_result
        if res is not None and len(res["input"]) != self.app.model.n_features:
            return None
        return res

    def _show_feedback(self, kind, title, text=""):
        self.fb_banner.set(kind, title, text)
        self.fb_banner.pack(fill="x", pady=(10, 0))

    def _confirm(self):
        res = self._current_result()
        if res is None:
            return
        if not res["recognized"]:
            self._show_feedback("warning", "«Не опознан» нельзя подтвердить",
                                "Укажите правильный класс ниже и нажмите «Дообучить».")
            return
        self.app.log("Мир подтвердил ответ «%s» для объекта «%s» %s — модель не меняется."
                     % (res["winner_name"], res["name"], ec.vec_str(res["input"])), "ok")
        self._show_feedback("success", "Мир подтвердил ответ «%s»" % res["winner_name"],
                            "Модель не меняется — в успешном случае ответа модель "
                            "остаётся неизменной.")

    def _retrain(self):
        res = self._current_result()
        if res is None:
            return
        cls = self.truth_var.get().strip()
        if not cls:
            self._show_feedback("warning", "Укажите правильный класс",
                                "Выберите его из списка или введите новое название.")
            return
        is_new = cls not in self.app.model.classes
        rep = self.app.feedback(res["input"], cls)
        if rep is None:
            self._show_feedback("danger", "Дообучение невозможно", "Подробности — в журнале.")
            return
        now = self.app.last_result
        notes = []
        if is_new:
            notes.append("Добавлен новый класс «%s» и пример для него." % cls)
        elif rep.added_example:
            notes.append("Пример добавлен в базу знаний для класса «%s»." % cls)
        else:
            notes.append("Такой пример уже был в базе — система доучилась на нём.")
        answer = now["winner_name"] if now else cls
        if rep.converged and answer == cls:
            self._show_feedback(
                "success", "Система дообучена за %s; теперь объект опознаётся как «%s»"
                % (_epochs(rep.n_epochs), answer), " ".join(notes))
        else:
            notes.append("Возможно, у классов совпадают признаки — добавьте различающий "
                         "признак в «Базе знаний».")
            self._show_feedback("warning", "Дообучение не сошлось: %s" % rep.reason,
                                " ".join(notes))
        self.truth_var.set("")

    def _check_clicked(self):
        rows = self.app.model.check_all()
        bad = sum(1 for r in rows if not r["ok"])
        self.app.log("Проверка всех примеров базы: верно %d из %d."
                     % (len(rows) - bad, len(rows)), "ok" if bad == 0 else "err")
        self._fill_check()

    # ------------------------------------------------------------------
    # отрисовка результата
    # ------------------------------------------------------------------

    def refresh(self):
        m = self.app.model
        self._sync_features()
        self._fill_combos()
        res = self._current_result()
        self._rendered = self.app.last_result
        self._sync_inputs(res)
        if m.trained:
            self.train_banner.pack_forget()
        else:
            self.train_banner.set("warning", "Модель ещё не обучена",
                                  "Перейдите на вкладку «Обучение»: пока матрица пуста, "
                                  "все голоса демонов равны нулю.")
            self.train_banner.pack(fill="x", pady=(0, 10), padx=(0, 6),
                                   before=self.answer_card)
        self._render_result(res)
        self._fill_votes(res)
        self._fill_check()
        state = "normal" if res is not None else "disabled"
        for w in (self.ok_btn, self.retrain_btn, self.truth_cb):
            w.configure(state=state)

    def _render_result(self, res):
        if res is None:
            self.result_frame.pack_forget()
            self.placeholder.pack(fill="x")
            self._bar_items = []
            return
        self.placeholder.pack_forget()
        self.result_frame.pack(fill="x")
        ok = res["recognized"]
        color = COLORS["primary" if ok else "danger"]
        self.stripe.configure(background=color)
        self.obj_lbl.configure(text="Объект «%s» · X = %s" % (res["name"], ec.vec_str(res["input"])))
        self.big_lbl.configure(text=res["winner_name"] if ok else "Объект не опознан",
                               foreground=color)
        ranking = res["ranking"]
        if ok:
            ru = res["runner_up"]
            if ru is None:
                text = "В базе один класс — отрыв не определён."
            else:
                text = ("Уверенность — отрыв от «второго места» (%s, голос %s): Δ = %s"
                        % (ru[0], ec.fmt(ru[1]), _num(res["gap"])))
                if res["gap"] <= 0.5:
                    text += "\nОбъект вблизи границы классов — возможна ошибка."
        else:
            top = ranking[0][1]
            tied = [n for n, v in ranking if abs(v - top) <= ec.TIE_EPS]
            text = ("Максимальный голос %s делят классы: %s. Выбрать один невозможно — "
                    "нужен различающий признак или дообучение." % (ec.fmt(top), ", ".join(tied)))
        self.conf_lbl.configure(text=text)
        if res["known_as"]:
            self.known_badge.set("эталон из базы: совпадает с «%s»" % res["known_as"], "info")
        else:
            self.known_badge.set("новый объект — прогноз", "accent")
        self._update_stale(res)
        # длина полоски — голос относительно самого «тихого» демона, подписи — настоящие Yj
        self._bar_base = min(v for _, v in ranking)
        self._bar_items = [(n, v - self._bar_base) for n, v in ranking]
        self._bar_highlight = 0 if ok else None
        self._bar_width = 0
        self.after_idle(self._draw_bars)

    def _update_stale(self, res):
        stale = (res is not None and not self.auto_var.get()
                 and list(res["input"]) != self._vector())
        if stale:
            self.stale_badge.set("Признаки изменены — нажмите «Определить инструмент»",
                                 "warning")
            self.stale_badge.pack(anchor="w", pady=(6, 0), before=self.explain_btn)
        else:
            self.stale_badge.pack_forget()

    def _draw_bars(self, _event=None):
        if not self._bar_items:
            self.bars.delete("all")
            return
        width = int(self.bars.winfo_width())
        if width == self._bar_width and self.bars.find_all():
            return
        self._bar_width = width
        th.draw_hbars(self.bars, self._bar_items, self._bar_highlight, bar_h=18, gap=5,
                      label_w=140, value_fmt=lambda v: ec.fmt(v + self._bar_base))

    @staticmethod
    def _fit_table(tree, n, limit):
        """Высота по числу строк; полоса прокрутки — только если строки не помещаются."""
        tree.configure(height=max(3, min(n, limit)))
        for sb in tree.master.grid_slaves(column=1):
            if n > limit:
                sb.grid()
            else:
                sb.grid_remove()

    def _fill_votes(self, res):
        m = self.app.model
        self._fit_table(self.votes, m.n_classes, 12)
        if res is None:
            th.fill_table(self.votes, [(j + 1, c, "—", "—", "") for j, c in enumerate(m.classes)])
            return
        ranking = res["ranking"]
        top = ranking[0][1]
        rows, tags = [], []
        for k, (name, v) in enumerate(ranking):
            win = res["recognized"] and k == 0
            tie = not res["recognized"] and abs(v - top) <= ec.TIE_EPS
            rows.append((m.classes.index(name) + 1, name, ec.fmt(v),
                         "—" if k == 0 else "−" + ec.fmt(top - v),
                         "🏆 победитель" if win else "⚖ ничья за максимум" if tie else ""))
            tags.append("ok" if win else "warn" if tie else "")
        th.fill_table(self.votes, rows, tags)

    def _fill_check(self):
        m = self.app.model
        rows_in = m.check_all()
        self._fit_table(self.check, len(rows_in), 10)
        rows, tags = [], []
        for r in rows_in:
            rows.append((r["class"], ec.vec_str(r["vector"]), r["predicted"], _num(r["gap"]),
                         "✅ верно" if r["ok"] else "❌ ошибка"))
            tags.append("ok" if r["ok"] else "err")
        th.fill_table(self.check, rows, tags)
        bad = sum(1 for r in rows_in if not r["ok"])
        if not rows_in:
            self.check_banner.set("info", "База примеров пуста")
        elif not m.trained:
            self.check_banner.set("warning", "Модель не обучена — ответы пока не осмысленны")
        elif bad == 0:
            self.check_banner.set("success", "Верно %d из %d — система адекватна базе"
                                  % (len(rows_in), len(rows_in)))
        else:
            self.check_banner.set("danger", "Ошибок %d — вернитесь в режим обучения (дообучите)"
                                  % bad)
