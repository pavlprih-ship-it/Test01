# -*- coding: utf-8 -*-
"""Вкладка «Обучение»: параметры, ход обучения по эпохам, итоговая модель.

Слева — параметры (любое изменение сразу переобучает систему), справа —
наглядный отчёт: баннер результата, счётчики, таблица эпох, детали
выбранной эпохи, система уравнений и замечания к базе.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import expert_core as ec
import ui_theme as th
from ui_theme import COLORS

LEFT_W = 340
EPOCH_COLS = ["Эпоха", "Ошибок в ходе", "Не опознано после", "Коррекций", "Итог"]
PRES_COLS = ["№", "Пример", "Вектор X", "Реакция системы", "Статус", "Коррекция"]
CHECK_COLS = ["Класс", "Ответ системы", "Отрыв Δ", "Статус"]
WORD_EPOCH = ("эпоха", "эпохи", "эпох")
WORD_ERROR = ("ошибка", "ошибки", "ошибок")


# ---------------------------------------------------------------------------
# общие помощники (используются и во вкладке «База знаний»)
# ---------------------------------------------------------------------------

def plural(n: int, forms: tuple) -> str:
    """Склонение по числу: plural(2, ("эпоха", "эпохи", "эпох")) → «эпохи»."""
    n = abs(int(n))
    if 11 <= n % 100 <= 14:
        return forms[2]
    r = n % 10
    return forms[0] if r == 1 else forms[1] if 2 <= r <= 4 else forms[2]


def guard_wheel(widget):
    """Колесо мыши прокручивает сам виджет, пока ему есть что прокручивать,
    и только потом — страницу целиком."""
    def on_wheel(event):
        lo, hi = widget.yview()
        if lo <= 0.0 and hi >= 1.0:
            return None
        num = getattr(event, "num", None)
        step = -1 if num == 4 else 1 if num == 5 else (-1 if event.delta > 0 else 1)
        widget.yview_scroll(step, "units")
        return "break"
    for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
        widget.bind(seq, on_wheel)


def hint(parent, text: str) -> tk.Label:
    """Серая поясняющая строка, переносящаяся по ширине контейнера."""
    lbl = th.label(parent, text, "small", "muted", wraplength=600)
    lbl.bind("<Configure>", lambda e: lbl.configure(wraplength=max(200, e.width - 4)))
    return lbl


def make_tree(parent, columns, widths, anchors, max_rows: int = 8):
    """Таблица темы: все колонки растягиваются, колесо мыши не «залипает»."""
    frame, tree = th.make_table(parent, columns, widths, height=max_rows, anchors=anchors)
    for col in columns:
        tree.column(col, stretch=True)
    guard_wheel(tree)
    tree.max_rows = max_rows
    tree.vsb = next(w for w in frame.winfo_children() if isinstance(w, ttk.Scrollbar))
    return frame, tree


def set_rows(tree: ttk.Treeview, rows, tags=None):
    """Заполнить таблицу; высота подгоняется под число строк, полоса
    прокрутки видна, только если строк больше max_rows."""
    th.fill_table(tree, rows, tags)
    tree.configure(height=min(max(len(rows), 1), tree.max_rows))
    if len(rows) > tree.max_rows:
        tree.vsb.grid()
    else:
        tree.vsb.grid_remove()


def fit_text_height(text: tk.Text, lines: int):
    """Подогнать высоту Text под содержимое (шрифты тегов выше базового)."""
    text.configure(height=lines * 2)
    text.update_idletasks()               # без раскладки пиксели посчитать нельзя
    px = text.count("1.0", "end-1c", "ypixels")
    px = px[0] if isinstance(px, tuple) else px
    line = (th.FONTS["base"].metrics("linespace")
            + int(text.cget("spacing1")) + int(text.cget("spacing3")))
    text.configure(height=lines if not px else max(1, -(-px // line)))


def auto_hide_vsb(scroll: "th.ScrollFrame"):
    """Показывать вертикальную полосу ScrollFrame, только когда содержимое не помещается."""
    def sync():
        need = scroll.inner.winfo_reqheight() > scroll.canvas.winfo_height()
        shown = bool(scroll.vsb.winfo_manager())
        if need and not shown:
            scroll.vsb.pack(side="right", fill="y", before=scroll.canvas)
        elif not need and shown:
            scroll.vsb.pack_forget()

    scroll.inner.bind("<Configure>", lambda e: (scroll._on_inner(e), sync()))
    scroll.canvas.bind("<Configure>", lambda e: (
        scroll.canvas.itemconfigure(scroll._win, width=e.width), sync()))


class HScroll(tk.Frame):
    """Область с горизонтальной прокруткой для широких сеток; полоса
    появляется, только если содержимое не помещается. Содержимое — в `.inner`."""

    def __init__(self, parent):
        super().__init__(parent, background=COLORS["card"])
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0,
                                background=COLORS["card"], height=1)
        self.hsb = ttk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.hsb.set)
        self.canvas.grid(row=0, column=0, sticky="ew")
        self.hsb.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.inner = tk.Frame(self.canvas, background=COLORS["card"])
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._sync)
        self.canvas.bind("<Configure>", self._sync)

    def _sync(self, _event=None):
        w, h = self.inner.winfo_reqwidth(), self.inner.winfo_reqheight()
        self.canvas.configure(scrollregion=(0, 0, w, h), height=h)
        if w <= self.canvas.winfo_width():
            self.hsb.grid_remove()
            self.canvas.xview_moveto(0)
        else:
            self.hsb.grid()


# ---------------------------------------------------------------------------
# вкладка
# ---------------------------------------------------------------------------

class TrainingTab(ttk.Frame):
    """«🎓 Обучение»: параметры слева, результаты и ход обучения справа."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self._guard = False            # True, пока контролы синхронизируются с моделью
        self._shown = None             # эпоха, показанная в деталях
        self._sel_number = None        # номер выбранной эпохи
        self._rule_keys = {self._short(v): k for k, v in ec.UPDATE_RULES.items()}
        self._build_left()
        self._build_right()

    @staticmethod
    def _short(text: str) -> str:
        return text.split(" (")[0]

    # ------------------------------------------------------------------
    # левая колонка: параметры
    # ------------------------------------------------------------------

    def _build_left(self):
        outer = tk.Frame(self, width=LEFT_W, background=COLORS["bg"])
        outer.pack(side="left", fill="y", padx=(0, 10))
        outer.pack_propagate(False)
        sf = th.ScrollFrame(outer)
        sf.pack(fill="both", expand=True)
        auto_hide_vsb(sf)
        left = sf.inner
        wrap = LEFT_W - 80

        card = th.Card(left, "Параметры обучения")
        card.pack(fill="x", padx=(0, 2))
        body = card.body
        self.v_step = tk.StringVar()
        self.v_rule = tk.StringVar()
        self.v_norm = tk.BooleanVar()
        self.v_epochs = tk.StringVar()
        self.v_stop = tk.BooleanVar()

        def row():
            f = tk.Frame(body, background=COLORS["card"])
            f.pack(fill="x", pady=(0, 4))
            return f

        r = row()
        th.label(r, "Шаг обучения h").pack(side="left")
        self.sp_step = ttk.Spinbox(r, from_=0.05, to=5, increment=0.05, width=7,
                                   format="%.2f", textvariable=self.v_step,
                                   command=self._apply)
        self.sp_step.pack(side="right")
        th.label(body, "Правило коррекции").pack(fill="x", pady=(4, 0))
        self.cb_rule = ttk.Combobox(body, state="readonly", textvariable=self.v_rule,
                                    values=list(self._rule_keys))
        self.cb_rule.pack(fill="x", pady=(2, 2))
        self.rule_hint = th.label(body, "", "small", "muted", wraplength=wrap)
        self.rule_hint.pack(fill="x", pady=(0, 6))

        ttk.Checkbutton(body, text="Нормализация в гиперкуб [-1; 1]",
                        style="Card.TCheckbutton", variable=self.v_norm,
                        command=self._apply).pack(anchor="w")
        th.label(body, "после каждого акта обучения", "small", "muted"
                 ).pack(anchor="w", padx=(24, 0), pady=(0, 6))

        r = row()
        th.label(r, "Предел эпох").pack(side="left")
        self.sp_epochs = ttk.Spinbox(r, from_=1, to=200, increment=1, width=7,
                                     format="%.0f", textvariable=self.v_epochs,
                                     command=self._apply)
        self.sp_epochs.pack(side="right")
        ttk.Checkbutton(body, text="Авто-остановка при 0 ошибок",
                        style="Card.TCheckbutton", variable=self.v_stop,
                        command=self._apply).pack(anchor="w", pady=(2, 0))

        for w in (self.sp_step, self.sp_epochs):
            w.bind("<Return>", self._apply)
            w.bind("<FocusOut>", self._apply)
        self.cb_rule.bind("<<ComboboxSelected>>", self._apply)

        th.hsep(body, pady=10)
        ttk.Button(body, text="▶ Обучить с нуля", style="Accent.TButton",
                   command=self._on_retrain).pack(fill="x")
        ttk.Button(body, text="↻ Дообучить (+1 эпоха)",
                   command=self._on_more).pack(fill="x", pady=(6, 0))
        ttk.Button(body, text="Нормализовать матрицу сейчас",
                   command=self.app.normalize_now).pack(fill="x", pady=(6, 0))

        how = th.Card(left, "Как это работает")
        how.pack(fill="x", pady=(10, 0), padx=(0, 2))
        for line in (
            "«Мир» предъявляет пример, система реагирует: побеждает наибольший "
            "голос Yj = Σ aij·xi.",
            "«Мир» поощряет и наказывает: вектор примера (с шагом h) прибавляется к "
            "столбцу верного класса и вычитается из остальных.",
            "Эпоха — проход по всем примерам, затем экспертиза всех.",
            "Учимся до 0 ошибок или до предела эпох.",
        ):
            th.label(how.body, "•  " + line, "small", wraplength=wrap
                     ).pack(fill="x", pady=(0, 2))

    # ------------------------------------------------------------------
    # правая колонка: результаты
    # ------------------------------------------------------------------

    def _build_right(self):
        self.scroll = th.ScrollFrame(self)
        self.scroll.pack(side="left", fill="both", expand=True)
        inner = self.scroll.inner
        inner.columnconfigure(0, weight=1)

        self.banner = th.StatusBanner(inner)
        self.banner.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        self.badge_row = tk.Frame(inner, background=COLORS["bg"])
        self.badge_row.grid(row=1, column=0, sticky="w", pady=(0, 10))
        self.badges = {}
        for key in ("epochs", "errors", "fixes", "examples", "features", "classes"):
            b = th.Badge(self.badge_row)
            b.configure(font=th.FONTS["bold"], padx=10, pady=3)
            b.pack(side="left", padx=(0, 6))
            self.badges[key] = b

        # ход по эпохам
        self.epoch_card = th.Card(inner, "Ход обучения по эпохам")
        self.epoch_card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        hint(self.epoch_card.body,
             "Строка — один проход по всем примерам. «Ошибок в ходе» — неверные реакции "
             "во время прохода, «Не опознано после» — итог экспертизы после эпохи, "
             "«Коррекций» — сколько раз менялась матрица. Выберите строку, чтобы увидеть "
             "детали.").pack(fill="x", pady=(0, 6))
        frame, self.epoch_tree = make_tree(
            self.epoch_card.body, EPOCH_COLS, [56, 130, 150, 110, 100],
            ["center", "center", "center", "center", "w"], max_rows=8)
        frame.pack(fill="x")
        self.epoch_tree.bind("<<TreeviewSelect>>", self._on_select)
        self.epoch_empty = th.label(
            self.epoch_card.body, "Истории эпох нет — нажмите «▶ Обучить с нуля».",
            "small", "muted")

        # детали эпохи
        self.detail = th.Card(inner, "Эпоха подробно")
        self.detail.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        d = self.detail.body
        self.pres_title = th.label(d, "Предъявление примеров", "h3")
        self.pres_title.pack(fill="x")
        hint(d, "«Мир» показывает пример → система отвечает (реакция показана до "
                "коррекции) → при ошибке или при правиле «всегда» столбцы матрицы "
                "корректируются.").pack(fill="x", pady=(0, 4))
        frame, self.pres_tree = make_tree(
            d, PRES_COLS, [34, 125, 145, 140, 105, 95],
            ["center", "w", "center", "w", "w", "center"], max_rows=10)
        frame.pack(fill="x")
        self.check_title = th.label(d, "Экспертиза после эпохи", "h3")
        self.check_title.pack(fill="x", pady=(12, 0))
        hint(d, "Отрыв Δ — на сколько голос победителя больше голоса ближайшего "
                "соперника.").pack(fill="x", pady=(0, 4))
        frame, self.check_tree = make_tree(
            d, CHECK_COLS, [140, 160, 90, 130], ["w", "w", "center", "w"], max_rows=10)
        frame.pack(fill="x")
        self.matrix_title = th.label(d, "", "h3")
        self.matrix_title.pack(fill="x", pady=(12, 4))
        hs = HScroll(d)
        hs.pack(fill="x")
        self.grid_matrix = th.MatrixGrid(hs.inner, cell_width=10)
        self.grid_matrix.pack(anchor="w")

        # итоговая модель
        self.eq_card = th.Card(inner, "Итоговая модель — система уравнений")
        self.eq_card.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.eq_text = th.RichText(self.eq_card.body, height=10, padx=4, pady=4)
        self.eq_text.pack(fill="x")

        # замечания
        self.prob_card = th.Card(inner, "Замечания к базе")
        self.prob_card.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.prob_labels: list[tk.Label] = []
        self.prob_card.body.bind("<Configure>", lambda e: [
            lbl.configure(wraplength=max(200, e.width - 8)) for lbl in self.prob_labels])

    # ------------------------------------------------------------------
    # действия
    # ------------------------------------------------------------------

    def _apply(self, _event=None) -> list:
        """Применить параметры к модели; вернёт список изменений (при них
        система уже переобучилась)."""
        if self._guard:
            return []
        m = self.app.model
        try:
            step = min(5.0, max(0.05, float(self.v_step.get().replace(",", "."))))
            epochs = min(200, max(1, int(float(self.v_epochs.get().replace(",", ".")))))
        except ValueError:
            self._sync_controls()
            return []
        if abs(step - m.step) < 0.005:      # в окошке виден шаг с точностью 0.01
            step = m.step
        changed = self.app.set_params(
            step=step, normalize=self.v_norm.get(),
            update_rule=self._rule_keys.get(self.v_rule.get()),
            max_epochs=epochs, auto_stop=self.v_stop.get())
        if not changed:
            self._sync_controls()
        return changed

    def _on_retrain(self):
        if not self._apply():
            self.app.retrain("по кнопке")

    def _on_more(self):
        self._apply()
        self.app.train_more(1)

    def _on_select(self, _event=None):
        sel = self.epoch_tree.selection()
        hist = self.app.model.history
        if not sel:
            return
        idx = self.epoch_tree.index(sel[0])
        if idx < len(hist):
            self._sel_number = hist[idx].number
            if hist[idx] is not self._shown:
                self._show_epoch(hist[idx])

    # ------------------------------------------------------------------
    # обновление
    # ------------------------------------------------------------------

    def refresh(self):
        m, rep = self.app.model, self.app.last_report
        self._sync_controls()
        self._update_banner(m, rep)
        self._update_badges(m)
        self._update_epochs(m)
        self._update_equations(m)
        self._update_problems(m)

    def _sync_controls(self):
        m = self.app.model
        self._guard = True
        try:
            rule = ec.UPDATE_RULES.get(m.update_rule, m.update_rule)
            for var, value in ((self.v_step, "%.2f" % m.step),
                               (self.v_rule, self._short(rule)),
                               (self.v_epochs, str(m.max_epochs))):
                if var.get() != value:
                    var.set(value)
            self.v_norm.set(m.normalize)
            self.v_stop.set(m.auto_stop)
            self.rule_hint.configure(text=rule)
        finally:
            self._guard = False

    def _update_banner(self, m, rep):
        total = len(m.examples)
        if not m.trained:
            self.banner.set("warning", "⚠ Модель ещё не обучена",
                            "Матрица коэффициентов пока нулевая. Нажмите «▶ Обучить с нуля» — "
                            "система пройдёт по всем примерам и будет учиться, пока не "
                            "перестанет ошибаться.")
            return
        live = m.errors_now()
        if rep is None or not rep.epochs:
            kind = "success" if live == 0 else "warning"
            title = ("✅ Матрица готова — все %d примеров опознаются верно" % total
                     if live == 0 else
                     "⚠ Матрица загружена, но не опознано %d из %d примеров" % (live, total))
            self.banner.set(kind, title, "Обучение в этой сессии не запускалось. Нажмите "
                            "«▶ Обучить с нуля», чтобы увидеть ход обучения по эпохам.")
            return

        k = rep.n_epochs
        fixes = sum(1 for e in rep.epochs for p in e.presentations if p.updated)
        curve = rep.curve_after
        if len(curve) > 12:
            curve = curve[:5] + ["…"] + curve[-5:]
        pr = rep.params
        rule = self._short(ec.UPDATE_RULES.get(pr.get("update_rule"), "—"))
        parts = [("Обучение заняло %d %s" if rep.kind == "scratch" else "Дообучение: +%d %s")
                 % (k, plural(k, WORD_EPOCH)),
                 "динамика ошибок после эпох: " + " → ".join(map(str, curve)),
                 "коррекций матрицы: %d" % fixes,
                 "шаг h=%s" % ec.fmt(pr.get("step", m.step)),
                 "правило: %s" % rule]
        if pr.get("normalize"):
            parts.append("нормализация в гиперкуб")
        if rep.kind != "scratch":
            parts.insert(1, "всего эпох: %d" % m.epochs_done)
        if rep.added_example:
            parts.append("добавлен пример «%s» %s" % (rep.added_example[0],
                                                       ec.vec_str(rep.added_example[1])))
        text = " · ".join(parts)
        verb = "Обучение" if rep.kind == "scratch" else "Дообучение"

        if live != rep.final_errors:
            self.banner.set("warning", "⚠ Матрица изменена после обучения — не опознано "
                            "%d из %d примеров" % (live, total),
                            "Нажмите «▶ Обучить с нуля» или «↻ Дообучить».")
        elif rep.converged:
            self.banner.set("success", "✅ %s завершено — все %d примеров опознаются верно"
                            % (verb, total), text)
        else:
            todo = ("Увеличьте предел эпох или проверьте базу: возможно, есть неразличимые "
                    "классы (одинаковые признаки) — см. замечания ниже.")
            if pr.get("normalize"):
                todo += " Можно также выключить нормализацию или изменить шаг h."
            bad = any("неразличимы" in n for n in m.problems())
            self.banner.set("danger" if bad else "warning",
                            "⚠ %s не завершено — не опознано %d из %d примеров"
                            % (verb, live, total), text + "\n" + todo)

    def _update_badges(self, m):
        total = len(m.examples)
        err = m.errors_now() if m.trained else total
        fixes = sum(1 for e in m.history for p in e.presentations if p.updated)
        b = self.badges
        b["epochs"].set("эпох: %d" % m.epochs_done, "info")
        b["errors"].set("ошибок: %d/%d" % (err, total), "success" if err == 0 else "danger")
        b["fixes"].set("коррекций: %d" % fixes, "accent")
        b["examples"].set("примеров: %d" % total, "muted")
        b["features"].set("признаков: %d" % m.n_features, "muted")
        b["classes"].set("классов: %d" % m.n_classes, "muted")

    def _update_epochs(self, m):
        hist = m.history
        rows, tags = [], []
        for rec in hist:
            n = rec.errors_after
            verdict = ("✅ все верно" if n == 0
                       else "❌ %d %s" % (n, plural(n, WORD_ERROR)))
            fixes = sum(1 for p in rec.presentations if p.updated)
            rows.append((rec.number, "%d из %d" % (rec.errors_during, rec.total),
                         "%d из %d" % (n, len(rec.check)), fixes, verdict))
            tags.append("ok" if n == 0 else "err")
        set_rows(self.epoch_tree, rows, tags)
        if not hist:
            self.epoch_empty.pack(fill="x", pady=(6, 0))
            self.detail.grid_remove()
            self._shown = None
            return
        self.epoch_empty.pack_forget()
        self.detail.grid()
        idx = len(hist) - 1
        for i, rec in enumerate(hist):
            if rec.number == self._sel_number:
                idx = i
        item = self.epoch_tree.get_children()[idx]
        self.epoch_tree.selection_set(item)
        self.epoch_tree.see(item)
        self._sel_number = hist[idx].number
        self._show_epoch(hist[idx])

    def _show_epoch(self, rec):
        m = self.app.model
        self._shown = rec
        self.detail.title_label.configure(text="Эпоха %d подробно" % rec.number)

        rows, tags = [], []
        for p in rec.presentations:
            if p.predicted is None:
                status, tag = "⚪ не опознан", "warn"
            elif p.ok:
                status, tag = "✅ верно", "ok"
            else:
                status, tag = "❌ ошибка", "err"
            fix = ("да" + (" · норм." if p.clipped else "")) if p.updated else "нет"
            rows.append((p.index, p.class_name, ec.vec_str(p.vector),
                         p.predicted_name, status, fix))
            tags.append(tag)
        set_rows(self.pres_tree, rows, tags)

        rows, tags = [], []
        for r in rec.check:
            if r["ok"]:
                status, tag = "✅ верно", "ok"
            elif r["predicted"] == "не опознан":
                status, tag = "⚪ не опознан", "warn"
            else:
                status, tag = "❌ ошибка", "err"
            rows.append((r["class"], r["predicted"], ec.fmt(r["gap"]), status))
            tags.append(tag)
        set_rows(self.check_tree, rows, tags)
        self.check_title.configure(
            text="Экспертиза после эпохи %d — ошибок: %d из %d"
            % (rec.number, rec.errors_after, len(rec.check)))

        mat = rec.matrix_after
        feats = ["x%d %s" % (i + 1, m.features[i] if i < len(m.features) else "")
                 for i in range(mat.shape[0])]
        classes = [m.classes[j] if j < len(m.classes) else "y%d" % (j + 1)
                   for j in range(mat.shape[1])]
        self.matrix_title.configure(
            text="Матрица коэффициентов после эпохи %d (строки — признаки, "
                 "столбцы — классы)" % rec.number)
        self.grid_matrix.set(mat, feats, classes, ec.fmt)

    def _update_equations(self, m):
        t = self.eq_text
        t.clear()
        for line in m.equations():
            t.mono_block(line)
        t.add("Ответ экспертизы — класс с максимальным Yj.", "note")
        fit_text_height(t, m.n_classes + 1)

    def _update_problems(self, m):
        for lbl in self.prob_labels:
            lbl.destroy()
        self.prob_labels = []
        notes = m.problems()
        if not notes:
            self.prob_card.grid_remove()
            return
        self.prob_card.grid()
        for note in notes:
            lbl = th.label(self.prob_card.body, "⚠  " + note, wraplength=600)
            lbl.pack(fill="x", pady=2)
            self.prob_labels.append(lbl)
