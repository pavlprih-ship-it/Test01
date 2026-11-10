# -*- coding: utf-8 -*-
"""Вкладка «📊 Графики»: выбор графика слева, рисунок matplotlib справа.

Контракт вкладки (см. App): __init__(parent, app) и refresh(). Перерисовывается
только выбранный график, а пока вкладка скрыта, обновление откладывается до показа.
"""
from __future__ import annotations

import re
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

import charts  # noqa: E402
import expert_core as ec  # noqa: E402
import ui_theme as th  # noqa: E402
from ui_theme import COLORS, FONTS  # noqa: E402

# (ключ, название, описание)
CHARTS = [
    ("learning", "Кривая обучения",
     "Число ошибок по эпохам: как система учится «кнутом и пряником». Зелёным отмечены эпохи без ошибок."),
    ("weights", "Весовая матрица",
     "Коэффициенты aij: строки — признаки, столбцы — классы. Синий — поощрение, красный — наказание; "
     "рамка — победитель последней экспертизы."),
    ("evolution", "Эволюция матрицы",
     "Матрица после каждой из первых эпох на общей цветовой шкале: видно, где «кнут» и «пряник» меняют веса."),
    ("regions", "Разделение классов",
     "Каждая область — класс (пересечение полупространств Yj > Yk). Линии — следы разделяющих гиперплоскостей, "
     "точки — примеры базы, звезда — объект последней экспертизы."),
    ("demons", "Голоса демонов",
     "Выход Yj каждого демона для объекта последней экспертизы: «кто громче крикнет, тот и прав». "
     "Δ — отрыв победителя от второго."),
    ("distances", "Расстояния: мир и сознание",
     "Слева различия классов в базе примеров, справа — в матрице. Совпадающие столбцы (расстояние 0) "
     "система не различает."),
    ("importance", "Влияние признаков",
     "Размах коэффициентов признака по классам. Нулевой размах — признак ни на что не влияет (следствие 1 лекции)."),
    ("steps", "Исследование шага h",
     "Обучение с нуля при разных h (на копиях модели). Малый шаг — «консерватизм», большой — «разболтанность»: "
     "система не сходится."),
]
TITLES = {k: t for k, t, _ in CHARTS}
HINTS = {k: d for k, _, d in CHARTS}
DEFAULT_STEPS = "0.1, 0.25, 0.5, 1, 2"


class ChartsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self.fig = None
        self._canvas = None
        self._dirty = True
        self._features_seen = None
        self._steps_cache = None             # (сигнатура модели, строки, нормализация)
        self.var_chart = tk.StringVar(value="learning")
        self.var_x, self.var_y = tk.StringVar(), tk.StringVar()
        self.var_base = tk.BooleanVar(value=True)
        self.var_steps = tk.StringVar(value=DEFAULT_STEPS)
        self.var_norm = tk.BooleanVar(value=True)
        self.columnconfigure(0, minsize=250)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_left()
        self._build_right()
        self.bind("<Map>", lambda _e: self._dirty and self._render())
        self._show_params("learning")

    # ------------------------------------------------------------------ каркас

    def _build_left(self):
        left = ttk.Frame(self, width=250, height=10)
        left.pack_propagate(False)           # ширина колонки не «прыгает» между графиками
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        pick = th.Card(left, "Графики")
        pick.pack(fill="x")
        for key, title, _ in CHARTS:
            ttk.Radiobutton(pick.body, text=title, value=key, variable=self.var_chart,
                            style="Card.TRadiobutton", command=self._on_select
                            ).pack(fill="x", pady=1)
        opts = th.Card(left, "Параметры")
        opts.pack(fill="both", expand=True, pady=(8, 0))
        pbox = tk.Frame(opts.body, background=COLORS["card"])
        pbox.pack(fill="x")
        self.p_regions = tk.Frame(pbox, background=COLORS["card"])
        for text, var in (("Ось X", self.var_x), ("Ось Y", self.var_y)):
            row = tk.Frame(self.p_regions, background=COLORS["card"])
            row.pack(fill="x", pady=2)
            th.label(row, text, width=5).pack(side="left")
            cb = ttk.Combobox(row, textvariable=var, state="readonly", width=12)
            cb.pack(side="left", fill="x", expand=True)
            cb.bind("<<ComboboxSelected>>", lambda _e: self._render())
            setattr(self, "cb_" + text[-1].lower(), cb)
        self._check(self.p_regions, self.var_base, "Остальные признаки — как у объекта экспертизы",
                    self._render)
        self.p_steps = tk.Frame(pbox, background=COLORS["card"])
        th.label(self.p_steps, "Шаги h через запятую", "small", "muted").pack(fill="x")
        entry = ttk.Entry(self.p_steps, textvariable=self.var_steps)
        entry.pack(fill="x", pady=(2, 4))
        entry.bind("<Return>", lambda _e: self._run_steps())
        self._check(self.p_steps, self.var_norm, "нормализация в гиперкуб [-1; 1]", self._run_steps)
        ttk.Button(self.p_steps, text="Запустить исследование", style="Accent.TButton",
                   command=self._run_steps).pack(fill="x", pady=(4, 0))
        self.steps_err = th.label(self.p_steps, "", "small", "danger", wraplength=215)
        self.steps_err.pack(fill="x")
        self.p_none = th.label(pbox, "У этого графика нет настроек.", "small", "muted")
        th.hsep(opts.body, pady=8)
        self.hint = th.label(opts.body, "", "small", "muted", wraplength=215)
        self.hint.pack(fill="x")

    def _check(self, parent, var, text, command):
        """Флажок с переносимой подписью (ttk.Checkbutton не умеет переносить текст)."""
        row = tk.Frame(parent, background=COLORS["card"])
        row.pack(fill="x", pady=(6, 0))
        ttk.Checkbutton(row, variable=var, style="Card.TCheckbutton", command=command
                        ).pack(side="left", anchor="n")
        lbl = th.label(row, text, wraplength=185)
        lbl.pack(side="left", fill="x", expand=True)
        lbl.bind("<Button-1>", lambda _e: (var.set(not var.get()), command()))

    def _build_right(self):
        self.card = th.Card(self, "Визуализация модели")
        self.card.grid(row=0, column=1, sticky="nsew")
        self.btn_save = ttk.Button(self.card.head, text="Сохранить PNG…", command=self.save_png)
        self.btn_save.pack(side="right")
        self.host = tk.Frame(self.card.body, background=COLORS["card"])
        self.host.pack(fill="both", expand=True)
        self.extra = tk.Frame(self.card.body, background=COLORS["card"])
        self.table_frame, self.table = th.make_table(
            self.extra, ["Шаг h", "Эпох", "Сошлось", "Ошибок в конце"], widths=[110, 110, 130, 150],
            height=5)
        self.table_frame.pack(fill="x", pady=(8, 0))

    # ------------------------------------------------------------------ API вкладки

    def chart_keys(self) -> list[str]:
        return [k for k, _, _ in CHARTS]

    def select_chart(self, key: str):
        if key not in TITLES:
            raise KeyError(key)
        self.var_chart.set(key)
        self._on_select()

    def refresh(self):
        self._dirty = True
        if self.winfo_ismapped():
            self._render()

    def save_png(self):
        if self.fig is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self, title="Сохранить график", defaultextension=".png",
            initialfile="graph_%s.png" % self.var_chart.get(), filetypes=[("PNG", "*.png")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=150, facecolor="white")
        except OSError as exc:
            messagebox.showerror("Сохранение графика", str(exc), parent=self)
            return
        self.app.log("График «%s» сохранён: %s" % (TITLES[self.var_chart.get()], path), "ok")

    # ------------------------------------------------------------------ отрисовка

    def _on_select(self):
        self._show_params(self.var_chart.get())
        self._render()

    def _show_params(self, key):
        for w in (self.p_regions, self.p_steps, self.p_none):
            w.pack_forget()
        {"regions": self.p_regions, "steps": self.p_steps}.get(key, self.p_none).pack(fill="x")
        self.hint.configure(text=HINTS[key])

    def _sync_features(self):
        """Обновить списки осей при смене набора признаков (по умолчанию x2 и x3)."""
        m = self.app.model
        if self._features_seen == m.features:
            return
        self._features_seen = list(m.features)
        names = ["x%d %s" % (i + 1, f) for i, f in enumerate(m.features)]
        for cb, var, default in ((self.cb_x, self.var_x, 1), (self.cb_y, self.var_y, 2)):
            cb.configure(values=names)
            var.set(names[min(default, len(names) - 1)] if names else "")

    def _render(self):
        self._dirty = False
        self._sync_features()
        key = self.var_chart.get()
        try:
            built = self._build(key)
        except Exception as exc:             # noqa: BLE001 - ошибка графика не должна ронять окно
            self.app.log("График «%s»: %s" % (TITLES[key], exc), "err")
            built = ("Не удалось построить график", str(exc))
        for w in self.host.winfo_children():
            w.destroy()
        self._canvas = None
        self.fig = built if not isinstance(built, tuple) else None
        self.btn_save.state(["!disabled"] if self.fig else ["disabled"])
        if self.fig is None:
            self._placeholder(*built)
        else:
            self._canvas = FigureCanvasTkAgg(self.fig, master=self.host)
            widget = self._canvas.get_tk_widget()
            widget.configure(width=200, height=200, highlightthickness=0, background="white")
            widget.pack(fill="both", expand=True)
        if key == "steps" and self.fig is not None:
            self.extra.pack(side="bottom", fill="x", before=self.host)
        else:
            self.extra.pack_forget()

    def _placeholder(self, title, text):
        box = tk.Frame(self.host, background=COLORS["card"])
        box.place(relx=0.5, rely=0.45, anchor="center")
        th.label(box, title, "h2", anchor="center", justify="center").pack()
        th.label(box, text, "base", "muted", anchor="center", justify="center",
                 wraplength=460).pack(pady=(6, 0))

    def _build(self, key):
        """Figure или кортеж (заголовок, пояснение) для заглушки."""
        m, res = self.app.model, self.app.last_result
        if res is not None and (len(res["input"]) != m.n_features or len(res["demons"]) != m.n_classes):
            res = None
        if key == "steps":
            return self._steps_figure()
        if not m.trained:
            return ("Модель ещё не обучена", "Запустите обучение на вкладке «Обучение» — "
                                             "после этого здесь появятся графики.")
        if key == "learning":
            return charts.fig_learning_curve(m, self.app.last_report)
        if key == "weights":
            return charts.fig_weight_heatmap(m, res["winner"] if res else None)
        if key == "evolution":
            return charts.fig_matrix_evolution(m)
        if key == "regions":
            if m.n_features < 2:
                return ("Мало признаков", "Для проекции нужно минимум два признака.")
            fi, fj = self.cb_x.current(), self.cb_y.current()
            if fi < 0 or fj < 0 or fi == fj:
                return ("Выберите две разные оси", "Оси X и Y должны быть разными признаками.")
            base = res["input"] if res and self.var_base.get() else None
            return charts.fig_class_regions(m, fi, fj, point=res["input"] if res else None,
                                            point_name=res["name"] if res else None, base_vector=base)
        if key == "demons":
            if res is None:
                return ("Сначала выполните экспертизу",
                        "Голоса демонов строятся для объекта последней экспертизы — "
                        "откройте вкладку «Экспертиза» и опознайте объект.")
            return charts.fig_demons(res, m)
        if key == "distances":
            return charts.fig_distances(m)
        return charts.fig_feature_importance(m)

    # ------------------------------------------------------------------ исследование шага

    def _signature(self):
        m = self.app.model
        return (id(m), m.epochs_done, hash(m.matrix.tobytes()), hash(repr(m.examples)), m.update_rule)

    def _parse_steps(self) -> list[float]:
        try:
            steps = [float(t) for t in re.split(r"[;,\s]+", self.var_steps.get().strip()) if t]
        except ValueError:
            raise ValueError("шаги — числа через запятую, например 0.1, 0.5, 1") from None
        steps = list(dict.fromkeys(steps))
        if not 1 <= len(steps) <= 8 or any(not 0 < h <= 100 for h in steps):
            raise ValueError("нужно от 1 до 8 шагов, каждый больше 0 и не больше 100")
        return steps

    def _compute_steps(self):
        """Обучить копии модели при заданных h и запомнить результат (ValueError — неверный ввод)."""
        norm = bool(self.var_norm.get())
        try:
            steps = self._parse_steps()
            rows = ec.step_study(self.app.model, steps=steps, normalize=norm)
        except ValueError as exc:
            self.steps_err.configure(text="Ошибка: %s" % exc)
            raise
        self.steps_err.configure(text="")
        self._steps_cache = (self._signature(), rows, norm)
        return steps, norm

    def _run_steps(self):
        try:
            steps, norm = self._compute_steps()
        except ValueError:
            return
        self.app.log("Исследование шага h: %s, нормализация %s."
                     % (", ".join(ec.fmt(h) for h in steps), "вкл" if norm else "выкл"))
        if self.var_chart.get() == "steps":
            self._render()

    def _steps_figure(self):
        m = self.app.model
        if not m.examples:
            return ("В базе нет примеров", "Добавьте примеры на вкладке «База знаний».")
        if self._steps_cache is None or self._steps_cache[0] != self._signature():
            try:
                self._compute_steps()
            except ValueError as exc:
                return ("Неверные параметры исследования", str(exc))
        _, rows, norm = self._steps_cache
        th.fill_table(self.table, [(ec.fmt(r["step"]), r["epochs"], "да" if r["converged"] else "нет",
                                    r["final_errors"]) for r in rows],
                      ["ok" if r["converged"] else "err" for r in rows])
        self.table.configure(height=max(1, min(len(rows), 6)))
        return charts.fig_step_study(rows, len(m.examples), normalize=norm)
