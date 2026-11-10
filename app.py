# -*- coding: utf-8 -*-
"""Экспертная система «Подбор музыкального инструмента» — главное окно.

Имитационный принцип (лекция «Построение модели в виде оценочной
функции»): линейная модель Yj = Σ aij·xi, обучение «кнутом и пряником»,
экспертиза по принципу демона, дообучение по обратной связи «мира».

Запуск:        py app.py            (Windows)   |   python3 app.py
Скриншот:      py app.py --screenshot shot.png --tab training
Самопроверка:  py -m pytest tests
"""
from __future__ import annotations

import argparse
import datetime as _dt
import sys
import tkinter as tk
from tkinter import ttk, messagebox

import expert_core as ec
from expert_core import ExpertModel, TrainReport
import ui_theme as th
from ui_theme import COLORS, FONTS

APP_TITLE = "Экспертная система: подбор музыкального инструмента"
MIN_W, MIN_H = 1120, 700

TAB_KEYS = ["training", "expertise", "explain", "charts", "knowledge", "journal"]
TAB_TITLES = {
    "training":  "  🎓 Обучение  ",
    "expertise": "  🔍 Экспертиза  ",
    "explain":   "  📖 Объяснение  ",
    "charts":    "  📊 Графики  ",
    "knowledge": "  🗂 База знаний  ",
    "journal":   "  📝 Журнал  ",
}


class JournalTab(ttk.Frame):
    """Журнал работы: все события обучения и экспертизы по времени."""

    def __init__(self, parent, app: "App"):
        super().__init__(parent, padding=10)
        self.app = app
        card = th.Card(self, "Журнал работы",
                       "Хронология: обучение, дообучение, экспертизы, изменения базы знаний.")
        card.pack(fill="both", expand=True)
        bar = tk.Frame(card.body, background=COLORS["card"])
        bar.pack(fill="x", pady=(0, 6))
        ttk.Button(bar, text="Очистить", command=self.clear).pack(side="right")
        ttk.Button(bar, text="Скопировать", command=self.copy).pack(side="right", padx=6)
        self.text = th.RichText(card.body, font=FONTS["mono"])
        vsb = ttk.Scrollbar(card.body, orient="vertical", command=self.text.yview)
        self.text.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        self.text.pack(fill="both", expand=True)

    def append(self, msg: str, tag: str):
        stamp = _dt.datetime.now().strftime("%H:%M:%S")
        self.text.add("%s  " % stamp, "muted", newline=False)
        self.text.add(msg, {"info": "body", "ok": "ok", "err": "err", "warn": "warn",
                            "head": "h3", "mono": "mono"}.get(tag, "body"))
        self.text.see("end")

    def clear(self):
        self.text.clear()

    def copy(self):
        self.app.clipboard_clear()
        self.app.clipboard_append(self.text.text())

    def refresh(self):
        pass


class App(tk.Tk):
    """Главное окно. Вкладки общаются с моделью только через методы App:

    model          — ExpertModel (expert_core)
    last_report    — TrainReport последнего обучения/дообучения или None
    last_result    — результат model.explain() последней экспертизы или None
    retrain(reason)            — обучить с нуля, обновить все вкладки
    train_more(epochs)         — дообучить поверх текущей матрицы
    normalize_now()            — одноразово вернуть матрицу в гиперкуб [-1; 1]
    run_expertise(vec, name, show_explanation=False) — экспертиза объекта
    feedback(vec, true_class)  — «мир» наказал: запомнить пример и доучить
    model_changed(reason)      — база знаний изменена → автопереобучение
    set_params(**kw)           — шаг, нормализация, правило, предел эпох, авто-стоп
    replace_model(model, reason) — загрузить другую базу
    log(msg, tag)              — запись в журнал и строку состояния
    select_tab(key)            — переключить вкладку
    refresh_all()              — tab.refresh() у всех вкладок
    """

    def __init__(self, auto_train: bool = True):
        super().__init__()
        self.title(APP_TITLE)
        self.fonts = th.apply_theme(self)
        self.colors = COLORS
        self.model = ExpertModel()
        self.last_report: TrainReport | None = None
        self.last_result: dict | None = None
        self.tabs: dict[str, ttk.Frame] = {}
        self._pending_log: list[tuple[str, str]] = []
        self._fit_window()
        self.minsize(MIN_W, MIN_H)
        self._build()
        self.log("Экспертная система «Подбор музыкального инструмента»", "head")
        self.log("Модель: Yj = a1j·x1 + … + anj·xn — имитационный принцип "
                 "(линейная гипотеза, обучение «кнут и пряник»).")
        self.log("База знаний: %d классов, %d признаков, %d примеров."
                 % (self.model.n_classes, self.model.n_features, len(self.model.examples)))
        self.refresh_all()
        if auto_train:
            self.after(150, lambda: self.retrain("запуск программы"))

    # ------------------------------------------------------------------
    # окно и каркас
    # ------------------------------------------------------------------

    def _fit_window(self):
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = max(MIN_W, min(1500, sw - 60))
        h = max(MIN_H, min(940, sh - 100))
        self.geometry("%dx%d+%d+%d" % (w, h, max(0, (sw - w) // 2), max(0, (sh - h) // 2 - 10)))

    def _build(self):
        header = tk.Frame(self, background=COLORS["primary"], padx=18, pady=10)
        header.pack(fill="x")
        left = tk.Frame(header, background=COLORS["primary"])
        left.pack(side="left", fill="x", expand=True)
        tk.Label(left, text="🎵 " + APP_TITLE, font=FONTS["h1"], foreground="white",
                 background=COLORS["primary"], anchor="w").pack(anchor="w")
        tk.Label(left, text="Имитационный принцип · модель Yj = Σ aij·xi · обучение «кнут и пряник» · "
                            "экспертиза по принципу демона", font=FONTS["small"],
                 foreground="#D6EAF3", background=COLORS["primary"], anchor="w").pack(anchor="w")
        right = tk.Frame(header, background=COLORS["primary"])
        right.pack(side="right")
        self.state_badge = th.Badge(right, "модель не обучена", "warning")
        self.state_badge.configure(font=FONTS["bold"], padx=12, pady=5)
        self.state_badge.pack(anchor="e")
        self.state_sub = tk.Label(right, text="", font=FONTS["small"],
                                  background=COLORS["primary"], foreground="#D6EAF3")
        self.state_sub.pack(anchor="e")

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=(8, 4))

        from tab_training import TrainingTab
        from tab_expertise import ExpertiseTab
        from tab_explain import ExplainTab
        from tab_charts import ChartsTab
        from tab_knowledge import KnowledgeTab
        builders = {
            "training": TrainingTab, "expertise": ExpertiseTab, "explain": ExplainTab,
            "charts": ChartsTab, "knowledge": KnowledgeTab, "journal": JournalTab,
        }
        for key in TAB_KEYS:
            tab = builders[key](self.notebook, self)
            self.notebook.add(tab, text=TAB_TITLES[key])
            self.tabs[key] = tab

        status = tk.Frame(self, background=COLORS["bg"])
        status.pack(fill="x", padx=12, pady=(0, 6))
        self.status_var = tk.StringVar(value="Готово.")
        tk.Label(status, textvariable=self.status_var, font=FONTS["small"],
                 background=COLORS["bg"], foreground=COLORS["muted"], anchor="w"
                 ).pack(side="left", fill="x", expand=True)
        for key, msg in self._pending_log:
            self.tabs["journal"].append(msg, key)
        self._pending_log.clear()

    # ------------------------------------------------------------------
    # сервис для вкладок
    # ------------------------------------------------------------------

    def log(self, msg: str, tag: str = "info"):
        if "journal" in self.tabs:
            self.tabs["journal"].append(msg, tag)
        else:
            self._pending_log.append((tag, msg))
        if hasattr(self, "status_var"):
            self.status_var.set(msg.splitlines()[0][:160])

    def select_tab(self, key: str):
        if key in self.tabs:
            self.notebook.select(self.tabs[key])

    def current_tab(self) -> str:
        sel = self.notebook.select()
        for key, tab in self.tabs.items():
            if str(tab) == sel:
                return key
        return ""

    def refresh_all(self):
        self._update_badge()
        for tab in self.tabs.values():
            try:
                tab.refresh()
            except Exception as exc:        # noqa: BLE001 - одна вкладка не должна ронять всё
                self.log("Ошибка обновления вкладки %s: %s" % (type(tab).__name__, exc), "err")
                raise

    def _update_badge(self):
        m = self.model
        if not m.trained:
            self.state_badge.set("модель не обучена", "warning")
            self.state_sub.configure(text="")
            return
        errors = m.errors_now()
        total = len(m.examples)
        if errors == 0:
            self.state_badge.set("✓ обучена · ошибок 0/%d" % total, "success")
        else:
            self.state_badge.set("⚠ ошибок %d/%d" % (errors, total), "danger")
        self.state_sub.configure(text="эпох: %d · шаг h=%s · %s · %s"
                                 % (m.epochs_done, ec.fmt(m.step),
                                    "нормализация" if m.normalize else "без нормализации",
                                    "коррекция при ошибке" if m.update_rule == "on_error"
                                    else "коррекция всегда"))

    # ------------------------------------------------------------------
    # действия
    # ------------------------------------------------------------------

    def set_params(self, step=None, normalize=None, update_rule=None,
                   max_epochs=None, auto_stop=None, retrain: bool = True):
        m = self.model
        changed = []
        if step is not None and float(step) != m.step:
            m.step = float(step)
            changed.append("шаг h=%s" % ec.fmt(m.step))
        if normalize is not None and bool(normalize) != m.normalize:
            m.normalize = bool(normalize)
            changed.append("нормализация %s" % ("вкл" if m.normalize else "выкл"))
        if update_rule is not None and update_rule != m.update_rule:
            m.update_rule = update_rule
            changed.append("коррекция: %s" % ec.UPDATE_RULES.get(update_rule, update_rule))
        if max_epochs is not None and int(max_epochs) != m.max_epochs:
            m.max_epochs = int(max_epochs)
            changed.append("предел эпох %d" % m.max_epochs)
        if auto_stop is not None and bool(auto_stop) != m.auto_stop:
            m.auto_stop = bool(auto_stop)
            changed.append("авто-остановка %s" % ("вкл" if m.auto_stop else "выкл"))
        if changed and retrain:
            self.retrain("изменены параметры: " + ", ".join(changed))
        return changed

    def retrain(self, reason: str = "") -> TrainReport | None:
        m = self.model
        try:
            if not m.examples:
                raise ValueError("база примеров пуста")
            self.log("Режим обучения (%s): с нуля, шаг h=%s, нормализация %s, "
                     "коррекция %s, предел %d эпох."
                     % (reason or "по запросу", ec.fmt(m.step),
                        "вкл" if m.normalize else "выкл",
                        "при ошибке" if m.update_rule == "on_error" else "всегда",
                        m.max_epochs), "head")
            report = m.train()
        except ValueError as exc:
            self.log("Обучение невозможно: %s" % exc, "err")
            self.last_report = None
            self.refresh_all()
            return None
        self.last_report = report
        for e in report.epochs:
            self.log("Эпоха %d: ошибок реакции %d/%d, после эпохи не опознано %d/%d."
                     % (e.number, e.errors_during, e.total, e.errors_after, e.total),
                     "ok" if e.errors_after == 0 else "err")
        if report.converged:
            self.log("Обучение завершено: %s." % report.reason, "ok")
        else:
            self.log("Обучение не завершено: %s. Увеличьте предел эпох или проверьте базу "
                     "(неразличимые классы)." % report.reason, "warn")
        for note in m.problems():
            self.log("Замечание к базе: " + note, "warn")
        if self.last_result is not None:
            self.last_result = m.explain(self.last_result["input"], self.last_result["name"])
        self.refresh_all()
        return report

    def train_more(self, epochs: int = 1) -> TrainReport | None:
        m = self.model
        self.log("Дообучение: %d проход(а) по базе поверх текущей матрицы." % epochs, "head")
        try:
            report = m.train_continue(epochs)
        except ValueError as exc:
            self.log("Дообучение невозможно: %s" % exc, "err")
            return None
        self.last_report = report
        for e in report.epochs:
            self.log("Эпоха %d: ошибок реакции %d/%d, после эпохи не опознано %d/%d."
                     % (e.number, e.errors_during, e.total, e.errors_after, e.total),
                     "ok" if e.errors_after == 0 else "err")
        if report.final_errors == 0:
            self.log("Дообучение выполнено — все %d примеров опознаются верно."
                     % len(m.examples), "ok")
        else:
            bad = [r["class"] + " → " + r["predicted"] for r in report.epochs[-1].check if not r["ok"]]
            self.log("После дообучения не опознаются: %s." % ", ".join(bad), "warn")
        if self.last_result is not None:
            self.last_result = m.explain(self.last_result["input"], self.last_result["name"])
        self.refresh_all()
        return report

    def normalize_now(self):
        import numpy as np
        m = self.model
        before = m.matrix.copy()
        m.matrix = np.clip(m.matrix, -1.0, 1.0)
        changed = int(np.sum(before != m.matrix))
        self.log("Нормализация: %d коэффициентов возвращены на границу гиперкуба [-1; 1]; "
                 "ошибок теперь %d/%d." % (changed, m.errors_now(), len(m.examples)),
                 "info" if changed else "muted")
        if self.last_result is not None:
            self.last_result = m.explain(self.last_result["input"], self.last_result["name"])
        self.refresh_all()

    def run_expertise(self, vector, name: str = "объект",
                      show_explanation: bool = False) -> dict:
        m = self.model
        result = m.explain(vector, name or "объект")
        self.last_result = result
        if result["recognized"]:
            self.log("Экспертиза «%s» %s → %s (отрыв %s)."
                     % (result["name"], ec.vec_str(result["input"]),
                        result["winner_name"], ec.fmt(result["gap"])), "ok")
        else:
            self.log("Экспертиза «%s» %s → объект не опознан: равные максимальные голоса."
                     % (result["name"], ec.vec_str(result["input"])), "warn")
        for key in ("explain", "charts"):
            if key in self.tabs:
                self.tabs[key].refresh()
        self._update_badge()
        if show_explanation:
            self.select_tab("explain")
        return result

    def feedback(self, vector, true_class: str) -> TrainReport | None:
        m = self.model
        self.log("«Мир» сообщил: объект %s на самом деле — «%s». Дообучение…"
                 % (ec.vec_str(vector), true_class), "head")
        try:
            report = m.learn_feedback(vector, true_class)
        except ValueError as exc:
            self.log("Дообучение невозможно: %s" % exc, "err")
            return None
        self.last_report = report
        if report.added_example:
            self.log("Пример «%s» %s добавлен в базу знаний."
                     % (true_class, ec.vec_str(vector)))
        for e in report.epochs:
            self.log("Эпоха %d: ошибок реакции %d/%d, после эпохи не опознано %d/%d."
                     % (e.number, e.errors_during, e.total, e.errors_after, e.total),
                     "ok" if e.errors_after == 0 else "err")
        if report.converged:
            self.log("Дообучение завершено за %d эпох(и): %s." % (report.n_epochs, report.reason), "ok")
        else:
            self.log("Дообучение не сошлось: %s." % report.reason, "warn")
        self.last_result = m.explain(vector, self.last_result["name"] if self.last_result else true_class)
        self.refresh_all()
        return report

    def model_changed(self, reason: str = "база знаний изменена"):
        """После правки признаков/классов/примеров система переобучается сама."""
        self.log("Данные изменились (%s) — переобучение…" % reason, "head")
        if self.last_result is not None and len(self.last_result["input"]) != self.model.n_features:
            self.last_result = None
        return self.retrain(reason)

    def replace_model(self, model: ExpertModel, reason: str = "загружена база"):
        self.model = model
        self.last_result = None
        self.last_report = None
        self.log("База знаний заменена (%s): %d классов, %d признаков, %d примеров."
                 % (reason, model.n_classes, model.n_features, len(model.examples)), "head")
        if model.trained and model.errors_now() == 0:
            self.log("Загруженная матрица уже обучена — все примеры опознаются верно.", "ok")
            self.refresh_all()
        else:
            self.retrain(reason)

    def ask_yes_no(self, title: str, text: str) -> bool:
        return bool(messagebox.askyesno(title, text, parent=self))


# ----------------------------------------------------------------------
# запуск / скриншоты
# ----------------------------------------------------------------------

def _screenshot(app: App, path: str, tab: str, delay_ms: int = 1500):
    """Сохранить снимок окна (нужен виртуальный дисплей и ImageMagick `import`)."""
    import subprocess

    def shoot():
        app.update_idletasks()
        app.update()
        wid = app.winfo_id()
        try:
            subprocess.run(["import", "-window", "root", path], check=True, timeout=30)
            print("screenshot saved:", path, "window id", wid)
        except Exception as exc:          # noqa: BLE001
            print("screenshot failed:", exc)
        app.destroy()

    app.select_tab(tab)
    app.after(delay_ms, shoot)


def main(argv=None):
    parser = argparse.ArgumentParser(description=APP_TITLE)
    parser.add_argument("--screenshot", metavar="PNG", help="сохранить снимок окна и выйти")
    parser.add_argument("--tab", default="training", choices=TAB_KEYS,
                        help="вкладка, которую открыть при старте")
    parser.add_argument("--geometry", default=None, help="например 1400x900")
    parser.add_argument("--demo-expertise", action="store_true",
                        help="перед снимком выполнить экспертизу примера «Гитара»")
    args = parser.parse_args(argv)

    app = App()
    if args.geometry:
        app.geometry(args.geometry)
    if args.tab != "training":
        app.after(400, lambda: app.select_tab(args.tab))
    if args.demo_expertise:
        app.after(600, lambda: app.run_expertise([1, 0, 1, 0, 0, 0, 0, 1], "Гитара"))
    if args.screenshot:
        app.after(900, lambda: _screenshot(app, args.screenshot, args.tab))
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
