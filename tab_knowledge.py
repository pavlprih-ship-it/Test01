# -*- coding: utf-8 -*-
"""Вкладка «База знаний»: признаки, классы и обучающие примеры.

Любая правка вызывает app.model_changed(...), поэтому система
переобучается автоматически. Здесь же — матрица признаков инструментов,
файловые операции и замечания к базе.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np

import expert_core as ec
from expert_core import ExpertModel
import ui_theme as th
from ui_theme import COLORS, FONTS
from tab_training import HScroll, guard_wheel, hint, make_tree, set_rows

JSON_TYPES = [("База знаний (JSON)", "*.json"), ("Все файлы", "*.*")]
LOAD_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError)


class ExampleDialog(tk.Toplevel):
    """Модальное окно добавления/правки примера: класс + признаки-галочки."""

    def __init__(self, tab: "KnowledgeTab", index: int | None = None):
        app, m = tab.app, tab.app.model
        super().__init__(app)
        self.tab, self.index = tab, index
        self.title("Новый пример" if index is None else "Изменить пример")
        self.configure(background=COLORS["bg"], padx=12, pady=12)
        self.transient(app)
        self.resizable(False, False)

        cls0, vec0 = m.classes[0], [0.0] * m.n_features
        if index is not None:
            cls0, vec0 = m.examples[index]
        else:
            sel = tab.lb_cls.curselection()
            if sel:
                cls0 = m.classes[sel[0]]
        card = th.Card(self, "Обучающий пример",
                       "Отметьте признаки, которые есть у инструмента.")
        card.pack(fill="both", expand=True)
        body = card.body
        th.label(body, "Класс (инструмент)", "bold").pack(fill="x")
        self.cls_var = tk.StringVar(value=cls0)
        self.cb = ttk.Combobox(body, state="readonly", textvariable=self.cls_var,
                               values=m.classes, width=34)
        self.cb.pack(fill="x", pady=(2, 10))
        th.label(body, "Признаки", "bold").pack(fill="x")
        grid = tk.Frame(body, background=COLORS["card"])
        grid.pack(fill="x", pady=(2, 8))
        rows = (m.n_features + 1) // 2 if m.n_features > 6 else m.n_features
        self.vars = []
        for i, name in enumerate(m.features):
            var = tk.BooleanVar(value=vec0[i] > 0.5)
            self.vars.append(var)
            ttk.Checkbutton(grid, text="x%d  %s" % (i + 1, name), variable=var,
                            style="Card.TCheckbutton", command=self._preview
                            ).grid(row=i % rows, column=i // rows, sticky="w",
                                   padx=(0, 18), pady=1)
        self.preview = th.label(body, "", "mono", "muted")
        self.preview.pack(fill="x")
        self._preview()
        th.hsep(body, pady=8)
        bar = tk.Frame(body, background=COLORS["card"])
        bar.pack(fill="x")
        ttk.Button(bar, text="Отмена", command=self.destroy).pack(side="right")
        ttk.Button(bar, text="Сохранить", style="Accent.TButton",
                   command=self._save).pack(side="right", padx=(0, 8))
        self.bind("<Escape>", lambda e: self.destroy())
        self.bind("<Return>", lambda e: self._save())

        self.update_idletasks()
        x = app.winfo_rootx() + (app.winfo_width() - self.winfo_reqwidth()) // 2
        y = app.winfo_rooty() + (app.winfo_height() - self.winfo_reqheight()) // 3
        self.geometry("+%d+%d" % (max(0, x), max(0, y)))
        try:
            self.wait_visibility()
            self.grab_set()
        except tk.TclError:
            pass
        self.cb.focus_set()

    def _vector(self) -> list:
        return [1.0 if v.get() else 0.0 for v in self.vars]

    def _preview(self):
        self.preview.configure(text="Вектор X = " + ec.vec_str(self._vector()))

    def _save(self):
        cls, vec = self.cls_var.get(), self._vector()
        self.destroy()
        self.tab.commit_example(self.index, cls, vec)


class KnowledgeTab(ttk.Frame):
    """«🗂 База знаний»: редактор признаков, классов и примеров."""

    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        self._sig = None
        self._ex_cols = None
        self.scroll = th.ScrollFrame(self)
        self.scroll.pack(fill="both", expand=True)
        root = self.scroll.inner
        root.columnconfigure(1, weight=1)
        left = tk.Frame(root, background=COLORS["bg"])
        left.grid(row=0, column=0, sticky="new", padx=(0, 10))
        right = tk.Frame(root, background=COLORS["bg"])
        right.grid(row=0, column=1, sticky="new", padx=(0, 4))

        self.card_feat, self.lb_feat, self.en_feat, self.btn_fdel = self._list_card(
            left, "Признаки (x1…xn)", self._add_feature, self._del_feature)
        self.card_cls, self.lb_cls, self.en_cls, self.btn_cdel = self._list_card(
            left, "Классы (инструменты, y1…ym)", self._add_class, self._del_class)
        self._build_file_card(left)
        self._build_examples(right)
        self._build_matrix(right)
        self._build_problems(right)

    # ------------------------------------------------------------------
    # построение
    # ------------------------------------------------------------------

    def _list_card(self, parent, title, on_add, on_del):
        card = th.Card(parent, title)
        card.pack(fill="x", pady=(0, 10))
        body = card.body
        box = tk.Frame(body, background=COLORS["card"])
        box.pack(fill="x")
        lb = tk.Listbox(box, height=6, width=32, font=FONTS["base"], relief="flat",
                        activestyle="none", exportselection=False,
                        background=COLORS["card"], foreground=COLORS["text"],
                        selectbackground=COLORS["primary_bg"],
                        selectforeground=COLORS["primary_dk"],
                        highlightthickness=1, highlightbackground=COLORS["border"],
                        highlightcolor=COLORS["primary"])
        sb = ttk.Scrollbar(box, orient="vertical", command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        lb.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        guard_wheel(lb)
        lb.bind("<<ListboxSelect>>", lambda e: self._update_buttons())
        row = tk.Frame(body, background=COLORS["card"])
        row.pack(fill="x", pady=(8, 0))
        entry = ttk.Entry(row)
        entry.pack(side="left", fill="x", expand=True)
        entry.bind("<Return>", lambda e: on_add())
        ttk.Button(row, text="Добавить", command=on_add).pack(side="left", padx=(6, 0))
        btn = ttk.Button(body, text="Удалить выбранный", command=on_del)
        btn.pack(fill="x", pady=(6, 0))
        return card, lb, entry, btn

    def _build_file_card(self, parent):
        card = th.Card(parent, "Файл")
        card.pack(fill="x")
        for text, cmd in (("💾 Сохранить базу (JSON)…", self._save_file),
                          ("📂 Загрузить базу (JSON)…", self._load_file),
                          ("↺ Сбросить к базе лабораторной (8 инструментов)", self._reset)):
            ttk.Button(card.body, text=text, command=cmd).pack(fill="x", pady=(0, 6))

    def _build_examples(self, parent):
        self.ex_card = th.Card(parent, "Обучающие примеры")
        self.ex_card.pack(fill="x", pady=(0, 10))
        body = self.ex_card.body
        bar = tk.Frame(body, background=COLORS["card"])
        bar.pack(fill="x", pady=(0, 8))
        ttk.Button(bar, text="➕ Добавить пример…", style="Accent.TButton",
                   command=self._add_example).pack(side="left")
        self.btn_edit = ttk.Button(bar, text="✎ Изменить…", command=self._edit_example)
        self.btn_edit.pack(side="left", padx=6)
        self.btn_del = ttk.Button(bar, text="🗑 Удалить", command=self._del_example)
        self.btn_del.pack(side="left")
        self.ex_host = tk.Frame(body, background=COLORS["card"])
        self.ex_host.pack(fill="x")
        self.legend = hint(body, "")
        self.legend.pack(fill="x", pady=(6, 0))

    def _build_matrix(self, parent):
        card = th.Card(parent, "📊 Матрица признаков инструментов")
        card.pack(fill="x", pady=(0, 10))
        hint(card.body, "Строки — признаки, столбцы — инструменты. 1 — признак есть, 0 — "
                        "нет; если у класса несколько примеров, показано среднее."
             ).pack(fill="x", pady=(0, 6))
        self.mx_empty = th.label(card.body, "Пока нет ни одного обучающего примера.",
                                 "small", "muted")
        self.mx_hs = HScroll(card.body)
        self.mx_hs.pack(fill="x")
        self.mx_grid = th.MatrixGrid(self.mx_hs.inner, cell_width=10)
        self.mx_grid.pack(anchor="w")

    def _build_problems(self, parent):
        self.prob_card = th.Card(parent, "Замечания к базе")
        self.prob_card.pack(fill="x", pady=(0, 10))
        self.prob_labels: list[tk.Label] = []
        self.prob_card.body.bind("<Configure>", lambda e: [
            w.configure(wraplength=max(200, e.width - 8)) for w in self.prob_labels])

    # ------------------------------------------------------------------
    # обновление
    # ------------------------------------------------------------------

    def refresh(self):
        m = self.app.model
        sig = (tuple(m.features), tuple(m.classes),
               tuple((c, tuple(v)) for c, v in m.examples))
        if sig == self._sig:
            return
        self._sig = sig
        self.card_feat.title_label.configure(text="Признаки (x1…x%d)" % m.n_features)
        self.card_cls.title_label.configure(text="Классы (инструменты, y1…y%d)" % m.n_classes)
        have = {c for c, _ in m.examples}
        self._fill_list(self.lb_feat, ["x%d  %s" % (i + 1, f) for i, f in enumerate(m.features)])
        self._fill_list(self.lb_cls, ["y%d  %s" % (j + 1, c) for j, c in enumerate(m.classes)],
                        dim=[c not in have for c in m.classes])
        self._fill_examples(m)
        self._fill_matrix(m)
        self._fill_problems(m)
        self._update_buttons()

    @staticmethod
    def _fill_list(lb: tk.Listbox, items, dim=None):
        keep = lb.curselection()
        lb.delete(0, "end")
        for i, text in enumerate(items):
            lb.insert("end", text)
            if dim and dim[i]:                  # класс без примеров
                lb.itemconfigure(i, foreground=COLORS["warning"])
        lb.configure(height=min(max(len(items), 5), 10))
        if keep and keep[0] < len(items):
            lb.selection_set(keep[0])
        elif len(items) > 10:
            lb.see("end")

    def _fill_examples(self, m):
        keep = self._selected_example()
        cols = ["Класс"] + ["x%d" % (i + 1) for i in range(m.n_features)]
        if cols != self._ex_cols:
            for w in self.ex_host.winfo_children():
                w.destroy()
            frame, self.ex_tree = make_tree(
                self.ex_host, cols, [150] + [44] * m.n_features,
                ["w"] + ["center"] * m.n_features, max_rows=12)
            frame.pack(fill="x")
            self.ex_tree.bind("<<TreeviewSelect>>", lambda e: self._update_buttons())
            self.ex_tree.bind("<Double-1>", self._on_double)
            self._ex_cols = cols
        rows = [(c,) + tuple("●" if v >= 1 else ec.fmt(v) if v else "·" for v in vec)
                for c, vec in m.examples]
        set_rows(self.ex_tree, rows)
        self.ex_card.title_label.configure(text="Обучающие примеры (%d)" % len(rows))
        if keep is not None and keep < len(rows):
            self._select_example(keep)
        self.legend.configure(text="● — признак есть, · — нет.   " + ";  ".join(
            "x%d — %s" % (i + 1, f) for i, f in enumerate(m.features)))

    def _fill_matrix(self, m):
        protos = m.prototypes()
        cols = [c for c in m.classes if c in protos]
        if not cols:
            self.mx_hs.pack_forget()
            self.mx_empty.pack(fill="x")
            return
        self.mx_empty.pack_forget()
        self.mx_hs.pack(fill="x")
        mat = np.array([[protos[c][i] for c in cols] for i in range(m.n_features)])
        self.mx_grid.set(mat, ["x%d %s" % (i + 1, f) for i, f in enumerate(m.features)],
                         cols, ec.fmt)

    def _fill_problems(self, m):
        for w in self.prob_labels:
            w.destroy()
        notes = m.problems()
        if notes:
            self.prob_labels = [th.label(self.prob_card.body, "⚠  " + n, wraplength=600)
                                for n in notes]
        else:
            self.prob_labels = [th.label(self.prob_card.body,
                                         "✅  Замечаний нет — база согласована.",
                                         color_key="success", wraplength=600)]
        for w in self.prob_labels:
            w.pack(fill="x", pady=2)

    def _update_buttons(self):
        def state(widget, on):
            widget.configure(state="normal" if on else "disabled")
        state(self.btn_fdel, bool(self.lb_feat.curselection()))
        state(self.btn_cdel, bool(self.lb_cls.curselection()))
        has = self._selected_example() is not None
        state(self.btn_edit, has)
        state(self.btn_del, has)

    # ------------------------------------------------------------------
    # изменение базы
    # ------------------------------------------------------------------

    def _warn(self, text: str):
        self.app.log("База знаний: " + text, "err")
        messagebox.showwarning("База знаний", text, parent=self.app)

    def _mutate(self, action) -> bool:
        """Выполнить правку модели (action вернёт причину для журнала) и
        запустить автопереобучение."""
        try:
            reason = action()
        except ValueError as exc:
            self._warn(str(exc))
            return False
        self.app.model_changed(reason)
        return True

    def _add_feature(self):
        name = self.en_feat.get().strip()

        def act():
            self.app.model.add_feature(name)
            return "новый признак «%s»" % name
        if self._mutate(act):
            self.en_feat.delete(0, "end")
            self.lb_feat.see("end")

    def _add_class(self):
        name = self.en_cls.get().strip()

        def act():
            self.app.model.add_class(name)
            return "новый класс «%s»" % name
        if self._mutate(act):
            self.en_cls.delete(0, "end")
            self.lb_cls.see("end")

    def _del_feature(self):
        sel = self.lb_feat.curselection()
        if not sel:
            return
        name = self.app.model.features[sel[0]]
        if not self.app.ask_yes_no("Удалить признак", "Удалить признак «%s»? Он исчезнет "
                                   "из всех примеров." % name):
            return
        self._mutate(lambda: "удалён признак «%s»" % self.app.model.remove_feature(sel[0]))

    def _del_class(self):
        sel = self.lb_cls.curselection()
        if not sel:
            return
        m = self.app.model
        name = m.classes[sel[0]]
        n = sum(1 for c, _ in m.examples if c == name)
        if not self.app.ask_yes_no("Удалить класс", "Удалить класс «%s»%s?" % (
                name, " вместе с его примерами (%d)" % n if n else "")):
            return
        self._mutate(lambda: "удалён класс «%s»" % m.remove_class(sel[0]))

    # --- примеры -------------------------------------------------------

    def _selected_example(self):
        sel = self.ex_tree.selection() if self._ex_cols else ()
        return self.ex_tree.index(sel[0]) if sel else None

    def _select_example(self, idx: int):
        kids = self.ex_tree.get_children()
        if 0 <= idx < len(kids):
            self.ex_tree.selection_set(kids[idx])
            self.ex_tree.see(kids[idx])

    def _open_dialog(self, index):
        m = self.app.model
        if not m.classes or not m.features:
            self._warn("Сначала добавьте хотя бы один признак и один класс.")
            return
        ExampleDialog(self, index)

    def _add_example(self):
        self._open_dialog(None)

    def _edit_example(self):
        idx = self._selected_example()
        if idx is not None:
            self._open_dialog(idx)

    def _on_double(self, event):
        if self.ex_tree.identify_region(event.x, event.y) == "cell":
            self._edit_example()

    def _del_example(self):
        idx = self._selected_example()
        if idx is not None:
            self._mutate(lambda: "удалён пример «%s»" % self.app.model.remove_example(idx)[0])

    def commit_example(self, index, cls, vec):
        def act():
            m = self.app.model
            if index is None:
                m.add_example(cls, vec)
                return "новый пример «%s»" % cls
            m.set_example(index, cls, vec)
            return "изменён пример «%s»" % cls
        if self._mutate(act):
            self._select_example(len(self.app.model.examples) - 1 if index is None else index)

    # --- файл ----------------------------------------------------------

    def _save_file(self):
        path = filedialog.asksaveasfilename(
            parent=self.app, title="Сохранить базу знаний", defaultextension=".json",
            initialfile="instrument_base.json", filetypes=JSON_TYPES)
        if not path:
            return
        try:
            self.app.model.save_json(path)
        except OSError as exc:
            self._warn("не удалось сохранить файл: %s" % exc)
            return
        self.app.log("База знаний сохранена: %s" % path, "ok")

    def _load_file(self):
        path = filedialog.askopenfilename(parent=self.app, title="Загрузить базу знаний",
                                          filetypes=JSON_TYPES)
        if not path:
            return
        try:
            model = ExpertModel.load_json(path)
        except LOAD_ERRORS as exc:
            self._warn("файл «%s» не похож на базу знаний (%s)" % (os.path.basename(path), exc))
            return
        self.app.replace_model(model, "загружен файл «%s»" % os.path.basename(path))

    def _reset(self):
        if not self.app.ask_yes_no(
                "Сброс базы", "Вернуть базу лабораторной (8 инструментов, 8 признаков)? "
                "Текущие признаки, классы и примеры будут потеряны."):
            return
        m = self.app.model
        self.app.replace_model(
            ExpertModel(step=m.step, normalize=m.normalize, update_rule=m.update_rule,
                        max_epochs=m.max_epochs, auto_stop=m.auto_stop),
            "сброс к базе лабораторной")
