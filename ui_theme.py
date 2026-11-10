# -*- coding: utf-8 -*-
"""Дизайн-система приложения: палитра, шрифты, стили ttk и готовые
виджеты (карточка, баннер статуса, бейдж, прокручиваемая область,
таблица, тепловая сетка матрицы, форматированный текст).

Все вкладки используют только эти помощники, чтобы интерфейс выглядел
единообразно. Тема Sun Valley (sv_ttk) подключается, если установлена;
иначе используется встроенная тема «clam» с теми же цветами.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, font as tkfont

try:
    import sv_ttk
except Exception:                      # noqa: BLE001 - тема необязательна
    sv_ttk = None

# ---------------------------------------------------------------------------
# палитра
# ---------------------------------------------------------------------------

COLORS = {
    "bg":        "#F3F5F9",   # фон окна
    "card":      "#FFFFFF",   # фон карточек
    "border":    "#D8DEE8",
    "text":      "#1E2A35",
    "muted":     "#64748B",
    "primary":   "#2E86AB",
    "primary_dk": "#1F6A88",
    "primary_bg": "#E6F1F7",
    "accent":    "#F18F01",
    "accent_bg": "#FFF1DD",
    "success":   "#2E9E5B",
    "success_bg": "#E4F5EA",
    "danger":    "#D64545",
    "danger_bg": "#FBE7E7",
    "warning":   "#C47F00",
    "warning_bg": "#FFF6DF",
    "info":      "#2E86AB",
    "info_bg":   "#E6F1F7",
    "row_alt":   "#F7F9FC",
    "heat_pos":  "#2E86AB",
    "heat_neg":  "#D64545",
}

KIND_COLORS = {
    "success": ("success", "success_bg"),
    "danger":  ("danger", "danger_bg"),
    "error":   ("danger", "danger_bg"),
    "warning": ("warning", "warning_bg"),
    "info":    ("info", "info_bg"),
    "accent":  ("accent", "accent_bg"),
    "muted":   ("muted", "row_alt"),
}

FONTS: dict[str, tkfont.Font] = {}


def _pick_family(candidates, fallback):
    available = set(tkfont.families())
    for c in candidates:
        if c in available:
            return c
    return fallback


def apply_theme(root: tk.Tk) -> dict:
    """Настроить тему, шрифты и именованные стили. Возвращает словарь
    шрифтов (также доступен как ui_theme.FONTS)."""
    ui_family = _pick_family(["Segoe UI", "SF Pro Text", "Helvetica Neue",
                              "DejaVu Sans", "Liberation Sans", "Arial"],
                             "TkDefaultFont")
    mono_family = _pick_family(["Consolas", "Cascadia Mono", "Menlo",
                                "DejaVu Sans Mono", "Liberation Mono",
                                "Courier New"], "TkFixedFont")
    FONTS.clear()
    FONTS.update({
        "base":  tkfont.Font(family=ui_family, size=10),
        "small": tkfont.Font(family=ui_family, size=9),
        "bold":  tkfont.Font(family=ui_family, size=10, weight="bold"),
        "h1":    tkfont.Font(family=ui_family, size=17, weight="bold"),
        "h2":    tkfont.Font(family=ui_family, size=12, weight="bold"),
        "h3":    tkfont.Font(family=ui_family, size=10, weight="bold"),
        "big":   tkfont.Font(family=ui_family, size=22, weight="bold"),
        "mono":  tkfont.Font(family=mono_family, size=10),
        "mono_small": tkfont.Font(family=mono_family, size=9),
    })
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
        try:
            tkfont.nametofont(name).configure(family=ui_family, size=10)
        except tk.TclError:
            pass
    try:
        tkfont.nametofont("TkFixedFont").configure(family=mono_family, size=10)
    except tk.TclError:
        pass

    style = ttk.Style(root)
    if sv_ttk is not None:
        sv_ttk.set_theme("light")
    else:
        style.theme_use("clam")
        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"],
                        font=FONTS["base"])
        style.configure("TButton", padding=(10, 5))
        style.configure("Accent.TButton", background=COLORS["primary"],
                        foreground="white", padding=(12, 6))
        style.map("Accent.TButton",
                  background=[("active", COLORS["primary_dk"]),
                              ("disabled", COLORS["border"])])
        style.configure("TNotebook.Tab", padding=(14, 7), font=FONTS["bold"])
        style.configure("Treeview", rowheight=24, fieldbackground=COLORS["card"],
                        background=COLORS["card"])
        style.configure("Treeview.Heading", font=FONTS["bold"],
                        background=COLORS["primary_bg"])

    root.configure(background=COLORS["bg"])
    style.configure("TFrame", background=COLORS["bg"])
    style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"])
    style.configure("Card.TFrame", background=COLORS["card"])
    style.configure("CardBody.TFrame", background=COLORS["card"])
    style.configure("Card.TLabel", background=COLORS["card"], foreground=COLORS["text"])
    style.configure("CardTitle.TLabel", background=COLORS["card"],
                    foreground=COLORS["text"], font=FONTS["h2"])
    style.configure("CardSub.TLabel", background=COLORS["card"],
                    foreground=COLORS["muted"], font=FONTS["small"])
    style.configure("Muted.TLabel", foreground=COLORS["muted"], font=FONTS["small"])
    style.configure("H1.TLabel", font=FONTS["h1"], foreground=COLORS["text"])
    style.configure("H2.TLabel", font=FONTS["h2"], foreground=COLORS["text"])
    style.configure("Header.TFrame", background=COLORS["primary"])
    style.configure("Header.TLabel", background=COLORS["primary"], foreground="white",
                    font=FONTS["h1"])
    style.configure("HeaderSub.TLabel", background=COLORS["primary"],
                    foreground="#D6EAF3", font=FONTS["small"])
    style.configure("Card.TCheckbutton", background=COLORS["card"])
    style.configure("Card.TRadiobutton", background=COLORS["card"])
    style.configure("Card.TLabelframe", background=COLORS["card"])
    style.configure("Card.TLabelframe.Label", background=COLORS["card"],
                    font=FONTS["h3"])
    style.configure("Treeview", rowheight=24)
    style.configure("Treeview.Heading", font=FONTS["bold"])
    return FONTS


# ---------------------------------------------------------------------------
# виджеты
# ---------------------------------------------------------------------------

class Card(ttk.Frame):
    """Белая карточка с рамкой, заголовком и подзаголовком.
    Содержимое добавляйте в `card.body`."""

    def __init__(self, parent, title: str | None = None,
                 subtitle: str | None = None, padding: int = 12, **kw):
        super().__init__(parent, style="Card.TFrame", **kw)
        self._border = tk.Frame(self, background=COLORS["border"])
        self._border.pack(fill="both", expand=True)
        inner = tk.Frame(self._border, background=COLORS["card"])
        inner.pack(fill="both", expand=True, padx=1, pady=1)
        self.inner = inner
        if title:
            head = tk.Frame(inner, background=COLORS["card"])
            head.pack(fill="x", padx=padding, pady=(padding - 2, 0))
            self.title_label = tk.Label(head, text=title, font=FONTS["h2"],
                                        background=COLORS["card"],
                                        foreground=COLORS["text"], anchor="w")
            self.title_label.pack(side="left", fill="x", expand=True)
            self.head = head
            if subtitle:
                tk.Label(inner, text=subtitle, font=FONTS["small"],
                         background=COLORS["card"], foreground=COLORS["muted"],
                         anchor="w", justify="left", wraplength=900
                         ).pack(fill="x", padx=padding)
        self.body = tk.Frame(inner, background=COLORS["card"])
        self.body.pack(fill="both", expand=True, padx=padding, pady=(6, padding))


class StatusBanner(tk.Frame):
    """Цветная плашка состояния: set(kind, title, text)."""

    def __init__(self, parent, **kw):
        super().__init__(parent, background=COLORS["info_bg"], **kw)
        self._title = tk.Label(self, font=FONTS["h3"], anchor="w",
                               background=COLORS["info_bg"], foreground=COLORS["info"])
        self._title.pack(fill="x", padx=12, pady=(8, 0))
        self._text = tk.Label(self, font=FONTS["base"], anchor="w", justify="left",
                              background=COLORS["info_bg"], foreground=COLORS["text"],
                              wraplength=820)
        self._text.pack(fill="x", padx=12, pady=(2, 8))
        self.bind("<Configure>", lambda e: self._text.configure(
            wraplength=max(200, e.width - 30)))

    def set(self, kind: str, title: str, text: str = ""):
        fg, bg = KIND_COLORS.get(kind, KIND_COLORS["info"])
        fg, bg = COLORS[fg], COLORS[bg]
        self.configure(background=bg)
        self._title.configure(text=title, background=bg, foreground=fg)
        self._text.configure(text=text, background=bg)
        if text:
            self._text.pack(fill="x", padx=12, pady=(2, 8))
        else:
            self._text.pack_forget()
            self._title.pack_configure(pady=8)


class Badge(tk.Label):
    """Маленькая цветная метка («✓ 0 ошибок», «4 эпохи»)."""

    def __init__(self, parent, text: str = "", kind: str = "info", **kw):
        super().__init__(parent, text=text, font=FONTS["small"], padx=8, pady=2, **kw)
        self.set(text, kind)

    def set(self, text: str, kind: str = "info"):
        fg, bg = KIND_COLORS.get(kind, KIND_COLORS["info"])
        self.configure(text=text, foreground=COLORS[fg], background=COLORS[bg])


class ScrollFrame(ttk.Frame):
    """Вертикально прокручиваемая область; содержимое кладите в `.inner`."""

    def __init__(self, parent, background: str | None = None, **kw):
        super().__init__(parent, **kw)
        bg = background or COLORS["bg"]
        self.canvas = tk.Canvas(self, highlightthickness=0, background=bg, borderwidth=0)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.vsb.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self.canvas, background=bg)
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind("<Configure>", self._on_inner)
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(
            self._win, width=e.width))
        for w in (self.canvas, self.inner):
            w.bind("<Enter>", self._bind_wheel)
            w.bind("<Leave>", self._unbind_wheel)

    def _on_inner(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _bind_wheel(self, _e=None):
        self.canvas.bind_all("<MouseWheel>", self._wheel)
        self.canvas.bind_all("<Button-4>", self._wheel)
        self.canvas.bind_all("<Button-5>", self._wheel)

    def _unbind_wheel(self, _e=None):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _wheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-1, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(1, "units")
        elif event.delta:
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def scroll_top(self):
        self.canvas.yview_moveto(0)


def make_table(parent, columns, widths=None, height: int = 8,
               anchors=None, stretch_last: bool = True):
    """ttk.Treeview с заголовками и вертикальной прокруткой.
    Возвращает (frame, tree). Теги строк: ok / err / warn / head / alt."""
    frame = tk.Frame(parent, background=COLORS["card"])
    tree = ttk.Treeview(frame, columns=list(columns), show="headings", height=height)
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    for i, col in enumerate(columns):
        tree.heading(col, text=col)
        w = (widths[i] if widths and i < len(widths) else 110)
        anchor = (anchors[i] if anchors and i < len(anchors) else "center")
        tree.column(col, width=w, minwidth=40, anchor=anchor,
                    stretch=(stretch_last and i == len(columns) - 1))
    tree.tag_configure("ok", background=COLORS["success_bg"])
    tree.tag_configure("err", background=COLORS["danger_bg"])
    tree.tag_configure("warn", background=COLORS["warning_bg"])
    tree.tag_configure("head", background=COLORS["primary_bg"], font=FONTS["bold"])
    tree.tag_configure("alt", background=COLORS["row_alt"])
    tree.tag_configure("bold", font=FONTS["bold"])
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return frame, tree


def fill_table(tree, rows, tags=None):
    """Заменить содержимое таблицы. rows — список кортежей,
    tags — список тегов (строка или кортеж) той же длины."""
    tree.delete(*tree.get_children())
    for i, row in enumerate(rows):
        tag = tags[i] if tags and i < len(tags) else ()
        if isinstance(tag, str):
            tag = (tag,)
        if not tag and i % 2:
            tag = ("alt",)
        tree.insert("", "end", values=tuple(row), tags=tag)


def heat_color(value: float, vmax: float) -> str:
    """Цвет ячейки для тепловой карты: синий (+) / красный (−) / белый (0)."""
    if vmax <= 0:
        return COLORS["card"]
    t = max(-1.0, min(1.0, float(value) / float(vmax)))
    base = COLORS["heat_pos"] if t >= 0 else COLORS["heat_neg"]
    t = abs(t) * 0.75
    r1, g1, b1 = int(base[1:3], 16), int(base[3:5], 16), int(base[5:7], 16)
    r = int(255 + (r1 - 255) * t)
    g = int(255 + (g1 - 255) * t)
    b = int(255 + (b1 - 255) * t)
    return "#%02x%02x%02x" % (r, g, b)


class MatrixGrid(tk.Frame):
    """Сетка чисел с подписями строк/столбцов и раскраской по знаку и
    величине (тепловая карта прямо в виджетах). set(matrix, rows, cols)."""

    def __init__(self, parent, cell_width: int = 7, **kw):
        super().__init__(parent, background=COLORS["card"], **kw)
        self.cell_width = cell_width
        self._cells: list[tk.Label] = []

    def set(self, matrix, row_labels, col_labels, fmt_fn=None,
            highlight_col: int | None = None, corner: str = ""):
        for c in self._cells:
            c.destroy()
        self._cells = []
        fmt_fn = fmt_fn or (lambda v: ("%g" % v))
        rows = len(row_labels)
        cols = len(col_labels)
        vmax = 0.0
        for r in range(rows):
            for c in range(cols):
                vmax = max(vmax, abs(float(matrix[r][c])))
        corner_lbl = tk.Label(self, text=corner, font=FONTS["small"], background=COLORS["card"],
                              foreground=COLORS["muted"], anchor="w", padx=6)
        corner_lbl.grid(row=0, column=0, sticky="nsew", padx=1, pady=1)
        self._cells.append(corner_lbl)
        for c, name in enumerate(col_labels):
            bg = COLORS["accent_bg"] if c == highlight_col else COLORS["primary_bg"]
            lbl = tk.Label(self, text=name, font=FONTS["bold"], background=bg,
                           foreground=COLORS["text"], width=self.cell_width + 3,
                           padx=4, pady=4)
            lbl.grid(row=0, column=c + 1, sticky="nsew", padx=1, pady=1)
            self._cells.append(lbl)
        for r, name in enumerate(row_labels):
            lbl = tk.Label(self, text=name, font=FONTS["bold"], background=COLORS["primary_bg"],
                           foreground=COLORS["text"], anchor="w", padx=6, pady=3)
            lbl.grid(row=r + 1, column=0, sticky="nsew", padx=1, pady=1)
            self._cells.append(lbl)
            for c in range(cols):
                v = float(matrix[r][c])
                cell = tk.Label(self, text=fmt_fn(v), font=FONTS["mono"],
                                background=heat_color(v, vmax),
                                foreground=COLORS["text"], width=self.cell_width, pady=3)
                if c == highlight_col:
                    cell.configure(relief="solid", borderwidth=1)
                cell.grid(row=r + 1, column=c + 1, sticky="nsew", padx=1, pady=1)
                self._cells.append(cell)


class RichText(tk.Text):
    """Текст с готовыми тегами: h1, h2, h3, body, mono, ok, err, warn,
    muted, bold, highlight. Методы: clear, add, heading, para, mono_block."""

    def __init__(self, parent, **kw):
        kw.setdefault("wrap", "word")
        kw.setdefault("relief", "flat")
        kw.setdefault("background", COLORS["card"])
        kw.setdefault("foreground", COLORS["text"])
        kw.setdefault("font", FONTS["base"])
        kw.setdefault("padx", 12)
        kw.setdefault("pady", 10)
        kw.setdefault("spacing1", 2)
        kw.setdefault("spacing3", 2)
        kw.setdefault("cursor", "arrow")
        super().__init__(parent, **kw)
        self.tag_configure("h1", font=FONTS["h1"], foreground=COLORS["primary"],
                           spacing1=10, spacing3=6)
        self.tag_configure("h2", font=FONTS["h2"], foreground=COLORS["primary_dk"],
                           spacing1=12, spacing3=4)
        self.tag_configure("h3", font=FONTS["h3"], foreground=COLORS["text"],
                           spacing1=8, spacing3=2)
        self.tag_configure("body", font=FONTS["base"])
        self.tag_configure("bold", font=FONTS["bold"])
        self.tag_configure("mono", font=FONTS["mono"], background=COLORS["row_alt"],
                           lmargin1=14, lmargin2=14, rmargin=10, spacing1=1, spacing3=1)
        self.tag_configure("ok", foreground=COLORS["success"], font=FONTS["bold"])
        self.tag_configure("err", foreground=COLORS["danger"], font=FONTS["bold"])
        self.tag_configure("warn", foreground=COLORS["warning"], font=FONTS["bold"])
        self.tag_configure("muted", foreground=COLORS["muted"], font=FONTS["small"])
        self.tag_configure("highlight", background=COLORS["accent_bg"])
        self.tag_configure("note", background=COLORS["info_bg"], lmargin1=10,
                           lmargin2=10, rmargin=10, spacing1=4, spacing3=4)
        self.configure(state="disabled")

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")

    def add(self, text: str, tag: str = "body", newline: bool = True):
        self.configure(state="normal")
        self.insert("end", text + ("\n" if newline else ""), tag)
        self.configure(state="disabled")

    def heading(self, text: str, level: int = 2):
        self.add(text, "h%d" % level)

    def para(self, text: str, tag: str = "body"):
        self.add(text, tag)

    def mono_block(self, text: str):
        self.add(text, "mono")

    def text(self) -> str:
        return self.get("1.0", "end").rstrip()


def hsep(parent, pady: int = 6, background: str | None = None):
    """Тонкая горизонтальная линия-разделитель."""
    f = tk.Frame(parent, height=1, background=COLORS["border"])
    f.pack(fill="x", pady=pady)
    return f


def label(parent, text, font_key: str = "base", color_key: str = "text",
          bg_key: str = "card", **kw):
    """tk.Label в цветах темы (для размещения внутри карточек)."""
    kw.setdefault("anchor", "w")
    kw.setdefault("justify", "left")
    return tk.Label(parent, text=text, font=FONTS[font_key],
                    foreground=COLORS[color_key], background=COLORS[bg_key], **kw)


def bar_canvas(parent, height: int = 26, **kw):
    """Холст для горизонтальных полосок (голоса демонов и т.п.)."""
    return tk.Canvas(parent, height=height, highlightthickness=0,
                     background=COLORS["card"], **kw)


def draw_hbars(canvas: tk.Canvas, items, highlight: int | None = None,
               bar_h: int = 20, gap: int = 6, label_w: int = 130,
               value_fmt=None):
    """Нарисовать горизонтальные полоски для items=[(name, value), …].
    Отрицательные значения рисуются влево от нулевой оси."""
    canvas.delete("all")
    if not items:
        return
    value_fmt = value_fmt or (lambda v: "%g" % v)
    width = max(int(canvas.winfo_width()), 300)
    n = len(items)
    canvas.configure(height=n * (bar_h + gap) + gap)
    vals = [float(v) for _, v in items]
    vmin, vmax = min(vals + [0.0]), max(vals + [0.0])
    span = max(vmax - vmin, 1e-9)
    x0 = label_w + 8
    x1 = width - 70
    zero_x = x0 + (0 - vmin) / span * (x1 - x0)
    canvas.create_line(zero_x, 2, zero_x, n * (bar_h + gap) + gap,
                       fill=COLORS["border"])
    for i, (name, v) in enumerate(items):
        y = gap + i * (bar_h + gap)
        xv = x0 + (v - vmin) / span * (x1 - x0)
        color = COLORS["accent"] if i == highlight else (
            COLORS["primary"] if v >= 0 else COLORS["danger"])
        canvas.create_rectangle(min(zero_x, xv), y, max(zero_x, xv), y + bar_h,
                                fill=color, outline="")
        canvas.create_text(x0 - 10, y + bar_h / 2, text=name, anchor="e",
                           font=FONTS["bold"] if i == highlight else FONTS["base"],
                           fill=COLORS["text"])
        canvas.create_text(x1 + 8, y + bar_h / 2, text=value_fmt(v), anchor="w",
                           font=FONTS["mono"], fill=COLORS["text"])
