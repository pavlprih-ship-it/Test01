# -*- coding: utf-8 -*-
"""Графики экспертной системы: чистый matplotlib (Figure, без pyplot и tkinter).

Каждая функция fig_* возвращает готовый Figure, поэтому работает и в окне
приложения (FigureCanvasTkAgg), и в автотестах (backend Agg).
"""
from __future__ import annotations

import itertools

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator

import expert_core as ec

# палитра совпадает с ui_theme.COLORS (здесь без импорта tkinter)
C = {"primary": "#2E86AB", "accent": "#F18F01", "success": "#2E9E5B",
     "danger": "#D64545", "muted": "#64748B", "text": "#1E2A35",
     "line": "#475569", "edge": "#CBD5E1", "unknown": "#9CA3AF"}
# красный (−) → белый (0) → синий (+): как тепловая сетка матрицы в приложении
DIVERGING = LinearSegmentedColormap.from_list("aij", [C["danger"], "#FFFFFF", C["primary"]])
SIZE = (8.6, 4.6)


# общие помощники

def _new(figsize, title) -> Figure:
    fig = Figure(figsize=figsize, facecolor="white", layout="constrained")
    fig.get_layout_engine().set(w_pad=0.15, h_pad=0.1)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=C["text"])
    return fig


def _style(ax, grid="both", integer=False):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["edge"])
    ax.tick_params(labelsize=9, colors=C["text"], length=3, color=C["edge"])
    ax.xaxis.label.set_size(9.5)
    ax.yaxis.label.set_size(9.5)
    if integer:                      # оси «эпоха» и «число ошибок»
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(True, axis=grid, alpha=0.3)
    ax.set_axisbelow(True)


def _empty(fig, text):
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color=C["muted"])
    return fig


def _feats(model):
    return ["x%d %s" % (i + 1, f) for i, f in enumerate(model.features)]


def _epochs_word(n):
    if n % 10 == 1 and n % 100 != 11:
        return "эпоху"
    return "эпохи" if n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14) else "эпох"


def _heat(ax, mat, cmap, vmin, vmax, fs=9, annotate=True):
    """Тепловая карта с белыми разделителями и подписями значений в ячейках."""
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    r, c = np.shape(mat)
    ax.set_xticks(np.arange(-0.5, c, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, r, 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if annotate:
        for i in range(r):
            for j in range(c):
                v = mat[i][j]
                if np.ma.is_masked(v) or not np.isfinite(v):
                    ax.text(j, i, "–", ha="center", va="center", fontsize=fs, color=C["muted"])
                    continue
                rgb = im.cmap(im.norm(v))[:3]
                dark = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] < 0.52
                ax.text(j, i, ec.fmt(v), ha="center", va="center", fontsize=fs,
                        color="white" if dark else C["text"])
    return im


def _ticks(ax, xlabels, ylabels, rot=30, fs=9):
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=rot, ha="right" if rot < 90 else "center", fontsize=fs)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=fs)


def _cbar(fig, im, ax, label=None, shrink=0.85):
    cb = fig.colorbar(im, ax=ax, pad=0.02, shrink=shrink)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=9)
    if label:
        cb.set_label(label, fontsize=9.5)


def _class_colors(n):
    """Качественная палитра без серых тонов (серый занят под «не опознан»)."""
    cols = colormaps["tab10" if n <= 9 else "tab20"].colors
    cols = [c for c in cols if max(c[:3]) - min(c[:3]) > 0.06]
    return [cols[i % len(cols)] for i in range(n)]


# --- 1. кривая обучения ---

def fig_learning_curve(model, report=None, figsize=SIZE) -> Figure:
    fig = _new(figsize, "Кривая обучения")
    hist = model.history
    if not hist:
        return _empty(fig, "История обучения пуста — запустите обучение.")
    ax = fig.add_subplot(111)
    _style(ax, integer=True)
    xs = [h.number for h in hist]
    after = [h.errors_after for h in hist]
    during = [h.errors_during for h in hist]
    n = max(len(model.examples), max(after + during))
    ax.plot(xs, during, color=C["muted"], ls="--", lw=1.6, marker="o", ms=4.5, mfc="white",
            label="ошибок реакции в ходе эпохи")
    ax.plot(xs, after, color=C["primary"], lw=2.4, marker="o", ms=6.5,
            label="ошибок экспертизы после эпохи")
    if len(xs) <= 30:
        for x, y in zip(xs, after):
            ax.annotate(str(y), (x, y), xytext=(0, 9), textcoords="offset points", ha="center",
                        fontsize=9, fontweight="bold", color=C["primary"])
    for zero, grp in itertools.groupby(zip(xs, after), key=lambda t: t[1] == 0):
        if zero:                     # серии эпох без ошибок подсвечиваем зелёным
            run = [x for x, _ in grp]
            ax.axvspan(run[0] - 0.5, run[-1] + 0.5, color=C["success"], alpha=0.1, lw=0)
    done = next((x for x, y in zip(xs, after) if y == 0), None)
    if done is not None:
        late = (done - xs[0]) / max(xs[-1] - xs[0], 1) > 0.6
        ax.text(done + (-0.6 if late else 0.6), 0.88, "обучение завершено (эпоха %d)" % done,
                transform=ax.get_xaxis_transform(), ha="right" if late else "left", va="top",
                fontsize=9, fontweight="bold", color=C["success"])
    if report is not None and report.kind in ("feedback", "continue") and report.epochs:
        x0 = report.epochs[0].number - 0.5
        ax.axvline(x0, color=C["accent"], ls=":", lw=2)
        ax.text(x0, 0.98, " дообучение", transform=ax.get_xaxis_transform(), ha="left",
                va="top", fontsize=9, fontweight="bold", color=C["accent"])
    ax.set(xlim=(xs[0] - 0.5, xs[-1] + 0.5), ylim=(-0.5, n + 1), xlabel="Номер эпохи (итерации)",
           ylabel="Количество ошибок")
    fig.legend(loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    return fig


# --- 2. весовая матрица ---

def fig_weight_heatmap(model, highlight_col=None, figsize=SIZE) -> Figure:
    fig = _new(figsize, "Весовая матрица экспертной системы (строки — признаки, столбцы — классы)")
    ax = fig.add_subplot(111)
    vmax = float(np.abs(model.matrix).max()) or 1.0
    im = _heat(ax, model.matrix, DIVERGING, -vmax, vmax, fs=10)
    _ticks(ax, model.classes, _feats(model), fs=9.5)
    if highlight_col is not None and 0 <= highlight_col < model.n_classes:
        ax.add_patch(Rectangle((highlight_col - 0.5, -0.5), 1, model.n_features, fill=False,
                               ec=C["accent"], lw=3, zorder=5, clip_on=False))
        ax.get_xticklabels()[highlight_col].set(color=C["accent"], fontweight="bold")
    _cbar(fig, im, ax, "Коэффициент aij", 0.9)
    return fig


# --- 3. эволюция матрицы ---

def fig_matrix_evolution(model, max_epochs=6, figsize=(10.5, 4.6)) -> Figure:
    fig = _new(figsize, "Как менялась матрица в процессе обучения")
    recs = [h for h in model.history if h.matrix_after is not None][:max_epochs]
    if not recs:
        return _empty(fig, "История обучения пуста — запустите обучение.")
    vmax = max(float(np.abs(r.matrix_after).max()) for r in recs) or 1.0
    axes = fig.subplots(1, len(recs), squeeze=False)[0]
    for k, (ax, rec) in enumerate(zip(axes, recs)):
        im = _heat(ax, rec.matrix_after, DIVERGING, -vmax, vmax, annotate=False)
        ax.set_title("после эпохи %d · ошибок %d" % (rec.number, rec.errors_after),
                     fontsize=9, color=C["text"])
        _ticks(ax, model.classes, _feats(model) if k == 0 else [""] * model.n_features, 90, 8)
    _cbar(fig, im, list(axes), "Коэффициент aij", 0.8)
    return fig


# --- 4. области классов в проекции на два признака ---

def fig_class_regions(model, fi, fj, point=None, point_name=None, base_vector=None,
                      figsize=SIZE) -> Figure:
    fig = _new(figsize, "Разделение классов в проекции на признаки x%d и x%d" % (fi + 1, fj + 1))
    ax = fig.add_subplot(111)
    _style(ax)
    m = model.n_classes
    colors = _class_colors(m)
    # сетка: вектор = базовый, два признака заменены координатами точки
    g = np.linspace(-0.5, 1.5, 240)
    gx, gy = np.meshgrid(g + 1.3e-6, g - 2.9e-6)   # сдвиг: узлы не попадают точно на границы
    base = np.zeros(model.n_features) if base_vector is None else np.array(base_vector, float)
    v = np.tile(base, (gx.size, 1))
    v[:, fi], v[:, fj] = gx.ravel(), gy.ravel()
    d = v @ model.matrix
    z = d.argmax(axis=1)
    if m > 1:
        top2 = np.partition(d, -2, axis=1)
        z = np.where(top2[:, -1] - top2[:, -2] <= ec.TIE_EPS, -1, z)
    z = z.reshape(gx.shape)
    ax.contourf(gx, gy, z, levels=np.arange(-1.5, m, 1.0), colors=[C["unknown"]] + colors,
                alpha=0.38, zorder=1)
    labels_present = np.unique(z)
    if len(labels_present) > 1:      # ранги меток: каждая граница рисуется один раз
        ax.contour(gx, gy, np.searchsorted(labels_present, z), colors=C["line"], linewidths=1.0,
                   levels=np.arange(0.5, len(labels_present) - 1, 1.0), zorder=2)
    visible = [int(k) for k in labels_present if k >= 0]
    # примеры базы: совпадающие точки раскладываем по окружности
    groups = {}
    for cls, vec in model.examples:
        groups.setdefault((round(vec[fi], 3), round(vec[fj], 3)), []).append(cls)
    star = None if point is None else (round(float(point[fi]), 3), round(float(point[fj]), 3))
    for (x, y), names in groups.items():
        n = len(names)
        with_star = star == (x, y)
        r = 0.0 if n == 1 and not with_star else min(0.05 + 0.02 * (n + with_star), 0.15)
        for k, cls in enumerate(names):
            th = 0.5 + 2 * np.pi * k / (n + with_star)
            ax.scatter(x + r * np.cos(th), y + r * np.sin(th), s=60, zorder=4,
                       color=colors[model.classes.index(cls)], edgecolor=C["text"], lw=0.9)
            c, s = np.cos(th) if r else 0.0, np.sin(th) if r else 1.0
            ax.annotate(cls, (x + r * np.cos(th), y + r * np.sin(th)), xytext=(11 * c, 11 * s),
                        textcoords="offset points", fontsize=8, zorder=6, color=C["text"],
                        ha="left" if c > 0.3 else "right" if c < -0.3 else "center",
                        va="bottom" if s > 0.3 else "top" if s < -0.3 else "center",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if point is not None:
        ax.scatter(point[fi], point[fj], marker="*", s=420, color=C["accent"], edgecolor=C["text"],
                   lw=1.2, zorder=7)
        if point_name:
            ax.annotate(point_name, (point[fi], point[fj]), xytext=(0, -17),
                        textcoords="offset points", ha="center", va="top", fontsize=9,
                        fontweight="bold", color=C["text"], zorder=8,
                        bbox=dict(boxstyle="round,pad=0.2", fc=C["accent"], ec="none", alpha=0.35))
    ax.set(xlim=(-0.5, 1.5), ylim=(-0.5, 1.5), xticks=[0, 1], yticks=[0, 1],
           xlabel="x%d — %s" % (fi + 1, model.features[fi]),
           ylabel="x%d — %s" % (fj + 1, model.features[fj]))
    handles = [Patch(fc=colors[k], ec=C["line"], alpha=0.6 if k in visible else 0.2) for k in range(m)]
    labels = [c if k in visible else "%s (нет области)" % c for k, c in enumerate(model.classes)]
    if (z == -1).any():
        handles.append(Patch(fc=C["unknown"], alpha=0.6, ec=C["line"]))
        labels.append("не опознан")
    handles.append(Line2D([], [], marker="o", ls="", mfc="white", mec=C["text"], ms=7))
    labels.append("примеры базы")
    if point is not None:
        handles.append(Line2D([], [], marker="*", ls="", mfc=C["accent"], mec=C["text"], ms=13))
        labels.append("объект экспертизы")
    leg = fig.legend(handles, labels, loc="outside right center", frameon=False, fontsize=9,
                     title="Классы", title_fontsize=9.5)
    for k, t in enumerate(leg.get_texts()[:m]):
        t.set_color(C["text"] if k in visible else C["muted"])
    fig._regions_info = {"classes_visible": [model.classes[k] for k in visible]}
    return fig


# --- 5. голоса демонов ---

def fig_demons(result, model, figsize=SIZE) -> Figure:
    ok = bool(result.get("recognized", result.get("winner") is not None))
    verdict = "побеждает «%s»" % result["winner_name"] if ok else "объект не опознан"
    fig = _new(figsize, "Голоса демонов для объекта «%s»: %s" % (result["name"], verdict))
    ax = fig.add_subplot(111)
    _style(ax, grid="x")
    names = [n for n, _ in result["ranking"]]
    vals = [float(v) for _, v in result["ranking"]]
    n, top = len(vals), max(vals)
    colors = [C["accent"] if (i == 0 and ok) else
              C["danger"] if (not ok and top - vals[i] <= ec.TIE_EPS) else C["primary"]
              for i in range(n)]
    ax.barh(range(n), vals, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color=C["line"], lw=1, zorder=4)
    for i, v in enumerate(vals):
        ax.annotate(ec.fmt(v), (v, i), xytext=(5 if v >= 0 else -5, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=9.5,
                    fontweight="bold" if i == 0 else "normal", color=C["text"])
    if n > 1:
        second, gap = vals[1], top - vals[1]
        ax.axvline(second, color=C["muted"], ls="--", lw=1.2, zorder=2)
        if gap > ec.TIE_EPS:
            ax.annotate("", (top, -0.85), (second, -0.85),
                        arrowprops=dict(arrowstyle="<->", color=C["text"], lw=1.2))
        ax.text((top + second) / 2, -1.0, "Δ = %s" % ec.fmt(gap) if gap > ec.TIE_EPS
                else "Δ = 0 — ничья", ha="center", va="bottom", fontsize=9.5,
                fontweight="bold", color=C["text"])
    lo, hi = min(0.0, min(vals)), max(0.0, top)
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - (0.12 if lo < 0 else 0.02) * span, hi + 0.12 * span)
    ax.set_ylim(n - 0.4, -1.7)
    ax.set_yticks(range(n), names, fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Голос демона Yj = Σ aij·xi    X = %s" % ec.vec_str(result["input"]))
    return fig


# --- 6. расстояния: мир и сознание ---

def _sym(dist, names, diag):
    idx = {c: i for i, c in enumerate(names)}
    mat = np.full((len(names), len(names)), np.nan)
    for (a, b), v in dist.items():
        if a in idx and b in idx:
            mat[idx[a], idx[b]] = mat[idx[b], idx[a]] = v
    for c in diag:
        mat[idx[c], idx[c]] = 0.0
    return np.ma.masked_invalid(mat)


def fig_distances(model, figsize=(10.5, 4.8)) -> Figure:
    fig = _new(figsize, "Различимость классов: мир и сознание системы")
    axes = fig.subplots(1, 2)
    specs = [(_sym(model.world_distances(), model.classes, model.prototypes()), "Blues",
              "Мир как есть —\nхэмминговы расстояния между эталонами"),
             (_sym(model.mind_distances(), model.classes, model.classes), "Oranges",
              "Мир в сознании системы —\nразличия столбцов матрицы")]
    for ax, (mat, cmap, title) in zip(axes, specs):
        cm = colormaps[cmap].copy()
        cm.set_bad("#F1F5F9")
        im = _heat(ax, mat, cm, 0, max(float(mat.max()), 1.0), fs=8.5)
        ax.set_title(title, fontsize=10, color=C["text"])
        _ticks(ax, model.classes, model.classes, 40, 8.5)
        _cbar(fig, im, ax, shrink=0.8)
    return fig


# --- 7. исследование шага обучения ---

def fig_step_study(rows, n_examples, normalize=True, figsize=SIZE) -> Figure:
    title = ("Исследование шага обучения h (нормализация в гиперкуб [-1; 1])" if normalize
             else "Исследование шага обучения h (без нормализации)")
    fig = _new(figsize, title)
    ax = fig.add_subplot(111)
    _style(ax, integer=True)
    cols = colormaps["viridis"](np.linspace(0.05, 0.85, max(len(rows), 1)))
    marks = "osD^v<>p"
    longest = 1
    for k, row in enumerate(rows):
        ys = list(row["curve_after"])
        longest = max(longest, len(ys))
        txt = ("сошлось за %d %s" % (row["epochs"], _epochs_word(row["epochs"]))
               if row["converged"] else "не сошлось")
        ax.plot(range(1, len(ys) + 1), ys, color=cols[k], lw=2.2 - 0.12 * k, alpha=0.9,
                marker=marks[k % len(marks)], ms=5.5, markevery=max(1, len(ys) // 10),
                label="h = %s\n%s" % (ec.fmt(row["step"]), txt))
    ax.set(xlim=(0.5, longest + 0.5), ylim=(-0.4, max(n_examples, 1) + 0.8),
           xlabel="Номер эпохи (итерации)", ylabel="Ошибок экспертизы")
    fig.legend(loc="outside right center", frameon=False, fontsize=9, title="Шаг обучения",
               title_fontsize=9.5)
    note = ("Вывод: малый шаг — консерватизм (медленно, но надёжно);\n"
            "большой — «разболтанность» (система не сходится)." if normalize else
            "Без нормализации шаг не влияет на решения:\nargmax не меняется от масштаба матрицы.")
    fig.supxlabel(note, fontsize=9.5, color=C["text"], linespacing=1.4,
                  bbox=dict(boxstyle="round,pad=0.5", fc="#E6F1F7", ec="none"))
    return fig


# --- 8. влияние признаков ---

def fig_feature_importance(model, figsize=SIZE) -> Figure:
    fig = _new(figsize, "Влияние признаков: размах коэффициента по классам")
    ax = fig.add_subplot(111)
    _style(ax, grid="x")
    items = sorted(zip(_feats(model), (r for _, r in model.feature_importance())),
                   key=lambda t: -t[1])
    n = len(items)
    top = max([r for _, r in items] + [1.0])
    for i, (name, r) in enumerate(items):
        zero = r <= ec.TIE_EPS
        ax.barh(i, r, height=0.62, color=C["edge"] if zero else C["primary"], zorder=3)
        ax.annotate("не влияет" if zero else ec.fmt(r), (r, i), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9.5,
                    fontstyle="italic" if zero else "normal",
                    color=C["muted"] if zero else C["text"],
                    fontweight="normal" if zero else "bold")
    ax.set_yticks(range(n), [name for name, _ in items], fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(n - 0.4, -0.6)
    ax.set_xlim(0, top * 1.15)
    ax.set_xlabel("Размах aij = max − min по классам")
    return fig
