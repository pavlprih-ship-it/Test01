# -*- coding: utf-8 -*-
"""Заглушка вкладки — будет заменена."""
import tkinter as tk
from tkinter import ttk
import ui_theme as th


class ExpertiseTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=10)
        self.app = app
        card = th.Card(self, "ExpertiseTab", "заглушка")
        card.pack(fill="both", expand=True)
        self.lbl = th.label(card.body, "...")
        self.lbl.pack(anchor="w")

    def refresh(self):
        m = self.app.model
        self.lbl.configure(text="trained=%s errors=%d" % (m.trained, m.errors_now() if m.trained else -1))
