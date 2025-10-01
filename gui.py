# gui.py
import tkinter as tk
from tkinter import ttk, scrolledtext
from models import GPTModel

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("HIT137 Group Assignment - GPT GUI")
        self.geometry("700x500")

        self.model = GPTModel()

        ttk.Label(self, text="Enter text prompt:").pack(pady=5)
        self.entry = tk.Entry(self, width=80)
        self.entry.pack(pady=5)

        ttk.Button(self, text="Generate", command=self.run_model).pack(pady=5)

        self.output = scrolledtext.ScrolledText(self, height=20)
        self.output.pack(fill="both", expand=True, padx=10, pady=5)

    def run_model(self):
        prompt = self.entry.get()
        result = self.model.run(prompt)
        self.output.delete(1.0, "end")
        self.output.insert("end", result)
