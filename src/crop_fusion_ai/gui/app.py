"""Tkinter desktop demo for CropNet preprocessing."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import pandas as pd

from crop_fusion_ai.gui.controller import PreprocessingController, UploadMetadata


def _safe_int(value: str) -> int | None:
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def _safe_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _frame_preview(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "No features extracted."
    with pd.option_context("display.max_columns", 20, "display.width", 140):
        return frame.to_string(index=False)


class CropFusionApp(tk.Tk):
    """Main Tkinter window for source-specific preprocessing uploads."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Crop Fusion AI - Preprocessing")
        self.geometry("1120x760")
        self.minsize(980, 680)
        self.controller = PreprocessingController()

        self._ag_path = tk.StringVar()
        self._ndvi_path = tk.StringVar()
        self._weather_path = tk.StringVar()
        self._county_id = tk.StringVar()
        self._crop_type = tk.StringVar()
        self._year = tk.StringVar()
        self._month = tk.StringVar()

        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        heading = ttk.Label(
            container,
            text="CropNet preprocessing",
            font=("Segoe UI", 20, "bold"),
        )
        heading.pack(anchor="w")
        subtitle = ttk.Label(
            container,
            text="Upload AG, NDVI, or weather inputs and preview stable monthly features.",
        )
        subtitle.pack(anchor="w", pady=(0, 12))

        metadata = ttk.LabelFrame(container, text="Shared metadata", padding=12)
        metadata.pack(fill="x", pady=(0, 12))
        self._build_metadata_row(metadata)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)

        self._ag_output = self._build_ag_tab()
        self._ndvi_output = self._build_ndvi_tab()
        self._weather_output = self._build_weather_tab()

    def _build_metadata_row(self, parent: ttk.Frame) -> None:
        fields = [
            ("County ID", self._county_id),
            ("Crop Type", self._crop_type),
            ("Year", self._year),
            ("Month", self._month),
        ]
        for index, (label_text, variable) in enumerate(fields):
            ttk.Label(parent, text=label_text).grid(row=0, column=index * 2, sticky="w", padx=(0, 6))
            ttk.Entry(parent, textvariable=variable, width=16).grid(
                row=0,
                column=index * 2 + 1,
                sticky="ew",
                padx=(0, 16),
            )
            parent.grid_columnconfigure(index * 2 + 1, weight=1)

    def _build_file_picker(
        self,
        parent: ttk.Frame,
        label: str,
        path_var: tk.StringVar,
        filetypes: tuple[tuple[str, str], ...],
        browse_text: str,
    ) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=path_var).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(
            row,
            text=browse_text,
            command=lambda: self._browse_file(path_var, filetypes),
        ).pack(side="left")

    def _browse_file(self, path_var: tk.StringVar, filetypes: tuple[tuple[str, str], ...]) -> None:
        selected = filedialog.askopenfilename(filetypes=filetypes)
        if selected:
            path_var.set(selected)

    def _build_output_panel(self, parent: ttk.Frame) -> tk.Text:
        output = tk.Text(parent, height=18, wrap="none")
        scrollbar_y = ttk.Scrollbar(parent, orient="vertical", command=output.yview)
        scrollbar_x = ttk.Scrollbar(parent, orient="horizontal", command=output.xview)
        output.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        output.pack(side="top", fill="both", expand=True)
        scrollbar_y.pack(side="right", fill="y")
        scrollbar_x.pack(side="bottom", fill="x")
        return output

    def _metadata(self) -> UploadMetadata:
        return UploadMetadata(
            county_id=_safe_text(self._county_id.get()),
            crop_type=_safe_text(self._crop_type.get()),
            year=_safe_int(self._year.get()),
            month=_safe_int(self._month.get()),
        )

    def _render_output(self, output: tk.Text, frame: pd.DataFrame) -> None:
        output.delete("1.0", tk.END)
        output.insert("1.0", _frame_preview(frame))

    def _process_ag(self, output: tk.Text) -> None:
        file_path = self._ag_path.get().strip()
        if not file_path:
            messagebox.showerror("AG preprocessing", "Please select an agriculture image.")
            return
        try:
            result = self.controller.process_ag(file_path, self._metadata())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("AG preprocessing", str(exc))
            return
        self._render_output(output, result)

    def _process_ndvi(self, output: tk.Text) -> None:
        file_path = self._ndvi_path.get().strip()
        if not file_path:
            messagebox.showerror("NDVI preprocessing", "Please select an NDVI image.")
            return
        try:
            result = self.controller.process_ndvi(file_path, self._metadata())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("NDVI preprocessing", str(exc))
            return
        self._render_output(output, result)

    def _process_weather(self, output: tk.Text) -> None:
        file_path = self._weather_path.get().strip()
        if not file_path:
            messagebox.showerror("Weather preprocessing", "Please select a weather CSV file.")
            return
        try:
            result = self.controller.process_weather(file_path, self._metadata())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Weather preprocessing", str(exc))
            return
        self._render_output(output, result)

    def _build_ag_tab(self) -> tk.Text:
        frame = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(frame, text="AG")
        self._build_file_picker(
            frame,
            "AG image",
            self._ag_path,
            (("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")),
            "Browse",
        )
        process_button = ttk.Button(frame, text="Extract AG features")
        process_button.pack(anchor="w", pady=(0, 10))
        output = self._build_output_panel(frame)
        process_button.configure(command=lambda: self._process_ag(output))
        return output

    def _build_ndvi_tab(self) -> tk.Text:
        frame = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(frame, text="NDVI")
        self._build_file_picker(
            frame,
            "NDVI image",
            self._ndvi_path,
            (("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")),
            "Browse",
        )
        process_button = ttk.Button(frame, text="Extract NDVI features")
        process_button.pack(anchor="w", pady=(0, 10))
        output = self._build_output_panel(frame)
        process_button.configure(command=lambda: self._process_ndvi(output))
        return output

    def _build_weather_tab(self) -> tk.Text:
        frame = ttk.Frame(self.notebook, padding=14)
        self.notebook.add(frame, text="Weather")
        self._build_file_picker(
            frame,
            "Weather CSV",
            self._weather_path,
            (("CSV files", "*.csv"), ("All files", "*.*")),
            "Browse",
        )
        process_button = ttk.Button(frame, text="Extract weather features")
        process_button.pack(anchor="w", pady=(0, 10))
        output = self._build_output_panel(frame)
        process_button.configure(command=lambda: self._process_weather(output))
        return output


def main() -> None:
    """Launch the desktop GUI."""
    app = CropFusionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
