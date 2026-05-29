"""Tkinter desktop demo for directory-based CropNet forecasting."""

from __future__ import annotations

import queue
import threading
from math import ceil
from pathlib import Path

import pandas as pd

try:  # pragma: no cover - tkinter is platform/runtime dependent
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except ModuleNotFoundError as exc:  # pragma: no cover - handled in main()
    tk = None
    filedialog = None
    messagebox = None
    ttk = None
    FigureCanvasTkAgg = None
    Figure = None
    _TKINTER_IMPORT_ERROR = exc
else:
    _TKINTER_IMPORT_ERROR = None

from crop_fusion_ai.gui.forecasting import DirectoryForecastResult, build_forecast_from_directory
from cropnet_forecasting.features import AG_CORE, NDVI_CORE, WEATHER_CORE
from cropnet_forecasting.config import ForecastingConfig

ROOT = Path(__file__).resolve().parents[3]
BaseTk = tk.Tk if tk is not None else object


def _safe_text(value: str) -> str | None:
    stripped = value.strip()
    return stripped or None


def _frame_preview(frame: pd.DataFrame, *, max_rows: int = 12) -> str:
    if frame.empty:
        return "No forecast rows were produced."
    preview = frame.head(max_rows)
    with pd.option_context("display.max_columns", 40, "display.width", 180):
        return preview.to_string(index=False)


def _format_feature_group_frame(frame: pd.DataFrame, feature_names: list[str]) -> str:
    if frame.empty:
        return "No forecast rows were produced."
    columns = ["date", "county_id", "crop_type", "year", "month", "source_note"] + feature_names[:6]
    columns = [column for column in columns if column in frame.columns]
    with pd.option_context("display.max_columns", 20, "display.width", 180):
        return frame[columns].to_string(index=False)


class CropFusionApp(BaseTk):
    """Main Tkinter window for directory-driven forecasting."""

    def __init__(self) -> None:
        if tk is None:
            raise ModuleNotFoundError(
                "tkinter is required to run the GUI"
            ) from _TKINTER_IMPORT_ERROR
        super().__init__()
        self.title("Crop Fusion AI - Forecasting")
        self.geometry("1280x900")
        self.minsize(1120, 780)

        self._directory = tk.StringVar(value=str(ROOT / "data" / "raw"))
        self._config_path = tk.StringVar(value=str(ROOT / "configs" / "residual_lstm_all.yaml"))
        self._county_id = tk.StringVar(value="01003")
        self._crop_type = tk.StringVar(value="corn")
        self._status = tk.StringVar(value="Choose a directory, then click Analyse.")
        self._progress_text = tk.StringVar(value="Idle")
        self._result: DirectoryForecastResult | None = None
        self._analysis_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._analysis_thread: threading.Thread | None = None
        self._analyse_button: ttk.Button | None = None
        self._progress_bar: ttk.Progressbar | None = None

        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="CropNet forecasting", font=("Segoe UI", 22, "bold")).pack(anchor="w")
        ttk.Label(
            header,
            text="Select a folder once, scan ag/ ndvi/ weather/ files recursively, then forecast the next 12 months.",
        ).pack(anchor="w", pady=(4, 0))

        controls = ttk.LabelFrame(container, text="Inputs", padding=12)
        controls.pack(fill="x", pady=(0, 12))
        self._build_inputs(controls)

        progress_frame = ttk.Frame(container)
        progress_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(progress_frame, textvariable=self._status).pack(anchor="w")
        ttk.Label(progress_frame, textvariable=self._progress_text).pack(anchor="w", pady=(2, 4))
        self._progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100.0)
        self._progress_bar.pack(fill="x")

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill="both", expand=True)
        self._build_empty_state()

    def _build_inputs(self, parent: ttk.Frame) -> None:
        row0 = ttk.Frame(parent)
        row0.pack(fill="x", pady=(0, 8))
        ttk.Label(row0, text="Data directory", width=16).pack(side="left")
        ttk.Entry(row0, textvariable=self._directory).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row0, text="Choose directory", command=self._browse_directory).pack(side="left")

        row1 = ttk.Frame(parent)
        row1.pack(fill="x", pady=(0, 8))
        ttk.Label(row1, text="Config file", width=16).pack(side="left")
        ttk.Entry(row1, textvariable=self._config_path).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Button(row1, text="Choose config", command=self._browse_config).pack(side="left")

        row2 = ttk.Frame(parent)
        row2.pack(fill="x", pady=(0, 12))
        ttk.Label(row2, text="County ID", width=16).pack(side="left")
        ttk.Entry(row2, textvariable=self._county_id, width=16).pack(side="left", padx=(0, 16))
        ttk.Label(row2, text="Crop type").pack(side="left")
        ttk.Entry(row2, textvariable=self._crop_type, width=16).pack(side="left", padx=(0, 16))
        self._analyse_button = ttk.Button(row2, text="Analyse", command=self._analyse)
        self._analyse_button.pack(side="right")

    def _browse_directory(self) -> None:
        selected = filedialog.askdirectory(title="Select a folder containing ag/ ndvi/ weather/")
        if selected:
            self._directory.set(selected)

    def _browse_config(self) -> None:
        selected = filedialog.askopenfilename(
            title="Select a forecasting config",
            filetypes=(("YAML files", "*.yaml *.yml"), ("JSON files", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self._config_path.set(selected)

    def _build_empty_state(self) -> None:
        self._clear_tabs()
        frame = ttk.Frame(self.notebook, padding=16)
        self.notebook.add(frame, text="Overview")
        ttk.Label(
            frame,
            text="No analysis has been run yet.\nChoose a directory and click Analyse.",
            justify="center",
        ).pack(expand=True, fill="both")

    def _clear_tabs(self) -> None:
        for tab_id in self.notebook.tabs():
            self.notebook.forget(tab_id)

    def _analyse(self) -> None:
        root_dir = _safe_text(self._directory.get())
        config_path = _safe_text(self._config_path.get())
        county_id = _safe_text(self._county_id.get())
        crop_type = _safe_text(self._crop_type.get())
        if not root_dir:
            messagebox.showerror("Forecasting", "Please choose a data directory.")
            return
        if not config_path:
            messagebox.showerror("Forecasting", "Please choose a config file.")
            return
        if not county_id or not crop_type:
            messagebox.showerror("Forecasting", "County ID and crop type are required.")
            return

        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            return

        self._set_busy(True)
        self._status.set("Starting analysis...")
        self._progress_text.set("Preparing job")
        if self._progress_bar is not None:
            self._progress_bar.configure(value=0.0, maximum=100.0)

        def worker() -> None:
            try:
                cfg = ForecastingConfig.from_path(config_path)
                checkpoint_path = cfg.checkpoint_path or "weights/lstm_best.pt"
                scaler_path = cfg.scaler_path or "weights/scaler.csv"

                def progress(stage: str, current: int, total: int, message: str) -> None:
                    self._analysis_queue.put(("progress", (stage, current, total, message)))

                result = build_forecast_from_directory(
                    root_dir,
                    county_id=county_id,
                    crop_type=crop_type,
                    checkpoint_path=checkpoint_path,
                    scaler_path=scaler_path,
                    config_path=config_path,
                    horizon=12,
                    device=cfg.device,
                    progress=progress,
                )
            except Exception as exc:  # noqa: BLE001
                self._analysis_queue.put(("error", exc))
            else:
                self._analysis_queue.put(("result", result))

        self._analysis_thread = threading.Thread(target=worker, daemon=True)
        self._analysis_thread.start()
        self.after(100, self._poll_analysis_queue)

    def _set_busy(self, busy: bool) -> None:
        if self._analyse_button is not None:
            self._analyse_button.configure(state="disabled" if busy else "normal")

    def _update_progress(self, stage: str, current: int, total: int, message: str) -> None:
        self._progress_text.set(message)
        if self._progress_bar is None:
            return
        if total <= 0:
            self._progress_bar.configure(mode="indeterminate")
            self._progress_bar.start(10)
            return
        self._progress_bar.stop()
        self._progress_bar.configure(mode="determinate", maximum=float(total))
        self._progress_bar["value"] = float(current)
        if stage == "feature_align" and total > 0:
            self._status.set(f"Model features aligned: {current}/{total}")

    def _poll_analysis_queue(self) -> None:
        try:
            while True:
                kind, payload = self._analysis_queue.get_nowait()
                if kind == "progress":
                    stage, current, total, message = payload  # type: ignore[misc]
                    self._update_progress(stage, current, total, message)
                elif kind == "error":
                    self._set_busy(False)
                    if self._progress_bar is not None:
                        self._progress_bar.stop()
                    messagebox.showerror("Forecasting", str(payload))
                    self._status.set("Analysis failed.")
                    self._progress_text.set("Idle")
                    return
                elif kind == "result":
                    result = payload  # type: ignore[assignment]
                    self._result = result
                    self._status.set(
                        f"Loaded {len(result.source_files)} files | model={result.predictor.model_name} | features={len(result.predictor.feature_names)}"
                    )
                    self._progress_text.set(
                        f"Completed forecast with {len(result.forecast)} rows and {len(result.predictor.feature_names)} model features"
                    )
                    if self._progress_bar is not None:
                        self._progress_bar.stop()
                        self._progress_bar["value"] = self._progress_bar["maximum"]
                    self._render_result(result)
                    self._set_busy(False)
                    return
        except queue.Empty:
            pass

        if self._analysis_thread is not None and self._analysis_thread.is_alive():
            self.after(100, self._poll_analysis_queue)
        else:
            self._set_busy(False)

    def _render_result(self, result: DirectoryForecastResult) -> None:
        self._clear_tabs()
        self._build_overview_tab(result)
        self._build_feature_tab("AG", result.forecast, AG_CORE)
        self._build_feature_tab("NDVI", result.forecast, NDVI_CORE)
        self._build_feature_tab("Weather", result.forecast, WEATHER_CORE)

    def _build_overview_tab(self, result: DirectoryForecastResult) -> None:
        frame = ttk.Frame(self.notebook, padding=12)
        self.notebook.add(frame, text="Overview")

        info = ttk.LabelFrame(frame, text="Run summary", padding=10)
        info.pack(fill="x", pady=(0, 10))
        ttk.Label(
            info,
            text=(
                f"Files scanned: {len(result.source_files)}\n"
                f"Monthly rows: {len(result.monthly_features)}\n"
                f"Forecast rows: {len(result.forecast)}\n"
                f"Model: {result.predictor.model_name}\n"
                f"Feature count: {len(result.predictor.feature_names)}\n"
                f"Model features aligned: {len(result.predictor.feature_names)}/{len(result.predictor.feature_names)}"
            ),
            justify="left",
        ).pack(anchor="w")

        files = ttk.LabelFrame(frame, text="Discovered files", padding=10)
        files.pack(fill="both", expand=False, pady=(0, 10))
        files_text = tk.Text(files, height=8, wrap="none")
        files_scroll_y = ttk.Scrollbar(files, orient="vertical", command=files_text.yview)
        files_scroll_x = ttk.Scrollbar(files, orient="horizontal", command=files_text.xview)
        files_text.configure(yscrollcommand=files_scroll_y.set, xscrollcommand=files_scroll_x.set)
        files_text.pack(side="top", fill="both", expand=True)
        files_scroll_y.pack(side="right", fill="y")
        files_scroll_x.pack(side="bottom", fill="x")
        files_text.insert(
            "1.0",
            "\n".join(f"{sample.modality}: {sample.path}" for sample in result.source_files),
        )
        files_text.configure(state="disabled")

        preview = ttk.LabelFrame(frame, text="Forecast preview", padding=10)
        preview.pack(fill="both", expand=True)
        preview_text = tk.Text(preview, wrap="none")
        preview_scroll_y = ttk.Scrollbar(preview, orient="vertical", command=preview_text.yview)
        preview_scroll_x = ttk.Scrollbar(preview, orient="horizontal", command=preview_text.xview)
        preview_text.configure(yscrollcommand=preview_scroll_y.set, xscrollcommand=preview_scroll_x.set)
        preview_text.pack(side="top", fill="both", expand=True)
        preview_scroll_y.pack(side="right", fill="y")
        preview_scroll_x.pack(side="bottom", fill="x")
        preview_text.insert("1.0", _frame_preview(result.forecast))
        preview_text.configure(state="disabled")

    def _build_feature_tab(self, title: str, forecast: pd.DataFrame, feature_names: list[str]) -> None:
        frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(frame, text=title)

        if forecast.empty:
            ttk.Label(frame, text="No forecast data available for this modality.").pack(expand=True, fill="both")
            return

        forecast = forecast.copy()
        forecast["date"] = pd.to_datetime(forecast["date"])

        summary = ttk.LabelFrame(frame, text="Feature summary", padding=8)
        summary.pack(fill="x", pady=(0, 10))
        ttk.Label(
            summary,
            text=(
                f"{len(feature_names)} features | "
                f"12-month forecast | "
                f"scroll down to inspect each series"
            ),
        ).pack(anchor="w")

        scroll_host = ttk.Frame(frame)
        scroll_host.pack(fill="both", expand=True)
        canvas = tk.Canvas(scroll_host, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_host, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        plot_container = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=plot_container, anchor="nw")

        def _sync_scrollregion(event: object) -> None:  # noqa: ANN401
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event: object) -> None:  # noqa: ANN401
            canvas.itemconfigure(window_id, width=getattr(event, "width", canvas.winfo_width()))

        plot_container.bind("<Configure>", _sync_scrollregion)
        canvas.bind("<Configure>", _sync_width)

        for feature in feature_names:
            plot_frame = ttk.LabelFrame(plot_container, text=feature, padding=8)
            plot_frame.pack(fill="x", expand=True, pady=(0, 10))

            fig = Figure(figsize=(11.5, 2.6), dpi=100, layout="constrained")
            ax = fig.add_subplot(111)
            series = pd.to_numeric(forecast[feature], errors="coerce")
            ax.plot(
                forecast["date"],
                series,
                marker="o",
                linewidth=2.0,
                color="#2d6cdf",
                markersize=4,
            )
            ax.set_title(feature, fontsize=11, loc="left")
            ax.grid(alpha=0.25)
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xlabel("Month", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            ax.set_facecolor("#fbfbfd")

            canvas_widget = FigureCanvasTkAgg(fig, master=plot_frame)
            canvas_widget.draw()
            widget = canvas_widget.get_tk_widget()
            widget.pack(fill="x", expand=True)



def main() -> None:
    """Launch the desktop GUI."""
    app = CropFusionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
