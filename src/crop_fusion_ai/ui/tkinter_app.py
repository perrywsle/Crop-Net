"""Tkinter desktop demo for Crop Fusion AI."""

import json
import tkinter as tk
from datetime import UTC, datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from pydantic import ValidationError

from crop_fusion_ai.config.schemas import (
    CropFeatures,
    ImagePrediction,
    WeatherFeatures,
    YieldPrediction,
)
from crop_fusion_ai.inference import CropFusionPipeline
from crop_fusion_ai.models import SegmentationResult

DEFAULT_YIELD_MODEL_PATH = Path("models/yield_model/yield_regressor.joblib")
DEFAULT_SEGMENTATION_MODEL_PATH = Path("models/image_model/segmentation_model.pt")
REPORTS_DIR = Path("reports")


class CropFusionTkinterApp:
    """Small Tkinter application for demo inference."""

    def __init__(self, root: tk.Tk) -> None:
        """Build the Tkinter UI."""
        self.root = root
        self.root.title("Crop Fusion AI Demo")
        self.root.geometry("760x720")

        self.image_path: Path | None = None
        self.latest_image_prediction: ImagePrediction | None = None
        self.latest_segmentation_result: SegmentationResult | None = None
        self.latest_yield_prediction: YieldPrediction | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self.pipeline = CropFusionPipeline(
            yield_model_path=DEFAULT_YIELD_MODEL_PATH,
            segmentation_model_path=DEFAULT_SEGMENTATION_MODEL_PATH,
        )
        self.input_vars: dict[str, tk.StringVar] = {}
        self.output_vars: dict[str, tk.StringVar] = {}
        self.warnings_text: tk.Text

        self._build_layout()

    def _build_layout(self) -> None:
        """Create and arrange all UI widgets."""
        main_frame = ttk.Frame(self.root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        image_frame = ttk.LabelFrame(main_frame, text="Image Input", padding=12)
        image_frame.pack(fill=tk.X)
        ttk.Button(
            image_frame,
            text="Select Image",
            command=self._select_image,
        ).grid(row=0, column=0, sticky=tk.W)
        self.image_path_var = tk.StringVar(value="No image selected")
        ttk.Label(
            image_frame,
            textvariable=self.image_path_var,
            wraplength=560,
        ).grid(row=0, column=1, padx=12, sticky=tk.W)
        self.image_preview_label = ttk.Label(image_frame, text="No preview")
        self.image_preview_label.grid(row=1, column=0, columnspan=2, pady=(12, 0))
        image_frame.columnconfigure(1, weight=1)

        input_frame = ttk.LabelFrame(
            main_frame, text="Crop and Weather Features", padding=12
        )
        input_frame.pack(fill=tk.X, pady=12)
        self._build_input_fields(input_frame)

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=(0, 12))
        ttk.Button(
            action_frame,
            text="Run Prediction",
            command=self._run_prediction,
        ).pack(side=tk.LEFT)
        ttk.Button(
            action_frame,
            text="Save Result",
            command=self._save_result,
        ).pack(side=tk.LEFT, padx=8)

        output_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding=12)
        output_frame.pack(fill=tk.BOTH, expand=True)
        self._build_output_fields(output_frame)

    def _build_input_fields(self, parent: ttk.LabelFrame) -> None:
        """Build text-entry fields for schema-backed inputs."""
        defaults = {
            "crop_type": "corn",
            "region": "01003",
            "year": "2022",
            "planting_age_days": "90",
            "temperature_mean": "25.0",
            "temperature_min": "",
            "temperature_max": "",
            "rainfall_total": "140.0",
            "humidity_mean": "70.0",
            "solar_radiation_mean": "",
        }
        for index, (field_name, default_value) in enumerate(defaults.items()):
            row = index // 2
            column = (index % 2) * 2
            self.input_vars[field_name] = tk.StringVar(value=default_value)
            ttk.Label(parent, text=field_name).grid(
                row=row,
                column=column,
                sticky=tk.W,
                padx=(0, 8),
                pady=4,
            )
            ttk.Entry(
                parent,
                textvariable=self.input_vars[field_name],
                width=24,
            ).grid(row=row, column=column + 1, sticky=tk.EW, pady=4)

        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)

    def _build_output_fields(self, parent: ttk.LabelFrame) -> None:
        """Build read-only output fields."""
        output_fields = (
            "disease_class",
            "health_score",
            "image_confidence",
            "segmentation_coverage",
            "segmentation_confidence",
            "predicted_yield",
            "unit",
        )
        for index, field_name in enumerate(output_fields):
            self.output_vars[field_name] = tk.StringVar(value="-")
            ttk.Label(parent, text=field_name).grid(
                row=index,
                column=0,
                sticky=tk.W,
                pady=4,
            )
            ttk.Label(parent, textvariable=self.output_vars[field_name]).grid(
                row=index,
                column=1,
                sticky=tk.W,
                pady=4,
            )

        ttk.Label(parent, text="warnings").grid(
            row=0, column=2, sticky=tk.NW, padx=(24, 8)
        )
        self.warnings_text = tk.Text(parent, height=10, width=48, wrap=tk.WORD)
        self.warnings_text.grid(
            row=1, column=2, rowspan=5, sticky=tk.NSEW, padx=(24, 0)
        )
        self.warnings_text.configure(state=tk.DISABLED)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(5, weight=1)

    def _select_image(self) -> None:
        """Let the user select an image file."""
        selected_path = filedialog.askopenfilename(
            title="Select plant image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*"),
            ],
        )
        if not selected_path:
            return
        self.image_path = Path(selected_path)
        self.image_path_var.set(str(self.image_path))
        self._show_preview(self.image_path)

    def _run_prediction(self) -> None:
        """Validate inputs, run the pipeline, and update the UI."""
        if self.image_path is None:
            messagebox.showerror("Missing Image", "Please select an image first.")
            return

        try:
            weather = self._build_weather_features()
            crop = self._build_crop_features()
            image_prediction, segmentation_result, yield_prediction = (
                self.pipeline.predict_from_image_and_features(
                    image_path=self.image_path,
                    weather=weather,
                    crop=crop,
                )
            )
        except (ValueError, FileNotFoundError, RuntimeError, ValidationError) as exc:
            messagebox.showerror("Prediction Error", str(exc))
            return

        self.latest_image_prediction = image_prediction
        self.latest_segmentation_result = segmentation_result
        self.latest_yield_prediction = yield_prediction
        self._display_results(image_prediction, segmentation_result, yield_prediction)

    def _save_result(self) -> None:
        """Save the latest prediction result as JSON under reports/."""
        if self.latest_image_prediction is None or self.latest_yield_prediction is None:
            messagebox.showerror("No Result", "Run a prediction before saving.")
            return

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        output_path = REPORTS_DIR / f"crop_fusion_result_{timestamp}.json"
        payload = {
            "image_path": str(self.image_path) if self.image_path is not None else None,
            "image_prediction": self.latest_image_prediction.model_dump(),
            "segmentation_result": (
                self.latest_segmentation_result.model_dump(mode="json")
                if self.latest_segmentation_result is not None
                else None
            ),
            "yield_prediction": self.latest_yield_prediction.model_dump(),
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")

        messagebox.showinfo("Result Saved", f"Saved result to {output_path}")

    def _build_weather_features(self) -> WeatherFeatures:
        """Create validated weather features from entry fields."""
        return WeatherFeatures(
            temperature_mean=self._required_float("temperature_mean"),
            temperature_min=self._optional_float("temperature_min"),
            temperature_max=self._optional_float("temperature_max"),
            rainfall_total=self._required_float("rainfall_total"),
            humidity_mean=self._optional_float("humidity_mean"),
            solar_radiation_mean=self._optional_float("solar_radiation_mean"),
        )

    def _build_crop_features(self) -> CropFeatures:
        """Create validated crop features from entry fields."""
        return CropFeatures(
            crop_type=self._required_string("crop_type"),
            region=self._optional_string("region"),
            year=self._required_int("year"),
            planting_age_days=self._optional_int("planting_age_days"),
        )

    def _display_results(
        self,
        image_prediction: ImagePrediction,
        segmentation_result: SegmentationResult,
        yield_prediction: YieldPrediction,
    ) -> None:
        """Update result widgets with model outputs."""
        self.output_vars["disease_class"].set(image_prediction.disease_class)
        self.output_vars["health_score"].set(f"{image_prediction.health_score:.3f}")
        self.output_vars["image_confidence"].set(f"{image_prediction.confidence:.3f}")
        self.output_vars["segmentation_coverage"].set(
            f"{segmentation_result.coverage_ratio:.3f}"
        )
        self.output_vars["segmentation_confidence"].set(
            f"{segmentation_result.confidence:.3f}"
        )
        self.output_vars["predicted_yield"].set(
            f"{yield_prediction.predicted_yield:.3f}"
        )
        self.output_vars["unit"].set(yield_prediction.unit)
        self._show_preview(segmentation_result.overlay_path)

        warnings = yield_prediction.warnings or ["No warnings."]
        self.warnings_text.configure(state=tk.NORMAL)
        self.warnings_text.delete("1.0", tk.END)
        self.warnings_text.insert(tk.END, "\n".join(f"- {item}" for item in warnings))
        self.warnings_text.configure(state=tk.DISABLED)

    def _show_preview(self, image_path: Path) -> None:
        """Render a thumbnail in the UI image preview area."""
        with Image.open(image_path) as image:
            preview = image.convert("RGB")
        preview.thumbnail((360, 240))
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.image_preview_label.configure(image=self.preview_photo, text="")

    def _required_string(self, field_name: str) -> str:
        """Read a required string input."""
        value = self.input_vars[field_name].get().strip()
        if not value:
            msg = f"{field_name} is required"
            raise ValueError(msg)
        return value

    def _optional_string(self, field_name: str) -> str | None:
        """Read an optional string input."""
        value = self.input_vars[field_name].get().strip()
        return value or None

    def _required_float(self, field_name: str) -> float:
        """Read a required float input."""
        value = self.input_vars[field_name].get().strip()
        if not value:
            msg = f"{field_name} is required"
            raise ValueError(msg)
        try:
            return float(value)
        except ValueError as exc:
            msg = f"{field_name} must be a number"
            raise ValueError(msg) from exc

    def _optional_float(self, field_name: str) -> float | None:
        """Read an optional float input."""
        value = self.input_vars[field_name].get().strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            msg = f"{field_name} must be a number"
            raise ValueError(msg) from exc

    def _required_int(self, field_name: str) -> int:
        """Read a required integer input."""
        value = self.input_vars[field_name].get().strip()
        if not value:
            msg = f"{field_name} is required"
            raise ValueError(msg)
        try:
            return int(value)
        except ValueError as exc:
            msg = f"{field_name} must be an integer"
            raise ValueError(msg) from exc

    def _optional_int(self, field_name: str) -> int | None:
        """Read an optional integer input."""
        value = self.input_vars[field_name].get().strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError as exc:
            msg = f"{field_name} must be an integer"
            raise ValueError(msg) from exc


def main() -> None:
    """Launch the Tkinter desktop demo."""
    root = tk.Tk()
    CropFusionTkinterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
