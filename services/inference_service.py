from __future__ import annotations

from pathlib import Path

import numpy as np

from core.schemas import Detection


class InferenceService:
    def __init__(self, model_path: Path, confidence: float = 0.35, iou: float = 0.45, class_confidence: dict[str, float] | None = None, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc
        if not model_path.exists(): raise FileNotFoundError(f"model not found: {model_path}")
        self.model = YOLO(str(model_path)); self.confidence = confidence; self.iou = iou; self.class_confidence = class_confidence or {}; self.imgsz = imgsz

    def _predict_results(self, source: str | Path | np.ndarray):
        prediction_confidence = min([self.confidence, *self.class_confidence.values()]) if self.class_confidence else self.confidence
        return self.model.predict(source=source, conf=prediction_confidence, iou=self.iou, imgsz=self.imgsz, verbose=False, save=False)

    def _extract_result(self, result) -> tuple[list[Detection], list[int]]:
        names = self.model.names
        detections: list[Detection] = []
        keep_indices: list[int] = []
        if result.boxes is not None:
            for index, box in enumerate(result.boxes):
                cls_id = int(box.cls.item()); confidence = float(box.conf.item()); xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
                class_name = str(names[cls_id])
                threshold = self.class_confidence.get(class_name, self.confidence)
                if confidence < threshold:
                    continue
                keep_indices.append(index)
                detections.append(Detection(class_name, confidence, xyxy))
        return detections, keep_indices

    def _filter_result_boxes(self, result, keep_indices: list[int]) -> None:
        if result.boxes is not None and len(keep_indices) != len(result.boxes):
            result.boxes = result.boxes[keep_indices]

    def predict(self, source: str | Path | np.ndarray, save_path: Path | None = None) -> list[dict]:
        results = self._predict_results(source)
        output: list[dict] = []
        for result in results:
            detections, keep_indices = self._extract_result(result)
            result_source = getattr(result, "path", None) or ("camera" if isinstance(source, np.ndarray) else str(source))
            item = {"source": str(result_source), "detections": [d.as_dict() for d in detections]}
            if save_path is not None:
                import cv2
                self._filter_result_boxes(result, keep_indices)
                save_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(save_path), result.plot()); item["annotated_path"] = str(save_path)
            output.append(item)
        return output

    def predict_frame(self, frame: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        """Run one BGR camera frame and return an annotated BGR frame."""
        results = self._predict_results(frame)
        if not results:
            return frame, []
        result = results[0]
        detections, keep_indices = self._extract_result(result)
        self._filter_result_boxes(result, keep_indices)
        return result.plot(), detections
