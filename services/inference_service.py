from __future__ import annotations

from pathlib import Path

from core.schemas import Detection


class InferenceService:
    def __init__(self, model_path: Path, confidence: float = 0.35, iou: float = 0.45, class_confidence: dict[str, float] | None = None, imgsz: int = 640):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is not installed") from exc
        if not model_path.exists(): raise FileNotFoundError(f"model not found: {model_path}")
        self.model = YOLO(str(model_path)); self.confidence = confidence; self.iou = iou; self.class_confidence = class_confidence or {}; self.imgsz = imgsz

    def predict(self, source: str | Path, save_path: Path | None = None) -> list[dict]:
        prediction_confidence = min([self.confidence, *self.class_confidence.values()]) if self.class_confidence else self.confidence
        results = self.model.predict(source=str(source), conf=prediction_confidence, iou=self.iou, imgsz=self.imgsz, verbose=False, save=False)
        output: list[dict] = []; names = self.model.names
        for result in results:
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
            item = {"source": str(result.path), "detections": [d.as_dict() for d in detections]}
            if save_path is not None:
                import cv2
                if result.boxes is not None and len(keep_indices) != len(result.boxes):
                    result.boxes = result.boxes[keep_indices]
                save_path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(save_path), result.plot()); item["annotated_path"] = str(save_path)
            output.append(item)
        return output
