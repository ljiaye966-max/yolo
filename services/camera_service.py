from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock

import av
import cv2
import numpy as np
from streamlit_webrtc import VideoProcessorBase

from services.database_service import EventStore
from services.inference_service import InferenceService
from services.safety_service import SafetyRuleEngine


class PPEVideoProcessor(VideoProcessorBase):
    """Process browser camera frames without calling Streamlit APIs in the callback."""

    def __init__(
        self,
        inference_service: InferenceService,
        event_store: EventStore,
        event_dir: Path,
        rule_kwargs: dict,
    ) -> None:
        self.inference_service = inference_service
        self.event_store = event_store
        self.event_dir = event_dir
        self.engine = SafetyRuleEngine(min_consecutive_frames=3, cooldown_seconds=10, **rule_kwargs)
        self._lock = Lock()
        self.last_detections: list[dict] = []
        self.last_error = ""
        self.event_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        try:
            annotated, detections = self.inference_service.predict_frame(image)
            events = self.engine.evaluate(detections, "camera")
            if events:
                self.event_dir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                snapshot = self.event_dir / f"camera_{stamp}.jpg"
                cv2.imwrite(str(snapshot), annotated)
                for event in events:
                    event.image_path = str(snapshot)
                    event.id = self.event_store.add(event)
                with self._lock:
                    self.event_count += len(events)
            with self._lock:
                self.last_detections = [d.as_dict() for d in detections]
                self.last_error = ""
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")
        except Exception as exc:  # Keep the camera stream alive and expose the error in the UI.
            with self._lock:
                self.last_error = str(exc)
            return frame

    def snapshot(self) -> dict:
        with self._lock:
            return {"detections": list(self.last_detections), "event_count": self.event_count, "error": self.last_error}
