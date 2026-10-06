from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from core.config import get_settings
from core.schemas import Detection
from services.database_service import EventStore
from services.inference_service import InferenceService
from services.llm_service import generate_llm_report
from services.safety_service import SafetyRuleEngine


@st.cache_resource(show_spinner=False)
def get_inference_service(model_path: str, confidence: float, iou: float, imgsz: int, class_confidence: tuple[tuple[str, float], ...]) -> InferenceService:
    """Load one YOLO model per model/config combination, not once per rerun."""
    return InferenceService(Path(model_path), confidence, iou, dict(class_confidence), imgsz)


def main() -> int:
    settings = get_settings(); settings.ensure_directories(); st.set_page_config(page_title="工地 PPE 智能监控", layout="wide"); st.title("工地安全生产智能监控系统"); st.caption("教学/原型系统：所有告警均为疑似违规，必须人工复核。")
    store = EventStore(settings.database_path); tabs = st.tabs(["图片检测", "历史事件", "数据质量", "结构化报告"])
    with tabs[0]:
        uploaded = st.file_uploader("上传图片", type=["jpg", "jpeg", "png", "bmp"])
        if uploaded:
            st.image(uploaded, caption="原始图片", use_container_width=True)
            if st.button("运行图片检测"):
                input_dir = settings.runs_dir / "web_inputs"; input_dir.mkdir(parents=True, exist_ok=True)
                input_path = input_dir / uploaded.name; input_path.write_bytes(uploaded.getvalue())
                if not settings.model_path.exists():
                    st.warning(f"未找到模型权重：{settings.model_path}。请先完成训练，或修改 configs/default.yaml。")
                else:
                    with st.spinner("正在推理..."):
                        class_confidence = tuple(sorted((str(k), float(v)) for k, v in settings.model.get("class_confidence", {}).items()))
                        service = get_inference_service(str(settings.model_path.resolve()), settings.model.get("conf", 0.10), settings.model.get("iou", 0.45), int(settings.model.get("inference_imgsz", 960)), class_confidence)
                        results = service.predict(input_path, input_dir / f"{input_path.stem}_annotated.jpg")
                    if results and results[0].get("annotated_path"):
                        st.image(results[0]["annotated_path"], caption="检测结果", use_container_width=True)
                    detections = [Detection(d["class_name"], d["confidence"], tuple(d["bbox"])) for d in (results[0]["detections"] if results else [])]
                    st.dataframe([d.as_dict() for d in detections], use_container_width=True)
                    events = SafetyRuleEngine(
                        settings.rules.get("min_confidence", 0.35),
                        min_consecutive_frames=1,
                        cooldown_seconds=0,
                        infer_missing_helmet_requires_vest=settings.rules.get("infer_missing_helmet_requires_vest", True),
                        missing_helmet_min_person_height_px=settings.rules.get("missing_helmet_min_person_height_px", 80),
                        helmet_upper_zone=settings.rules.get("helmet_upper_zone", 0.45),
                        class_min_confidence=settings.model.get("class_confidence", {}),
                    ).evaluate(detections, str(input_path), results[0].get("annotated_path") if results else None)
                    for event in events:
                        event.id = store.add(event)
                    if events: st.warning(f"生成 {len(events)} 条疑似事件，状态为 needs_human_review。")
                    else: st.success("本张图片没有生成疑似事件；这不等于现场一定不存在风险。")
    with tabs[1]:
        events = store.list(200); st.metric("已记录事件", len(events)); st.dataframe(events, use_container_width=True) if events else st.info("暂无事件。"); st.download_button("下载 CSV", store.export_csv(), "events.csv", "text/csv")
    with tabs[2]:
        report_path = settings.report_dir / "dataset_quality.json"; st.json(json.loads(report_path.read_text(encoding="utf-8"))) if report_path.exists() else st.info("请先运行 scripts/validate_dataset.py。")
    with tabs[3]:
        data = {"event_summary": store.count_by_type(), "events": store.list(200)}
        if st.button("生成安全分析摘要"):
            result = generate_llm_report(data); st.write(result["text"]); st.caption(f"模式：{result['mode']}；仅基于结构化事件，不推断事故原因或人员身份。")
    return 0


if __name__ == "__main__": raise SystemExit(main())
