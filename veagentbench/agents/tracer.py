from veadk.tracing.telemetry.opentelemetry_tracer import OpentelemetryTracer
import os
from typing_extensions import override
import json
from veadk.utils.logger import get_logger
from pydantic import BaseModel

logger = get_logger(__name__)

class VeOpentelemetryTracer(OpentelemetryTracer):

    trace_folder : str = './trace'

    def set_trace_folder(self, trace_folder: str):
        self.trace_folder = trace_folder
        if not os.path.exists(self.trace_folder):
            os.makedirs(self.trace_folder, exist_ok=True)
        logger.info(f"Trace folder is set to {self.trace_folder}")

    def get_trace_folder(self) -> str:
        return self.trace_folder

    def model_post_init(self, context):
        self.set_trace_folder(self.trace_folder)
        return super().model_post_init(context)

    def get_trace_file_path(self) -> str:
        return self._trace_file_path if hasattr(self, '_trace_file_path') else ""

    def get_spans(self, session_id) -> list:
        spans = self._inmemory_exporter._exporter.get_finished_spans(  # type: ignore
            session_id=session_id
        )
        data = (
            [
                {
                    "name": s.name,
                    "span_id": s.context.span_id,
                    "trace_id": s.context.trace_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "attributes": dict(s.attributes),
                    "parent_span_id": s.parent.span_id if s.parent else None,
                }
                for s in spans
            ]
            if spans
            else []
        )
        return data

    @override
    def dump(self, user_id = "unknown_user_id", session_id = "unknown_session_id", path = '/tmp') -> str:
        def _build_trace_file_path(path: str, user_id: str, session_id: str) -> str:
            return f"{path}/{self.name}_{user_id}_{session_id}_{self.trace_id}.json"

        if not self._inmemory_exporter:
            logger.warning(
                "InMemoryExporter is not initialized. Please check your tracer exporters."
            )
            return ""
        self.force_export()

        spans = self._inmemory_exporter._exporter.get_finished_spans(  # type: ignore
            session_id=session_id
        )
        data = (
            [
                {
                    "name": s.name,
                    "span_id": s.context.span_id,
                    "trace_id": s.context.trace_id,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "attributes": dict(s.attributes),
                    "parent_span_id": s.parent.span_id if s.parent else None,
                }
                for s in spans
            ]
            if spans
            else []
        )
        path = self.trace_folder if self.trace_folder else path
        self._trace_file_path = _build_trace_file_path(path, user_id, session_id)
        with open(self._trace_file_path, "w") as f:
            json.dump(
                data, f, indent=4, ensure_ascii=False
            )  # ensure_ascii=False to support Chinese characters

        logger.info(
            f"OpenTelemetryTracer dumps {len(spans)} spans to {self._trace_file_path}. Trace id: {self.trace_id} (hex)"
        )

        return self._trace_file_path
