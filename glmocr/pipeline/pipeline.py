"""GLM-OCR Pipeline

Three-stage async document parsing pipeline.  ``process()`` yields one
``PipelineResult`` per input unit (one image or one PDF).

Stages (all always enabled):
  1. PageLoader   — load images / PDF pages
  2. LayoutDetector — detect regions per page
  3. OCRClient    — recognise each region via VLM

Extension points:
  * Pass a custom ``layout_detector`` or ``result_formatter`` to the constructor.
  * Subclass ``Pipeline`` and override ``process()``.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional

from glmocr.dataloader import PageLoader
from glmocr.ocr_client import OCRClient
from glmocr.parser_result import PipelineResult
from glmocr.postprocess import ResultFormatter
from glmocr.utils.logging import get_logger

from glmocr.pipeline._common import extract_image_urls, extract_ocr_content, make_original_inputs
from glmocr.pipeline._state import PipelineState
from glmocr.pipeline._workers import data_loading_worker, layout_worker, recognition_worker
from glmocr.pipeline._unit_tracker import UnitTracker

if TYPE_CHECKING:
    from glmocr.config import PipelineConfig
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)


class Pipeline:
    """GLM-OCR pipeline.

    Processing flow:
      1. PageLoader:      load images / PDF into pages
      2. LayoutDetector:  detect regions
      3. OCRClient:       call OCR service
      4. ResultFormatter: format outputs

    Args:
        config: PipelineConfig instance.
        layout_detector: Custom layout detector (optional).
        result_formatter: Custom result formatter (optional).

    Example::

        from glmocr.config import load_config

        cfg = load_config()
        pipeline = Pipeline(cfg.pipeline)
        for result in pipeline.process(request_data):
            result.save(output_dir="./results")
    """

    def __init__(
        self,
        config: "PipelineConfig",
        layout_detector: Optional["BaseLayoutDetector"] = None,
        result_formatter: Optional[ResultFormatter] = None,
    ):
        self.config = config
        self.page_loader = PageLoader(config.page_loader)
        self.ocr_client = OCRClient(config.ocr_api)
        self.result_formatter = (
            result_formatter if result_formatter is not None
            else ResultFormatter(config.result_formatter)
        )

        if layout_detector is not None:
            self.layout_detector = layout_detector
        else:
            from glmocr.layout import PPDocLayoutDetector

            if PPDocLayoutDetector is None:
                from glmocr.layout import _raise_layout_import_error
                _raise_layout_import_error()

            self.layout_detector = PPDocLayoutDetector(config.layout)

        self.max_workers = config.max_workers
        self._page_maxsize = getattr(config, "page_maxsize", 100)
        self._region_maxsize = getattr(config, "region_maxsize", 800)
        self._current_state: Optional[PipelineState] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        request_data: Dict[str, Any],
        save_layout_visualization: bool = False,
        layout_vis_output_dir: Optional[str] = None,
        page_maxsize: Optional[int] = None,
        region_maxsize: Optional[int] = None,
    ) -> Generator[PipelineResult, None, None]:
        """Process a request; yield one ``PipelineResult`` per input unit.

        Uses three threads (load → layout → recognition) with bounded queues
        for back-pressure.

        Args:
            request_data: OpenAI-style request payload containing messages.
            save_layout_visualization: Save layout visualisation images.
            layout_vis_output_dir: Directory for visualisation output.
            page_maxsize: Bound for the page queue.
            region_maxsize: Bound for the region queue.

        Yields:
            One ``PipelineResult`` per input URL (image or PDF).
        """
        image_urls = extract_image_urls(request_data)

        if not image_urls:
            yield self._process_passthrough(request_data, layout_vis_output_dir)
            return

        num_units = len(image_urls)
        original_inputs = make_original_inputs(image_urls)

        state = PipelineState(
            page_maxsize=page_maxsize or self._page_maxsize,
            region_maxsize=region_maxsize or self._region_maxsize,
        )
        self._current_state = state

        tracker = UnitTracker(num_units)
        state.set_tracker(tracker)

        t1 = threading.Thread(
            target=data_loading_worker,
            args=(state, self.page_loader, image_urls),
            daemon=True,
        )
        t2 = threading.Thread(
            target=layout_worker,
            args=(state, self.layout_detector, save_layout_visualization, layout_vis_output_dir),
            daemon=True,
        )
        t3 = threading.Thread(
            target=recognition_worker,
            args=(state, self.page_loader, self.ocr_client, self.max_workers),
            daemon=True,
        )

        t1.start()
        t2.start()
        t3.start()

        try:
            yield from self._emit_results(state, tracker, original_inputs, layout_vis_output_dir)
            t1.join()
            t2.join()
            t3.join()
            state.raise_if_exceptions()
        finally:
            self._current_state = None

    def get_queue_stats(self) -> Optional[Dict[str, int]]:
        """Return current queue sizes, or ``None`` if no processing is active."""
        state = self._current_state
        if state is None:
            return None
        return {
            "page_queue_size": state.page_queue.qsize(),
            "page_queue_maxsize": state.page_queue.maxsize,
            "region_queue_size": state.region_queue.qsize(),
            "region_queue_maxsize": state.region_queue.maxsize,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the pipeline (layout detector + OCR client)."""
        logger.info("Starting Pipeline...")
        self.layout_detector.start()
        self.ocr_client.start()
        logger.info("Pipeline started!")

    def stop(self):
        """Stop the pipeline."""
        logger.info("Stopping Pipeline...")
        self.ocr_client.stop()
        self.layout_detector.stop()
        logger.info("Pipeline stopped!")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_passthrough(
        self,
        request_data: Dict[str, Any],
        layout_vis_output_dir: Optional[str],
    ) -> PipelineResult:
        """No image URLs — forward the request directly to the OCR API."""
        request_data = self.page_loader.build_request(request_data)
        response, status_code = self.ocr_client.process(request_data)
        if status_code != 200:
            raise Exception(
                f"OCR request failed: {response}, status_code: {status_code}"
            )
        content = extract_ocr_content(response)
        json_result, markdown_result = self.result_formatter.format_ocr_result(content)
        return PipelineResult(
            json_result=json_result,
            markdown_result=markdown_result,
            original_images=[],
            layout_vis_dir=layout_vis_output_dir,
        )

    def _emit_results(
        self,
        state: PipelineState,
        tracker: UnitTracker,
        original_inputs: List[str],
        layout_vis_output_dir: Optional[str],
    ) -> Generator[PipelineResult, None, None]:
        """Wait for units to complete and yield their formatted results.

        A unit enters the ready queue when:
          - ``finalize_unit`` has been called (region count is known), AND
          - all its regions have been recognised (``on_region_done`` counter
            reached the target).

        ``None`` from the ready queue signals a pipeline error (shutdown).
        """
        emitted: set = set()
        while len(emitted) < tracker.num_units:
            u = tracker.wait_next_ready_unit()
            if u is None:
                break
            if u in emitted:
                continue

            region_count = tracker.unit_region_count[u]
            if region_count is None:
                tracker._ready_queue.put(u)
                continue

            results = state.snapshot_recognition_results()
            page_indices = tracker.unit_image_indices[u]
            page_set = set(page_indices)
            count = sum(1 for r in results if r["page_idx"] in page_set)
            if count < region_count:
                tracker._ready_queue.put(u)
                continue

            page_to_pos = {idx: k for k, idx in enumerate(page_indices)}
            grouped: List[List[Dict]] = [[] for _ in page_indices]
            for r in results:
                pos = page_to_pos.get(r["page_idx"])
                if pos is not None:
                    grouped[pos].append(r["region"])

            json_u, md_u = self.result_formatter.process(grouped)
            yield PipelineResult(
                json_result=json_u,
                markdown_result=md_u,
                original_images=[original_inputs[u]],
                layout_vis_dir=layout_vis_output_dir,
                layout_image_indices=page_indices,
            )
            emitted.add(u)
