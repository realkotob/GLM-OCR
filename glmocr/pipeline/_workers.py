"""Thread workers for the three-stage async pipeline.

Stage 1 (data_loading_worker):   Load pages from URLs → page_queue
Stage 2 (layout_worker):         Layout detection     → region_queue
Stage 3 (recognition_worker):    Parallel OCR         → recognition_results

Queue message formats:

    page_queue::
        {"identifier": "image",     "page_idx": int, "unit_idx": int,
         "image": PIL.Image}
        {"identifier": "unit_done", "unit_idx": int}   ← all pages for this
                                                          unit have been queued
        {"identifier": "done"}
        {"identifier": "error"}

    region_queue::
        {"identifier": "region", "page_idx": int, "cropped_image": PIL.Image,
         "region": dict, "task_type": str}
        {"identifier": "done"}
        {"identifier": "error"}
"""

from __future__ import annotations

import queue
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from glmocr.pipeline._common import (
    IDENTIFIER_DONE,
    IDENTIFIER_ERROR,
    IDENTIFIER_IMAGE,
    IDENTIFIER_REGION,
    IDENTIFIER_UNIT_DONE,
)
from glmocr.pipeline._state import PipelineState
from glmocr.utils.image_utils import crop_image_region
from glmocr.utils.logging import get_logger

if TYPE_CHECKING:
    from glmocr.dataloader import PageLoader
    from glmocr.layout.base import BaseLayoutDetector

logger = get_logger(__name__)


# ======================================================================
# Stage 1: Data Loading
# ======================================================================

def data_loading_worker(
    state: PipelineState,
    page_loader: "PageLoader",
    image_urls: List[str],
) -> None:
    """Load pages from *image_urls* and push them onto ``state.page_queue``.

    For each page that is loaded, ``state.register_page()`` is called
    **before** the page message is enqueued, so that the tracker's
    ``page → unit`` mapping is always available by the time Stage 3 calls
    ``on_region_done``.

    After all pages for each input URL (unit) have been enqueued, a
    ``IDENTIFIER_UNIT_DONE`` sentinel is sent so that the layout worker can
    call ``state.finalize_unit()`` without waiting for *all* units to finish.

    Units that produce zero pages (e.g. broken URLs) still receive a
    ``UNIT_DONE`` sentinel so the tracker can finalise them with
    ``region_count=0``.
    """
    num_units = len(image_urls)
    page_idx = 0
    unit_indices_list: List[int] = []
    prev_unit_idx: Optional[int] = None
    sent_unit_done: set = set()
    try:
        for page, unit_idx in page_loader.iter_pages_with_unit_indices(image_urls):
            if prev_unit_idx is not None and unit_idx != prev_unit_idx:
                state.page_queue.put({
                    "identifier": IDENTIFIER_UNIT_DONE,
                    "unit_idx": prev_unit_idx,
                })
                sent_unit_done.add(prev_unit_idx)

            state.register_page(page_idx, unit_idx)
            state.images_dict[page_idx] = page
            state.page_queue.put({
                "identifier": IDENTIFIER_IMAGE,
                "page_idx": page_idx,
                "unit_idx": unit_idx,
                "image": page,
            })
            unit_indices_list.append(unit_idx)
            page_idx += 1
            state.num_images_loaded[0] = page_idx
            state.unit_indices_holder[0] = list(unit_indices_list)
            prev_unit_idx = unit_idx

        if prev_unit_idx is not None:
            state.page_queue.put({
                "identifier": IDENTIFIER_UNIT_DONE,
                "unit_idx": prev_unit_idx,
            })
            sent_unit_done.add(prev_unit_idx)

        for u in range(num_units):
            if u not in sent_unit_done:
                state.page_queue.put({
                    "identifier": IDENTIFIER_UNIT_DONE,
                    "unit_idx": u,
                })

        state.page_queue.put({"identifier": IDENTIFIER_DONE})
    except Exception as e:
        logger.exception("Data loading worker error: %s", e)
        state.num_images_loaded[0] = page_idx
        state.unit_indices_holder[0] = list(unit_indices_list)
        state.record_exception("DataLoadingWorker", e)
        state.page_queue.put({"identifier": IDENTIFIER_ERROR})


# ======================================================================
# Stage 2: Layout Detection
# ======================================================================

def layout_worker(
    state: PipelineState,
    layout_detector: "BaseLayoutDetector",
    save_visualization: bool,
    vis_output_dir: Optional[str],
) -> None:
    """Consume pages, run layout detection in batches, push regions.

    When a ``IDENTIFIER_UNIT_DONE`` sentinel arrives from Stage 1, the
    current batch is flushed immediately (it contains the last pages for
    that unit) and ``state.finalize_unit()`` is called with the total
    region count for that unit.  This lets the main thread start emitting
    results for the completed unit without waiting for later units.
    """
    try:
        batch_images: List[Any] = []
        batch_page_indices: List[int] = []
        batch_unit_indices: List[int] = []
        global_start_idx = 0

        # page_indices seen so far per unit, used to compute region counts.
        unit_page_indices: Dict[int, List[int]] = {}

        while True:
            try:
                msg = state.page_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            identifier = msg["identifier"]

            if identifier == IDENTIFIER_IMAGE:
                unit_idx = msg["unit_idx"]
                batch_images.append(msg["image"])
                batch_page_indices.append(msg["page_idx"])
                batch_unit_indices.append(unit_idx)
                if unit_idx not in unit_page_indices:
                    unit_page_indices[unit_idx] = []
                unit_page_indices[unit_idx].append(msg["page_idx"])

                if len(batch_images) >= layout_detector.batch_size:
                    _flush_layout_batch(
                        state, layout_detector, batch_images, batch_page_indices,
                        save_visualization, vis_output_dir, global_start_idx,
                    )
                    global_start_idx += len(batch_page_indices)
                    batch_images, batch_page_indices, batch_unit_indices = [], [], []

            elif identifier == IDENTIFIER_UNIT_DONE:
                unit_idx = msg["unit_idx"]
                # Flush any remaining pages in the batch (they all belong to
                # this unit since t1 sends UNIT_DONE after the last page).
                if batch_images:
                    _flush_layout_batch(
                        state, layout_detector, batch_images, batch_page_indices,
                        save_visualization, vis_output_dir, global_start_idx,
                    )
                    global_start_idx += len(batch_page_indices)
                    batch_images, batch_page_indices, batch_unit_indices = [], [], []

                # All pages for this unit have been layout-detected; compute
                # total region count and tell the tracker.
                pages_for_unit = unit_page_indices.get(unit_idx, [])
                region_count = sum(
                    len(state.layout_results_dict.get(pi, []))
                    for pi in pages_for_unit
                )
                state.finalize_unit(unit_idx, region_count)
                logger.debug(
                    "Unit %d finalised: %d pages, %d regions",
                    unit_idx, len(pages_for_unit), region_count,
                )

            elif identifier == IDENTIFIER_DONE:
                if batch_images:
                    _flush_layout_batch(
                        state, layout_detector, batch_images, batch_page_indices,
                        save_visualization, vis_output_dir, global_start_idx,
                    )
                state.region_queue.put({"identifier": IDENTIFIER_DONE})
                break

            elif identifier == IDENTIFIER_ERROR:
                state.region_queue.put({"identifier": IDENTIFIER_ERROR})
                break

    except Exception as e:
        logger.exception("Layout worker error: %s", e)
        state.record_exception("LayoutWorker", e)
        state.region_queue.put({"identifier": IDENTIFIER_ERROR})


def _flush_layout_batch(
    state: PipelineState,
    layout_detector: "BaseLayoutDetector",
    batch_images: List[Any],
    batch_page_indices: List[int],
    save_visualization: bool,
    vis_output_dir: Optional[str],
    global_start_idx: int,
) -> None:
    """Run layout detection on one batch and enqueue the resulting regions."""
    layout_results = layout_detector.process(
        batch_images,
        save_visualization=save_visualization and vis_output_dir is not None,
        visualization_output_dir=vis_output_dir,
        global_start_idx=global_start_idx,
    )
    for page_idx, image, layout_result in zip(
        batch_page_indices, batch_images, layout_results
    ):
        state.layout_results_dict[page_idx] = layout_result
        for region in layout_result:
            cropped = crop_image_region(image, region["bbox_2d"], region["polygon"])
            state.region_queue.put({
                "identifier": IDENTIFIER_REGION,
                "page_idx": page_idx,
                "cropped_image": cropped,
                "region": region,
            })


# ======================================================================
# Stage 3: VLM Recognition
# ======================================================================

def recognition_worker(
    state: PipelineState,
    page_loader: "PageLoader",
    ocr_client: Any,
    max_workers: int,
) -> None:
    """Consume regions, run parallel OCR, store results."""
    try:
        concurrency = min(max_workers, 128)
        executor = ThreadPoolExecutor(max_workers=concurrency)
        futures: Dict[Any, Dict[str, Any]] = {}
        pending_skip: List[Dict[str, Any]] = []
        processing_complete = False

        while True:
            _collect_done_futures(futures, state)

            if len(futures) >= concurrency:
                _wait_for_any(futures)
                _collect_done_futures(futures, state)
                continue

            try:
                msg = state.region_queue.get(timeout=0.01)
            except queue.Empty:
                if processing_complete and not futures:
                    _flush_pending_skips(pending_skip, state)
                    break
                if futures:
                    _wait_for_any(futures)
                continue

            identifier = msg["identifier"]

            if identifier == IDENTIFIER_REGION:
                if msg["region"]["task_type"] == "skip":
                    pending_skip.append(msg)
                else:
                    req = page_loader.build_request_from_image(
                        msg["cropped_image"], msg["region"]["task_type"],
                    )
                    future = executor.submit(ocr_client.process, req)
                    futures[future] = msg

            elif identifier == IDENTIFIER_DONE:
                processing_complete = True

            elif identifier == IDENTIFIER_ERROR:
                break

        for future in as_completed(futures.keys()):
            _handle_future_result(future, futures, state)
        executor.shutdown(wait=True)

    except Exception as e:
        logger.exception("Recognition worker error: %s", e)
        state.record_exception("RecognitionWorker", e)


# ------------------------------------------------------------------
# Recognition helpers
# ------------------------------------------------------------------

def _collect_done_futures(
    futures: Dict[Any, Dict[str, Any]],
    state: PipelineState,
) -> None:
    for f in list(futures):
        if f.done():
            _handle_future_result(f, futures, state)


def _handle_future_result(
    future: Any,
    futures: Dict[Any, Dict[str, Any]],
    state: PipelineState,
) -> None:
    msg = futures.pop(future)
    region = msg["region"]
    page_idx = msg["page_idx"]
    try:
        response, status_code = future.result()
        if status_code == 200:
            region["content"] = response["choices"][0]["message"]["content"].strip()
        else:
            region["content"] = ""
    except Exception as e:
        logger.warning("Recognition failed: %s", e)
        region["content"] = ""
    state.add_recognition_result(page_idx, region)


def _flush_pending_skips(
    pending: List[Dict[str, Any]],
    state: PipelineState,
) -> None:
    for msg in pending:
        msg["region"]["content"] = None
        state.add_recognition_result(msg["page_idx"], msg["region"])


def _wait_for_any(futures: Dict) -> None:
    done_list = [f for f in futures if f.done()]
    if not done_list:
        try:
            next(as_completed(futures.keys(), timeout=0.05))
        except Exception:
            pass
