"""Base parser result.

Defines common fields and JSON/Markdown save logic.
"""

from __future__ import annotations

import copy
import json
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from glmocr.utils.logging import get_logger
from glmocr.utils.markdown_utils import crop_and_replace_images, extract_image_refs

logger = get_logger(__name__)


class BaseParserResult(ABC):
    """Base parser result.

    Common interface: json_result, markdown_result, original_images; abstract save().
    """

    def __init__(
        self,
        json_result: Union[str, dict, list],
        markdown_result: Optional[str] = None,
        original_images: Optional[List[str]] = None,
    ):
        """Initialize.

        Args:
            json_result: JSON result (string, dict, or list).
            markdown_result: Markdown result (optional).
            original_images: Original image paths.
        """
        if isinstance(json_result, str):
            try:
                self.json_result: Union[str, dict, list] = json.loads(json_result)
            except json.JSONDecodeError:
                self.json_result = json_result
        else:
            self.json_result = json_result

        self.markdown_result = markdown_result
        self.original_images = [
            str(Path(p).absolute()) for p in (original_images or [])
        ]

    @abstractmethod
    def save(
        self,
        output_dir: Union[str, Path] = "./results",
        save_layout_visualization: bool = True,
    ) -> None:
        """Save result to disk. Subclasses implement layout vis etc."""
        pass

    @staticmethod
    def _build_image_path_map(
        markdown_text: str, image_prefix: str = "cropped"
    ) -> Dict[Tuple[int, ...], str]:
        """Build a mapping from (page_idx, *bbox) to the relative image path.

        The mapping is derived purely from the markdown image references so
        it stays in sync with what ``crop_and_replace_images`` will produce,
        without performing any file I/O here.
        """
        mapping: Dict[Tuple[int, ...], str] = {}
        refs = extract_image_refs(markdown_text)
        for idx, (page_idx, bbox, _) in enumerate(refs):
            key = (page_idx, *bbox)
            rel = f"imgs/{image_prefix}_page{page_idx}_idx{idx}.jpg"
            mapping[key] = rel
        return mapping

    @staticmethod
    def _annotate_json_image_paths(
        json_data: Any,
        image_path_map: Dict[Tuple[int, ...], str],
    ) -> Any:
        """Return a deep-copied json_data with ``image_path`` added to image regions.

        ``json_data`` is expected to be a list-of-pages (list of lists of region
        dicts).  For every region whose ``label`` is ``"image"``, the relative
        path is looked up by ``(page_idx, *bbox_2d)`` and written into the copy.
        The original ``json_data`` is never mutated.
        """
        if not image_path_map or not isinstance(json_data, list):
            return json_data

        result = []
        for page_idx, page in enumerate(json_data):
            if not isinstance(page, list):
                result.append(page)
                continue
            page_copy = []
            for region in page:
                if not isinstance(region, dict) or region.get("label") != "image":
                    page_copy.append(region)
                    continue
                bbox = region.get("bbox_2d")
                region_copy = copy.copy(region)
                if bbox:
                    key = (page_idx, *bbox)
                    rel = image_path_map.get(key)
                    if rel:
                        region_copy["image_path"] = rel
                page_copy.append(region_copy)
            result.append(page_copy)
        return result

    def _save_json_and_markdown(self, output_dir: Union[str, Path]) -> None:
        """Save JSON and Markdown to output_dir (by first image name or 'result')."""
        output_dir = Path(output_dir).absolute()
        if self.original_images:
            image_path = Path(self.original_images[0])
            output_path = output_dir / image_path.stem
        else:
            output_path = output_dir / "result"

        output_path.mkdir(parents=True, exist_ok=True)
        base_name = output_path.name

        # Build image_path_map from markdown refs so JSON can reference the
        # same filenames that crop_and_replace_images will produce below.
        image_path_map: Dict[Tuple[int, ...], str] = {}
        if self.markdown_result and self.original_images:
            image_path_map = self._build_image_path_map(
                self.markdown_result, image_prefix="cropped"
            )

        # JSON — annotate image regions with their relative image_path
        json_file = output_path / f"{base_name}.json"
        try:
            json_data = self.json_result
            if isinstance(json_data, str):
                try:
                    json_data = json.loads(json_data)
                except json.JSONDecodeError:
                    pass
            if isinstance(json_data, list):
                json_data = self._annotate_json_image_paths(json_data, image_path_map)
            with open(json_file, "w", encoding="utf-8") as f:
                if isinstance(json_data, (dict, list)):
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                else:
                    f.write(str(json_data))
        except Exception as e:
            logger.warning("Failed to save JSON: %s", e)
            traceback.print_exc()

        # Markdown — crop image regions and replace bbox tags with file paths
        if self.markdown_result and self.markdown_result.strip():
            md_text = self.markdown_result
            if self.original_images:
                try:
                    imgs_dir = output_path / "imgs"
                    md_text, _ = crop_and_replace_images(
                        md_text,
                        self.original_images,
                        imgs_dir,
                        image_prefix="cropped",
                    )
                except Exception as e:
                    logger.warning("Failed to process image regions: %s", e)
            md_file = output_path / f"{base_name}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict of the result.

        Useful for agents and programmatic consumers that need a structured
        representation without touching the file system.
        """
        d: dict = {
            "json_result": self.json_result,
            "markdown_result": self.markdown_result or "",
            "original_images": self.original_images,
        }
        # Include optional metadata set by MaaS mode.
        for attr in ("_usage", "_data_info", "_error"):
            val = getattr(self, attr, None)
            if val is not None:
                d[attr.lstrip("_")] = val
        return d

    def to_json(self, **kwargs: Any) -> str:
        """Serialise the result to a JSON string.

        Keyword arguments are forwarded to :func:`json.dumps`.
        """
        kwargs.setdefault("ensure_ascii", False)
        kwargs.setdefault("indent", 2)
        return json.dumps(self.to_dict(), **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(images={len(self.original_images)})"
