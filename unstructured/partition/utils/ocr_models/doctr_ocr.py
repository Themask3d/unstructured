from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL import Image as PILImage
    from unstructured_inference.inference.elements import TextRegion
    from unstructured_inference.inference.layoutelement import LayoutElement


from unstructured.logger import logger
from unstructured.partition.utils.constants import Source
from unstructured.partition.utils.ocr_models.ocr_interface import OCRAgent
from unstructured.utils import requires_dependencies


class OCRAgentDocTR(OCRAgent):
    """OCR service implementation for docTR."""

    def __init__(self, language: str = "en"):
        # NOTE(faical): docTR's language support is based on the model, not an argument.
        self.agent = self.load_agent()

    @staticmethod
    @requires_dependencies(["doctr"], "doctr")
    def load_agent():
        """Loads the docTR agent."""
        from doctr.models import ocr_predictor
        import torch

        logger.info("Loading doctr model...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # NOTE(faical): As per the user's request, instantiate the predictor with a specific
        # detection and recognition architecture.
        predictor = ocr_predictor(
            det_arch="db_resnet50",
            reco_arch="crnn_vgg16_bn",
            pretrained=True,
        ).to(device)

        return predictor

    def get_text_from_image(self, image: PILImage.Image) -> str:
        """Gets the text from an image using docTR."""
        image_np = np.array(image)
        result = self.agent([image_np])
        return result.render()

    def is_text_sorted(self) -> bool:
        return False

    @requires_dependencies(["doctr", "unstructured_inference"])
    def get_layout_from_image(self, image: PILImage.Image) -> list[TextRegion]:
        """Get the OCR regions from an image as a list of text regions with docTR."""
        image_np = np.array(image)
        result = self.agent([image_np])

        image_height, image_width = image.size[1], image.size[0]
        return self.parse_data(result, image_width=image_width, image_height=image_height)

    @requires_dependencies("unstructured_inference")
    def get_layout_elements_from_image(self, image: PILImage.Image) -> list[LayoutElement]:
        from unstructured.partition.pdf_image.inference_utils import (
            build_layout_elements_from_ocr_regions,
        )

        ocr_regions = self.get_layout_from_image(image)
        # NOTE(faical): docTR's output is not guaranteed to be sorted,
        # so we don't group by ocr_text.
        return build_layout_elements_from_ocr_regions(
            ocr_regions=ocr_regions,
        )

    @requires_dependencies("unstructured_inference")
    def parse_data(self, result, image_width: int, image_height: int) -> list[TextRegion]:
        """Parse the docTR result to extract a list of TextRegion objects."""
        from unstructured.partition.pdf_image.inference_utils import build_text_region_from_coords

        text_regions = []
        if not result or not result.pages:
            return []

        page = result.pages[0]
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    if word.value:
                        (x1, y1), (x2, y2) = word.geometry

                        abs_x1 = x1 * image_width
                        abs_y1 = y1 * image_height
                        abs_x2 = x2 * image_width
                        abs_y2 = y2 * image_height

                        text_region = build_text_region_from_coords(
                            abs_x1,
                            abs_y1,
                            abs_x2,
                            abs_y2,
                            text=word.value,
                            source=Source.OCR_DOCTR,
                        )
                        text_regions.append(text_region)
        return text_regions
