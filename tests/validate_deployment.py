import logging
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.append("/home/gaurav/small-idea")
from lmfast.deployment import BrowserExporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestBrowserExport(unittest.TestCase):
    def setUp(self):
        self.output_dir = "./test_browser_export"
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    def tearDown(self):
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)

    @patch("lmfast.deployment.browser.BrowserExporter._export_onnx")
    def test_export_onnx_structure(self, mock_export):
        logger.info("Testing Browser Export Structure (ONNX)...")

        # Mock the internal export function to avoid actual heavy model conversion
        mock_export.return_value = {
            "onnx_model": Path(f"{self.output_dir}/onnx/model.onnx"),
            "tokenizer": Path(f"{self.output_dir}/onnx/tokenizer.json"),
            "config": Path(f"{self.output_dir}/web_config.json"),
        }

        exporter = BrowserExporter("dummy/model-path", target="onnx")

        # We need to bypass validation of model path existence for this test
        exporter.model_path = Path("dummy/model-path")

        artifacts = exporter.export(self.output_dir, create_demo=True)

        # Check if demo was created
        self.assertTrue(os.path.exists(f"{self.output_dir}/demo/index.html"))
        self.assertTrue(os.path.exists(f"{self.output_dir}/export_config.json"))

        logger.info(f"Artifacts generated: {artifacts.keys()}")
        self.assertIn("demo", artifacts)

    @patch("lmfast.deployment.browser.BrowserExporter._export_webllm")
    def test_export_webllm_structure(self, mock_export):
        logger.info("Testing Browser Export Structure (WebLLM)...")

        mock_export.return_value = {
            "config": Path(f"{self.output_dir}/mlc/mlc-chat-config.json"),
            "readme": Path(f"{self.output_dir}/mlc/README.md"),
        }

        exporter = BrowserExporter("dummy/model-path", target="webllm")
        exporter.model_path = Path("dummy/model-path")

        artifacts = exporter.export(self.output_dir, create_demo=True)

        self.assertTrue(os.path.exists(f"{self.output_dir}/demo/index.html"))
        self.assertIn("demo", artifacts)


if __name__ == "__main__":
    unittest.main()
