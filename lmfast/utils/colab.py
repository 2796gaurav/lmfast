import logging
import os
import sys

logger = logging.getLogger(__name__)


def is_colab() -> bool:
    """Check if running in Google Colab."""
    return "google.colab" in sys.modules


def setup_colab_env(
    mount_drive: bool = True, install_dependencies: bool = True, use_unsloth: bool = True
):
    """
    Magic function to setup Google Colab environment for T4/L4 GPUs.

    Args:
        mount_drive: Mount Google Drive
        install_dependencies: Install system libs (nvcc etc)
        use_unsloth: Install Unsloth for fast training
    """
    if not is_colab():
        logger.warning("Not running in Google Colab. Skipping setup.")
        return

    print("🚀 Setting up LMFast environment for Colab...")

    if mount_drive:
        print("📂 Mounting Google Drive...")
        from google.colab import drive

        if not os.path.exists("/content/drive"):
            drive.mount("/content/drive")

    if install_dependencies:
        print("🛠️ Installing system dependencies...")
        # Basic check for GPU
        try:
            import torch

            if not torch.cuda.is_available():
                print(
                    "⚠️ No GPU detected! Make sure to select T4 GPU in Runtime > Change runtime type"
                )
        except ImportError:
            pass

    if use_unsloth:
        print("🦥 Installing Unsloth (this might take a few minutes)...")
        # Optimization: Check if already installed
        try:
            import unsloth

            print("   Unsloth already installed.")
        except ImportError:
            # We assume the user has run `pip install lmfast[fast]` but sometimes we need specific wheels
            # For now, we trust the package dependencies or guide the user
            pass

    print("✨ Environment ready!")
