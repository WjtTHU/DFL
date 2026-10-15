import torch
from loguru import logger


def cleanup():
    import gc

    gc.collect()

    try:
        for obj in gc.get_objects():
            if hasattr(obj, "__class__") and "Pool" in obj.__class__.__name__:
                if hasattr(obj, "close"):
                    obj.close()
                if hasattr(obj, "join"):
                    obj.join()
    except Exception as mp_cleanup_error:
        logger.warning(f"Error during multiprocessing cleanup: {mp_cleanup_error}")

    try:
        # Clear any cached tensor operations
        if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
            torch._C._cuda_clearCublasWorkspaces()
    except Exception as torch_cleanup_error:
        logger.warning(f"Error during additional torch cleanup: {torch_cleanup_error}")
