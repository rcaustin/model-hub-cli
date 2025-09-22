from src.Interfaces import ModelData
from src.Metric import Metric
from loguru import logger


class SizeMetric(Metric):
    """
    SizeMetric evaluates model compatibility across different hardware devices
    based on model size and available memory.

    Scoring System:
    Score = min(1.0, (Usable Device Memory - Model Size) / Usable Device Memory)
    All negative scores become 0. All scores are capped at 1.0.
    Score range for each device: 0.0 - 1.0

    Device Memory Calculations:
    - Raspberry Pi 5: 16GB RAM * 0.125 penalty = 2GB usable (CPU-only, inefficient)
    - Jetson Nano: 4GB RAM * 0.75 penalty = 3GB usable (GPU available but limited)
    - Desktop PC (RTX 4090): 24GB VRAM - 4GB overhead = 20GB usable (efficient GPU)
    - AWS Server (g4dn.12xlarge): 64GB VRAM - 4GB overhead = 60GB usable (multi-GPU)

    Justification:
    - Memory penalties reflect computational efficiency differences between devices
    - GPU devices (Desktop/AWS) are most efficient, followed by Jetson (limited GPU),
      then Raspberry Pi (CPU-only)
    - Fixed overhead accounts for OS and model loading requirements on GPU devices
    - Percentage penalties for CPU devices capture both memory constraints and
      computational inefficiency

    Model Size Calculation:
    Model Size = Number of Parameters * Average Bytes per Parameter
    Uses Hugging Face API to get parameter count and tensor types.
    Assumes even split if multiple tensor types are present.
    """

    # Device specifications with usable memory (after overhead and penalties)
    DEVICE_SPECS = {
        "raspberry_pi": 2.0,  # 16GB * 0.125 penalty = 2GB usable
        "jetson_nano": 3.0,   # 4GB * 0.75 penalty = 3GB usable  
        "desktop_pc": 20.0,   # 24GB - 4GB overhead = 20GB usable
        "aws_server": 60.0    # 64GB - 4GB overhead = 60GB usable
    }

    # Tensor type sizes in bytes per parameter
    TENSOR_SIZES = {
        "float32": 4,
        "float16": 2,
        "int8": 1,
        "int4": 0.5
    }

    def evaluate(self, model: ModelData) -> dict[str, float]:
        """
        Evaluate model size compatibility across different devices.
        Returns a dictionary with device names as keys and scores as values.
        """
        try:
            # Get model size in GB
            model_size_gb = self._get_model_size(model)

            if model_size_gb is None:
                logger.warning(f"Could not determine model size for {model.modelLink}")
                return {device: 0.0 for device in self.DEVICE_SPECS.keys()}

            # Calculate score for each device
            scores = {}
            for device, usable_memory in self.DEVICE_SPECS.items():
                score = (usable_memory - model_size_gb) / usable_memory
                score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
                scores[device] = score

            return scores

        except Exception as e:
            logger.error(f"Error evaluating size metric: {e}")
            return {device: 0.0 for device in self.DEVICE_SPECS.keys()}

    def _get_model_size(self, model: ModelData) -> float:
        """
        Get model size in GB using model's hf_metadata property.
        Returns None if size cannot be determined.
        """
        try:
            # Use the model's hf_metadata property (which handles fetching automatically)
            if not model.hf_metadata:
                logger.warning(f"No Hugging Face metadata available for {model.modelLink}")
                return None

            metadata = model.hf_metadata

            # Try to get parameter count from metadata
            param_count = None

            # Common locations for parameter count in HF metadata
            if "config" in metadata:
                config = metadata["config"]
                # Check various parameter count fields
                for field in ["num_parameters", "n_parameters", "total_params"]:
                    if field in config:
                        param_count = config[field]
                        break

            if param_count is None:
                logger.warning(f"Could not find parameter count in metadata for {model.modelLink}")
                return None

            # Determine tensor types (assume float16 if not specified)
            tensor_types = self._get_tensor_types(metadata)
            avg_bytes_per_param = self._calculate_avg_bytes_per_param(tensor_types)

            # Calculate size in GB
            size_bytes = param_count * avg_bytes_per_param
            size_gb = size_bytes / (1024 ** 3)  # Convert to GB

            return size_gb

        except Exception as e:
            logger.error(f"Error getting model size: {e}")
            return None

    def _get_tensor_types(self, metadata: dict) -> list:
        """Extract tensor types from metadata."""
        # Default to float16 if not specified
        default_types = ["float16"]

        try:
            if "config" in metadata:
                config = metadata["config"]
                if "torch_dtype" in config:
                    return [config["torch_dtype"]]
                if "dtype" in config:
                    return [config["dtype"]]

            return default_types

        except Exception:
            return default_types

    def _calculate_avg_bytes_per_param(self, tensor_types: list) -> float:
        """Calculate average bytes per parameter for given tensor types."""
        if not tensor_types:
            return self.TENSOR_SIZES["float16"]  # Default

        total_bytes = 0
        valid_types = 0

        for tensor_type in tensor_types:
            if tensor_type in self.TENSOR_SIZES:
                total_bytes += self.TENSOR_SIZES[tensor_type]
                valid_types += 1

        if valid_types == 0:
            return self.TENSOR_SIZES["float16"]  # Default

        return total_bytes / valid_types
