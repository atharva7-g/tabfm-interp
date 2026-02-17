from typing import Any
from collections import defaultdict
import atexit


class ShapeInspector:
    """Context manager for inspecting tensor shapes, dtypes, and devices during execution.

    Usage:
        with ShapeInspector("hook_name") as inspector:
            def hook(...):
                inspector.record(tensor)
    """

    # Class-level registry to track all active inspectors
    _active_inspectors = {}

    def __init__(self, name: str):
        self.name = name
        self.tensor_info = []
        self.call_count = 0
        self._active_inspectors[name] = self

    def __enter__(self):
        return self

    def record(self, tensor: Any) -> None:
        """Record tensor shape, dtype, and device information."""
        if hasattr(tensor, "shape") and hasattr(tensor, "dtype"):
            info = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device) if hasattr(tensor, "device") else "N/A",
            }
            self.tensor_info.append(info)
            self.call_count += 1
        else:
            # Handle non-tensor objects
            self.tensor_info.append(
                {
                    "shape": "N/A",
                    "dtype": "N/A",
                    "device": "N/A",
                    "type": type(tensor).__name__,
                }
            )
            self.call_count += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.call_count == 0:
            print(f"[ShapeInspector: {self.name}] No tensors recorded")
            return

        # Group by unique combinations
        unique_info = defaultdict(int)
        for info in self.tensor_info:
            if info["shape"] != "N/A":
                key = (info["shape"], info["dtype"], info["device"])
                unique_info[key] += 1
            else:
                key = (info["type"],)
                unique_info[key] += 1

        print(f"\n[ShapeInspector: {self.name}]")
        print(f"  Total calls: {self.call_count}")
        print(f"  Unique tensor configurations: {len(unique_info)}")
        print("  Details:")
        for (shape, dtype, device), count in unique_info.items():
            if shape != "N/A":
                print(
                    f"    Shape: {shape}, Dtype: {dtype}, Device: {device} (count: {count})"
                )
            else:
                print(f"    Type: {shape} (count: {count})")


# Register cleanup function to print all inspectors at program exit
def _print_all_inspectors():
    for name, inspector in ShapeInspector._active_inspectors.items():
        if inspector.call_count > 0:
            # Reuse the __exit__ logic but without the "no tensors" message
            unique_info = defaultdict(int)
            for info in inspector.tensor_info:
                if info["shape"] != "N/A":
                    key = (info["shape"], info["dtype"], info["device"])
                    unique_info[key] += 1
                else:
                    key = (info["type"],)
                    unique_info[key] += 1

            print(f"\n[ShapeInspector: {inspector.name}]")
            print(f"  Total calls: {inspector.call_count}")
            print(f"  Unique tensor configurations: {len(unique_info)}")
            print("  Details:")
            for (shape, dtype, device), count in unique_info.items():
                if shape != "N/A":
                    print(
                        f"    Shape: {shape}, Dtype: {dtype}, Device: {device} (count: {count})"
                    )
                else:
                    print(f"    Type: {shape} (count: {count})")


atexit.register(_print_all_inspectors)
