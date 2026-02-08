from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import atexit


class ModelInspector:
    """Context manager for inspecting model architecture and modules.

    Usage:
        with ModelInspector("my_model") as inspector:
            for name, module in model.named_modules():
                inspector.record_module(name, module)

        # Or use convenience function:
        inspector = inspect_model(model, "tabpfn_model")
    """

    _active_inspectors: Dict[str, "ModelInspector"] = {}

    def __init__(self, name: str, max_depth: Optional[int] = None):
        """Initialize the inspector.

        Args:
            name: Identifier for this inspection
            max_depth: Maximum depth to record (None for unlimited)
        """
        self.name = name
        self.modules: List[Dict[str, Any]] = []
        self.max_depth = max_depth
        self._active_inspectors[name] = self

    def record_module(self, name: str, module: Any) -> None:
        """Record a module's name and type information.

        Args:
            name: Full module name (e.g., 'transformer_encoder.layers.0.self_attn')
            module: The module object
        """
        depth = name.count(".") if name else 0

        if self.max_depth is not None and depth > self.max_depth:
            return

        module_info = {
            "name": name,
            "type": type(module).__name__,
            "module_type": type(module).__module__,
            "depth": depth,
            "is_hookable": self._is_hookable(name, type(module).__name__),
        }
        self.modules.append(module_info)

    def _is_hookable(self, name: str, type_name: str) -> bool:
        """Check if a module is suitable for hooking."""
        hookable_keywords = ["attn", "attention", "mlp", "ffn", "linear", "conv"]
        combined = f"{name.lower()} {type_name.lower()}"
        return any(keyword in combined for keyword in hookable_keywords)

    def __enter__(self) -> "ModelInspector":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - print summary."""
        self._print_summary()

    def _print_summary(self) -> None:
        """Print a comprehensive summary of the model structure."""
        if not self.modules:
            print(f"\n[ModelInspector: {self.name}] No modules recorded")
            return

        print(f"\n{'=' * 70}")
        print(f"[ModelInspector: {self.name}]")
        print(f"{'=' * 70}")

        # 1. Tree Visualization
        print("\n📊 Model Architecture Tree:")
        print("-" * 50)
        self._print_tree()

        # 2. Module Type Statistics
        print("\n📈 Module Type Statistics:")
        print("-" * 50)
        self._print_type_statistics()

        # 3. Hookable Modules
        print("\n🔌 Hookable Modules (for activation patching):")
        print("-" * 50)
        self._print_hookable_modules()

        # 4. Depth Distribution
        print("\n📏 Depth Distribution:")
        print("-" * 50)
        self._print_depth_distribution()

        print(f"\n{'=' * 70}\n")

    def _print_tree(self) -> None:
        """Print ASCII tree of model structure."""
        if not self.modules:
            return

        # Build tree structure
        tree: Dict[str, Any] = {}
        for mod in self.modules:
            parts = mod["name"].split(".") if mod["name"] else ["(root)"]
            current = tree
            for i, part in enumerate(parts):
                if part not in current:
                    current[part] = {"_children": {}, "_info": None}
                if i == len(parts) - 1:
                    current[part]["_info"] = mod
                current = current[part]["_children"]

        # Print tree
        def print_node(node_dict: Dict, prefix: str = "", is_last: bool = True) -> None:
            items = list(node_dict.items())
            for i, (name, data) in enumerate(items):
                is_last_item = i == len(items) - 1
                connector = "└── " if is_last_item else "├── "

                if data["_info"]:
                    type_name = data["_info"]["type"]
                    hook_marker = " 🔌" if data["_info"]["is_hookable"] else ""
                    print(f"{prefix}{connector}{name}: {type_name}{hook_marker}")
                else:
                    print(f"{prefix}{connector}{name}")

                if data["_children"]:
                    extension = "    " if is_last_item else "│   "
                    print_node(data["_children"], prefix + extension, is_last_item)

        print_node(tree)

    def _print_type_statistics(self) -> None:
        """Print statistics about module types."""
        type_counts: Dict[str, int] = defaultdict(int)
        type_examples: Dict[str, List[str]] = defaultdict(list)

        for mod in self.modules:
            type_name = mod["type"]
            type_counts[type_name] += 1
            if len(type_examples[type_name]) < 3:
                type_examples[type_name].append(mod["name"] or "(root)")

        sorted_types = sorted(type_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"{'Type':<30} {'Count':<8} {'Examples'}")
        print("-" * 70)
        for type_name, count in sorted_types[:10]:  # Top 10
            examples = ", ".join(type_examples[type_name])
            if len(examples) > 35:
                examples = examples[:32] + "..."
            print(f"{type_name:<30} {count:<8} {examples}")

        if len(sorted_types) > 10:
            print(f"... and {len(sorted_types) - 10} more types")

    def _print_hookable_modules(self) -> None:
        """Print list of hookable modules."""
        hookable = [mod for mod in self.modules if mod["is_hookable"]]

        if not hookable:
            print("No hookable modules found")
            return

        # Group by type
        by_type: Dict[str, List[str]] = defaultdict(list)
        for mod in hookable:
            by_type[mod["type"]].append(mod["name"])

        for type_name, names in sorted(by_type.items()):
            print(f"\n{type_name}:")
            for name in names[:5]:  # Show first 5
                print(f"  • {name}")
            if len(names) > 5:
                print(f"  ... and {len(names) - 5} more")

    def _print_depth_distribution(self) -> None:
        """Print distribution of modules by depth."""
        depth_counts: Dict[int, int] = defaultdict(int)
        for mod in self.modules:
            depth_counts[mod["depth"]] += 1

        max_depth = max(depth_counts.keys()) if depth_counts else 0
        total = len(self.modules)

        for depth in range(max_depth + 1):
            count = depth_counts.get(depth, 0)
            percentage = (count / total * 100) if total > 0 else 0
            bar = "█" * int(percentage / 5)
            print(f"Depth {depth:<2}: {count:>4} modules ({percentage:>5.1f}%) {bar}")

    def get_hookable_modules(self) -> List[Tuple[str, str]]:
        """Get list of hookable module names and types.

        Returns:
            List of (name, type) tuples for hookable modules
        """
        return [
            (mod["name"], mod["type"]) for mod in self.modules if mod["is_hookable"]
        ]

    def get_modules_by_type(self, type_name: str) -> List[str]:
        """Get all module names of a specific type.

        Args:
            type_name: Module type to search for

        Returns:
            List of module names
        """
        return [mod["name"] for mod in self.modules if mod["type"] == type_name]


def inspect_model(
    model: Any, name: str = "model", max_depth: Optional[int] = None
) -> ModelInspector:
    """Quick model inspection without context manager.

    Args:
        model: The model to inspect
        name: Identifier for this inspection
        max_depth: Maximum depth to record

    Returns:
        ModelInspector instance with recorded modules

    Example:
        >>> inspector = inspect_model(model, "tabpfn")
        >>> hookable = inspector.get_hookable_modules()
    """
    inspector = ModelInspector(name, max_depth)
    for mod_name, module in model.named_modules():
        inspector.record_module(mod_name, module)
    inspector._print_summary()
    return inspector


def quick_inspect(model: Any, max_depth: int = 3) -> None:
    """Quick inspection - just print the tree.

    Args:
        model: The model to inspect
        max_depth: Maximum depth to show
    """
    inspector = ModelInspector("quick_inspect", max_depth)
    for name, module in model.named_modules():
        inspector.record_module(name, module)
    inspector._print_summary()


# Register cleanup function to print all inspectors at program exit
def _print_all_inspectors():
    """Print any remaining inspectors at exit."""
    for name, inspector in ModelInspector._active_inspectors.items():
        if inspector.modules:
            inspector._print_summary()


atexit.register(_print_all_inspectors)
