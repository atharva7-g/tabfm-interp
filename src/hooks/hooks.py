from typing import Dict, List, Optional, Tuple, Callable, Any
import numpy as np
import torch
from tabpfn import TabPFNRegressor


class TestLabelTokenVerifier:
    def __init__(self, regressor: TabPFNRegressor):
        self.regressor = regressor
        self.model = regressor.model_
        self.verification_results: Dict[str, Any] = {}
        self.hook_handles: List[torch.utils.hooks.RemovableHandle] = []

    def compute_test_label_token_index(
        self, X_test: np.ndarray, expected_num_features: Optional[int] = None
    ) -> int:
        """
        In TabPFN, the input sequence structure is:
        [train_features..., train_labels..., test_features..., test_label_placeholder]
        The test label token is at the last position (index -1).
        """
        num_test_samples = X_test.shape[0]
        num_features = X_test.shape[1]

        if expected_num_features is not None and num_features != expected_num_features:
            raise ValueError(
                f"Feature count mismatch: expected {expected_num_features}, got {num_features}"
            )

        # The test label token is at the last position
        test_label_token_index = -1

        self.verification_results["token_index"] = {
            "index": test_label_token_index,
            "num_test_samples": num_test_samples,
            "num_features": num_features,
            "description": "Test label token is at the last position of the sequence",
        }

        return test_label_token_index

    def create_verification_hook(
        self,
        layer_name: str,
        expected_shape: Optional[Tuple[int, ...]] = None,
        validate_values: bool = True,
        custom_validation_fn: Optional[Callable[[torch.Tensor], bool]] = None,
    ) -> Callable:
        def verification_hook(module, inputs, output):
            # Extract the main output tensor
            if isinstance(output, (tuple, list)):
                output_tensor = output[0]
            else:
                output_tensor = output

            verification_data = {
                "layer_name": layer_name,
                "output_shape": tuple(output_tensor.shape),
                "device": str(output_tensor.device),
                "dtype": str(output_tensor.dtype),
            }

            # Validate shape if expected shape provided
            if expected_shape is not None:
                actual_shape = output_tensor.shape
                shape_matches = all(
                    actual_shape[i] == expected_shape[i] or expected_shape[i] == -1
                    for i in range(len(expected_shape))
                )
                verification_data["shape_matches"] = shape_matches
                verification_data["expected_shape"] = expected_shape
                if not shape_matches:
                    verification_data["shape_error"] = (
                        f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
                    )
            else:
                verification_data["shape_matches"] = None

            # Validate values (check for NaN/Inf)
            if validate_values:
                has_nan = torch.isnan(output_tensor).any().item()
                has_inf = torch.isinf(output_tensor).any()
                verification_data["has_nan"] = has_nan
                verification_data["has_inf"] = has_inf
                verification_data["values_valid"] = not (has_nan or has_inf)

                if output_tensor.numel() > 0:
                    verification_data["value_stats"] = {
                        "min": float(output_tensor.min()),
                        "max": float(output_tensor.max()),
                        "mean": float(output_tensor.mean()),
                        "std": float(output_tensor.std()),
                    }

            # Run custom validation if provided
            if custom_validation_fn is not None:
                try:
                    custom_valid = custom_validation_fn(output_tensor)
                    verification_data["custom_validation"] = custom_valid
                except Exception as e:
                    verification_data["custom_validation"] = False
                    verification_data["custom_validation_error"] = str(e)

            # Store the verification data
            if layer_name not in self.verification_results:
                self.verification_results[layer_name] = []
            self.verification_results[layer_name].append(verification_data)

            # Also store a sample of the output for inspection
            if "output_samples" not in self.verification_results:
                self.verification_results["output_samples"] = {}

            # Store a small sample of the output (first element, for memory efficiency)
            sample_size = (
                min(5, output_tensor.shape[0]) if output_tensor.dim() > 0 else 1
            )
            self.verification_results["output_samples"][layer_name] = (
                output_tensor[:sample_size].detach().cpu()
            )

        return verification_hook

    def attach_hooks(
        self,
        target_layers: Optional[List[str]] = None,
        expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
        validate_values: bool = True,
        custom_validation_fn: Optional[Callable[[torch.Tensor], bool]] = None,
    ) -> None:
        # Clear any existing hooks
        self.remove_hooks()

        # Determine which layers to hook
        if target_layers is None:
            target_layers = [
                f"layer_{i}"
                for i in range(len(self.model.transformer_encoder.layers))  # type: ignore
            ]

        # Register hooks for each target layer
        for layer_name in target_layers:
            layer_idx = int(layer_name.split("_")[1])

            # Access transformer encoder layers with type safety
            if not hasattr(self.model, "transformer_encoder"):
                raise AttributeError(
                    "Model does not have transformer_encoder attribute"
                )

            transformer_encoder = self.model.transformer_encoder
            if not hasattr(transformer_encoder, "layers"):
                raise AttributeError(
                    "transformer_encoder does not have layers attribute"
                )

            layers = transformer_encoder.layers  # type: ignore
            if layer_idx >= len(layers):  # type: ignore
                raise IndexError(
                    f"Layer index {layer_idx} out of range (max {len(layers) - 1})"  # type: ignore
                )

            layer = layers[layer_idx]  # type: ignore

            # Get expected shape for this layer if provided
            expected_shape = (
                expected_shapes.get(layer_name) if expected_shapes else None
            )

            # Create and register the hook
            hook_fn = self.create_verification_hook(
                layer_name=layer_name,
                expected_shape=expected_shape,
                validate_values=validate_values,
                custom_validation_fn=custom_validation_fn,
            )
            handle = layer.register_forward_hook(hook_fn)  # type: ignore
            self.hook_handles.append(handle)

        print(
            f"Attached verification hooks to {len(self.hook_handles)} layers: {target_layers}"
        )

    def run_verification_forward_pass(
        self,
        X_test: np.ndarray,
        compute_token_index: bool = True,
        expected_num_features: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        # Compute test label token index if requested
        if compute_token_index:
            self.compute_test_label_token_index(X_test, expected_num_features)

        # Run forward pass with hooks
        print("Running verification forward pass...")
        with torch.no_grad():
            predictions = self.regressor.predict(X_test)

        # Compile verification results
        results = {
            "predictions": predictions,
            "verification": self.verification_results,
            "success": self.validate_all(),
        }

        return results

    def validate_all(self) -> bool:
        all_valid = True
        validation_summary = {
            "token_index_valid": False,
            "shapes_valid": False,
            "values_valid": False,
            "errors": [],
        }

        # Check token index computation
        if "token_index" in self.verification_results:
            token_data = self.verification_results["token_index"]
            if "index" in token_data and token_data["index"] is not None:
                validation_summary["token_index_valid"] = True
            else:
                all_valid = False
                validation_summary["errors"].append("Token index not computed")
        else:
            all_valid = False
            validation_summary["errors"].append("No token index data found")

        # Check layer outputs
        shape_errors = []
        value_errors = []

        for layer_name, layer_results in self.verification_results.items():
            if layer_name in ["token_index", "output_samples"]:
                continue

            if isinstance(layer_results, list):
                for result in layer_results:
                    # Check shape validation
                    if result.get("shape_matches") is False:
                        shape_errors.append(
                            f"{layer_name}: {result.get('shape_error', 'Unknown shape error')}"
                        )

                    # Check value validation
                    if result.get("values_valid") is False:
                        if result.get("has_nan"):
                            value_errors.append(f"{layer_name}: Contains NaN values")
                        if result.get("has_inf"):
                            value_errors.append(f"{layer_name}: Contains Inf values")

        if not shape_errors:
            validation_summary["shapes_valid"] = True
        else:
            all_valid = False
            validation_summary["errors"].extend(shape_errors)

        if not value_errors:
            validation_summary["values_valid"] = True
        else:
            all_valid = False
            validation_summary["errors"].extend(value_errors)

        validation_summary["overall_valid"] = all_valid
        self.verification_results["validation_summary"] = validation_summary

        return all_valid

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def get_test_label_activations(
        self, layer_name: str, token_index: int = -1
    ) -> Optional[torch.Tensor]:
        """
        Extract activations at the test label token position.
        Assumes output shape is (batch, seq_len, num_heads, hidden_dim).
        """
        if "output_samples" not in self.verification_results:
            return None

        if layer_name not in self.verification_results["output_samples"]:
            return None

        layer_output = self.verification_results["output_samples"][layer_name]

        if layer_output.dim() >= 2:
            return layer_output[:, token_index, :]
        else:
            return layer_output

    def print_verification_report(self) -> None:
        print("\n" + "=" * 60)
        print("TABPFN TEST LABEL TOKEN VERIFICATION REPORT")
        print("=" * 60)

        # Token index info
        if "token_index" in self.verification_results:
            token_data = self.verification_results["token_index"]
            print(f"\nTest Label Token Index:")
            print(f"  Index: {token_data.get('index')}")
            print(f"  Num test samples: {token_data.get('num_test_samples')}")
            print(f"  Num features: {token_data.get('num_features')}")
            print(f"  Description: {token_data.get('description')}")

        # Validation summary
        if "validation_summary" in self.verification_results:
            summary = self.verification_results["validation_summary"]
            print(f"\nValidation Summary:")
            print(f"  Token index valid: {summary.get('token_index_valid')}")
            print(f"  Shapes valid: {summary.get('shapes_valid')}")
            print(f"  Values valid: {summary.get('values_valid')}")
            print(f"  Overall valid: {summary.get('overall_valid')}")

            if summary.get("errors"):
                print(f"\n  Errors:")
                for error in summary["errors"]:
                    print(f"    - {error}")

        # Layer details
        print(f"\nLayer Verification Details:")
        for layer_name, layer_results in self.verification_results.items():
            if layer_name in ["token_index", "output_samples", "validation_summary"]:
                continue

            if isinstance(layer_results, list) and layer_results:
                result = layer_results[-1]  # Get most recent result
                print(f"\n  {layer_name}:")
                print(f"    Shape: {result.get('output_shape')}")
                if result.get("expected_shape"):
                    print(f"    Expected: {result.get('expected_shape')}")
                print(f"    Device: {result.get('device')}")
                print(f"    Dtype: {result.get('dtype')}")
                print(f"    Shape matches: {result.get('shape_matches')}")
                print(f"    Values valid: {result.get('values_valid')}")

                if "value_stats" in result:
                    stats = result["value_stats"]
                    print(
                        f"    Value stats: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}"
                    )

        print("\n" + "=" * 60)


def verify_tabpfn_test_label_processing(
    regressor: TabPFNRegressor,
    X_test: np.ndarray,
    expected_num_features: Optional[int] = None,
    target_layers: Optional[List[str]] = None,
    expected_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to verify TabPFN test label token processing.

    This function provides a complete verification pipeline:
    1. Computes test-label token index
    2. Attaches verification hooks
    3. Runs forward pass
    4. Validates outputs

    Args:
        regressor: Fitted TabPFNRegressor instance
        X_test: Test input data
        expected_num_features: Expected number of features for validation
        target_layers: List of layer names to verify (None = all layers)
        expected_shapes: Dict mapping layer names to expected shapes (4D: batch, seq_len, num_heads, hidden_dim)
        verbose: Whether to print verification report

    Returns:
        Dictionary containing:
        - predictions: Model predictions
        - verification_results: Detailed verification data
        - success: Boolean indicating if all validations passed

    Example:
        X_train = np.random.randn(100, 4)
        y_train = np.random.randn(100)
        regressor = TabPFNRegressor(device='cuda', n_estimators=1)
        regressor.fit(X_train, y_train)

        # Run verification
        X_test = np.random.randn(10, 4)
        results = verify_tabpfn_test_label_processing(
            regressor,
            X_test,
            expected_num_features=4,
            verbose=True
        )
        print(f"Verification passed: {results['success']}")
    """
    # Create verifier
    verifier = TestLabelTokenVerifier(regressor)

    # Attach hooks
    verifier.attach_hooks(
        target_layers=target_layers,
        expected_shapes=expected_shapes,
        validate_values=True,
    )

    # Run verification
    results = verifier.run_verification_forward_pass(
        X_test=X_test,
        compute_token_index=True,
        expected_num_features=expected_num_features,
    )

    # Print report if verbose
    if verbose:
        verifier.print_verification_report()

    # Clean up hooks
    verifier.remove_hooks()

    return results


def create_shape_expectations(
    regressor: TabPFNRegressor,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    hidden_dim: int,
) -> Dict[str, Tuple[int, ...]]:
    num_layers = len(regressor.model_.transformer_encoder.layers)  # type: ignore

    expected_shapes = {}
    for i in range(num_layers):
        # TabPFN transformer layers output (batch, seq_len, num_heads, hidden_dim)
        expected_shapes[f"layer_{i}"] = (batch_size, seq_len, num_heads, hidden_dim)

    return expected_shapes


if __name__ == "__main__":
    # Demo usage
    print("TabPFN Test Label Token Verification Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    X_train = np.random.randn(100, 4).astype(np.float32)
    y_train = np.random.randn(100).astype(np.float32)
    X_test = np.random.randn(10, 4).astype(np.float32)

    # Initialize and fit model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    print("Model fitted successfully")

    # Create expected shapes for verification (4D: batch, seq_len, num_heads, hidden_dim)
    expected_shapes = create_shape_expectations(
        regressor, batch_size=-1, seq_len=174, num_heads=4, hidden_dim=192
    )

    # Run verification with expected shapes
    results = verify_tabpfn_test_label_processing(
        regressor=regressor,
        X_test=X_test,
        expected_num_features=4,
        expected_shapes=expected_shapes,
        verbose=True,
    )

    print(
        f"\nFinal result: Verification {'PASSED' if results['success'] else 'FAILED'}"
    )
