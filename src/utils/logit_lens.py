import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
from tabpfn.utils import translate_probs_across_borders
import torch.nn as nn
from typing import List, Tuple, Dict
from functools import partial
import matplotlib.pyplot as plt
import os
import shutil

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_dummy_dataset(weights: List[float], num_samples: int = 100, bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dummy dataset with given weights"""
    X = np.random.randn(num_samples, len(weights))
    y = X @ weights
    if bias:
        y += np.random.randn(num_samples)
    return X, y

def create_datasets(weights_list: List[List[float]], num_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Create a list of datasets with given weights"""
    datasets = []
    for weights in weights_list:
        X, y = create_dummy_dataset(weights, num_samples)
        datasets.append((X, y))
    return datasets

def logit_lens_accuracy(datasets: List[Tuple[np.ndarray, np.ndarray]]):
    """Test the accuracy of the model on the datasets"""
    for i, (X, y) in enumerate(datasets):
        print(f"Processing dataset {i+1}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)
        regressor = TabPFNRegressor(n_estimators=1)
        regressor.fit(X_train, y_train)
        (_,_,borders)=regressor.forward(X_test,use_inference_mode=True)
        predictions = regressor.predict(X_test)
        logit_lens(X_test,y_test,regressor,borders,predictions,i)
        print("--------------------------------")


def logit_lens(X_test: np.ndarray,y_test: np.ndarray, regressor: TabPFNRegressor,borders: List[np.ndarray],predictions: Dict[str, np.ndarray],dataset_index: int):
    """Test the accuracy of the model on the datasets"""
    predictions = {}
    def get_activation(name):
        def hook(model, input, output):
            layer_out = output[0].detach()[-len(X_test):]
            outputs=regressor.model_.decoder_dict['standard'](layer_out)
            outputs=outputs[:,-1,:].reshape(1,len(X_test),5000)
            """
            transformed_logits = [
            translate_probs_across_borders(
                logits,
                frm=torch.as_tensor(borders_t, device=logits.device),
                to=regressor.bardist_.borders.to(logits.device),
            )
            for logits, borders_t in zip(outputs, borders)
            ]
            """
            logits = outputs.softmax(dim=-1)
            """
            stacked_logits = torch.stack(transformed_logits, dim=0)
            #print(stacked_logits.shape)
            if regressor.average_before_softmax:
                logits = stacked_logits.log().mean(dim=0).softmax(dim=-1)
            else:
                logits = stacked_logits.mean(dim=0) 
            #print(logits.shape)
            # Post-process the logits
            """
            logits = logits.log()
            if logits.dtype == torch.float16:
                logits = logits.float()
            predictions[name] = regressor.normalized_bardist_.mean(logits)
        return hook

    hook_handles = []
    for i, layer in enumerate(regressor.model_.transformer_encoder.layers):
            handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
            hook_handles.append(handle)
        
    # Forward pass to extract activations
    with torch.no_grad():
        _ = regressor.predict(X_test)
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    logit_lens_list=[]
    for _, value in predictions.items():
        logit_lens_list.append(value[0].detach().cpu().numpy())
    y_test_list=[y_test[0]]*len(logit_lens_list)
    make_graph(logit_lens_list, y_test_list, dataset_index=dataset_index, title=f"Dataset {dataset_index+1} predictions vs target")


def make_graph(logit_lens_list: List[float], y_test_list: List[float], dataset_index: int, title: str = "",
               save_dir: str = "logit_lens", show: bool = False) -> None:
    """Plot prediction and target across layers as line graph.

    x-axis: layer index (0..num_layers-1)
    y-axis: answer value
    """
    os.makedirs(save_dir, exist_ok=True)
    num_layers = len(logit_lens_list)
    x_values = list(range(num_layers))

    plt.figure(figsize=(7, 4))
    plt.plot(x_values, logit_lens_list, label="Prediction", marker='o')
    plt.plot(x_values, y_test_list, label="Target", linestyle='--', marker='x')
    plt.xlabel("Layer")
    plt.ylabel("Answer")
    if title:
        plt.title(title)
    else:
        plt.title("Predictions and Target across Layers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"dataset_{dataset_index+1}.png")
    plt.savefig(out_path, dpi=150)
    if show:
        plt.show()
    else:
        plt.close()




def main():
    """Main function"""
    # Clean output directory at start
    out_dir = "logit_lens"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    weights_list = [[1.0, 1.0],[2.0,2.0],[3.0,3.0],[4.0,4.0],[5.0,5.0],[6.0,6.0],[7.0,7.0],[8.0,8.0],[9.0,9.0],[10.0,10.0]]
    datasets = create_datasets(weights_list)
    logit_lens_accuracy(datasets)

if __name__ == "__main__":
    main()
