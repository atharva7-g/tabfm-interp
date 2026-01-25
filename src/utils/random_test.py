import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tabpfn import TabPFNRegressor

def print_correlations(model):
    decoder_layer=model.model_.decoder_dict['standard'][2]
    print(decoder_layer.weight.shape)
    m,_=decoder_layer.weight.shape
    cosine_similarities=[]
    for i in range(m):
        #cosine similarity between i-th column and 2500-th column
        #use torch.nn.functional.cosine_similarity
        cosine_similarities.append(F.cosine_similarity(decoder_layer.weight[i,:], decoder_layer.weight[4000,:], dim=0).item())
    
    # Plot bar graph of cosine similarities
    plt.figure(figsize=(12, 6))
    plt.bar(range(m), cosine_similarities)
    plt.xlabel('Row Index')
    plt.ylabel('Cosine Similarity')
    plt.title(f'Cosine Similarities between each row and row 4000')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cosine_similarities.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar graph saved as 'cosine_similarities.png'")


if __name__ == "__main__":
    regressor = TabPFNRegressor(n_estimators=1)
    
    # Create dummy data for fitting (TabPFN needs to be fitted to initialize model_)
    X_dummy = np.random.randn(4, 3)  # 4 samples, 3 features
    y_dummy = np.random.randn(4)      # 4 targets
    
    # Fit the model first
    regressor.fit(X_dummy, y_dummy)
    
    # Now model_ is available
    print_correlations(regressor)
