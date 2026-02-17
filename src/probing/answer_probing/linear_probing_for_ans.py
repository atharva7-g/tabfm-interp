from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from src.probing.answer_probing.linear_probe import extract_activations
from src.utils.utils import set_seed, create_dummy_dataset
from src.probing.answer_probing.linear_probe import train_model


def main():
    """Main function"""
    set_seed(42)
    X_data1_train, y_data1_train = create_dummy_dataset(
        weights=[10, 13], num_samples=1000, bias=False
    )
    X_data1_test, y_data1_test = create_dummy_dataset(
        weights=[10, 13], num_samples=10000, bias=False
    )
    X_data2_train, y_data2_train = create_dummy_dataset(
        weights=[2, 11], num_samples=1000, bias=False
    )
    X_data2_test, y_data2_test = create_dummy_dataset(
        weights=[2, 11], num_samples=10000, bias=False
    )
    regressor1 = train_model(X_data1_train, y_data1_train)
    regressor2 = train_model(X_data2_train, y_data2_train)

    activations1 = extract_activations(
        regressor1, regressor1.model_, X_data1_test, device="cuda"
    )
    activations2 = extract_activations(
        regressor2, regressor2.model_, X_data2_test, device="cuda"
    )
    for layer, activation in activations1.items():
        model = LinearRegression()
        linear_train = activation[:8000, -1, :].detach().cpu().numpy()
        linear_y_train = y_data1_test[:8000]
        model.fit(linear_train, linear_y_train)
        print("Testing on linear regressor trained on dataset 2")
        linear_train = activations2[layer][:8000, -1, :].detach().cpu().numpy()
        linear_y_train = y_data2_test[:8000]
        print(
            f"Layer {layer} train mse: {mean_squared_error(linear_y_train, model.predict(linear_train)):.4f}"
        )
        print(
            f"Layer {layer} train r2 score: {model.score(linear_train, linear_y_train):.4f}"
        )
        print("--------------------------------")


if __name__ == "__main__":
    main()
