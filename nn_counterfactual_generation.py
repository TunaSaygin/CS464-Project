import get_datasets as get_ds
import binary_classifier_nn as nn_model
import torch
import NSGA_vanilla as nsga
X_train,y_train, X_val, y_val, X_test, y_test = get_ds.get_datasets("merged_data.csv")
model = nn_model.BinaryClassifier()
model.load_state_dict(torch.load("best_nn_sgd.pth"))
def nn_predict(samples):
    return torch.round(model(samples)).detach().numpy()
X_test_filtered = X_test[y_test == 1]
y_test_filtered = y_test[y_test == 1]
X_test_filtered_array = X_test_filtered.to_numpy()
y_test_filtered_array = y_test_filtered.to_numpy()
nn_final_population = nsga.create_counterfactuals(X_test_filtered_array[0],X_test_filtered_array,0,nn_predict,50,300, requiresTensor=True)
print("NN_final population[0] = ",nn_final_population[0])
nsga.plot_features(X_test_filtered_array[0],nn_final_population[0]["features"],y_test_filtered_array[0],nn_final_population[0]["prediction"])
nsga.save_counterfactual_results(X_test_filtered_array,nn_predict,"./nn_counterfactual.csv", requiresTensor=True)