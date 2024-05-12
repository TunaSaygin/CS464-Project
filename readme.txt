Hello, this is how you should use the program

There is a file called Models.py. In this file, there are logistic regression and both SVM models.
When you run this file it will calculate the accuracies, confusion matrices and SHAP graphs.
Same thing applies for other models. You can run nn_model.py to run the neural network, decision_tree_model.py
to run the decision tree model. These will also give the accuracies and confusion matrices.

Now to generate the counterfactuals for each model, this will create the "Original vs. Counterfactual"
plot which shows information about each feature. Run the following files to generate counterfactuals for each model:

- decision_tree_counterfactual.py
- nn_counterfactual_generation.py
- svm_counterfactual_generation.py
- Models.py (this will create the logistic regression counterfactual csv file, ignore the svm csv file)

then, put the path of the generated csv files into read_plot_vals.py file, where it says "pd.read_csv"
at the line 7. This will create the counterfactual plots.