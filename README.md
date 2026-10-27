# ML-examples

A reference collection of foundational machine learning algorithms in Python, mixing `scikit-learn`-based examples with from-scratch implementations written while studying the underlying math.

## Contents

### KNN (K-Nearest Neighbors)
- `Example1/` — Classifies the Pima Diabetes dataset (`KNN_Dataset.csv`) using `sklearn.neighbors.KNeighborsClassifier`, with standard scaling and a confusion matrix / F1 / accuracy report.
- `Example2/` — A from-scratch `KNN` class (`KNN.py`) implementing `fit`/`predict` via Euclidean distance and majority vote.
- `Example3/` — Notebook-based KNN exploration (`nb.ipynb`).

### LinearRegression
- `main.py` — Gradient descent linear regression from scratch (`gradient_descent`, `loss_function`) fit against `data.csv`, plotted with `matplotlib`.

### Perceptron
- `Example1/` — A from-scratch `Perceptron` class (`perceptron.py`) with the classic weight-update training loop and a unit-step activation.
- `Example2/` — Notebook variants exploring batch, mini-batch, and online training modes, plus a bias-focused notebook.

## Known Issues

While reading through the from-scratch implementations, two bugs stood out:

- **KNN `Example2/KNN.py`**: `euclidean_distance` takes the square root twice (`np.sqrt(np.sqrt(np.sum((x1-x2)**2)))`), which is not the actual Euclidean distance — it should be a single `np.sqrt(...)`.
- **LinearRegression `main.py`**: `return m, b` inside `gradient_descent` is indented inside the `for i in range(n)` loop, so the function returns after processing only the *first* data point on every call instead of accumulating the gradient over the full dataset.

## Requirements

`pandas`, `numpy`, `scikit-learn`, `matplotlib` (per example — see each script's imports).
