# Spam Email Classification Using Random Forest

This project classifies emails as **spam** or **not spam** using a **Random Forest Classifier**. The dataset has been preprocessed with **TF-IDF Vectorization** to convert the text into numeric features, and **GridSearchCV** has been applied to optimize the model's performance by tuning hyperparameters.

## Project Structure

- `spam-classification-randomforest.ipynb`: The Jupyter Notebook containing the code for preprocessing, training, and evaluating the model.
- `README.md`: The readme file (this file) explaining the project.

## Features

- **TF-IDF Vectorization**: To convert email text into numerical features.
- **Random Forest Classifier**: To classify emails as spam or not.
- **Hyperparameter Tuning with GridSearchCV**: To optimize the classifier's performance by testing different combinations of hyperparameters.
- **Evaluation**: Performance is evaluated using metrics such as precision, recall, F1-score, and accuracy.

## Installation

### 1. Clone the repository

To get started, clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/spam-email-classification.git
cd spam-email-classification
```

### 2. Install dependencies

Create a virtual environment and install the required packages using `pip`:

```bash
pip install -r requirements.txt
```

Ensure that the following libraries are included in your environment:

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib` (if visualization is used)

You can also manually install them:

```bash
pip install numpy pandas scikit-learn
```

### 3. Dataset

Ensure that the dataset for classifying emails is present. The notebook expects a column of text (email content) and a column of labels (indicating whether it's spam or not).

### 4. Running the Notebook

You can run the Jupyter Notebook in your local environment:

```bash
jupyter notebook spam-classification-randomforest.ipynb
```

Follow the instructions in the notebook to preprocess the data, train the model, and evaluate the results.

## How to Use

### 1. Preprocessing

The text data is preprocessed by:
- Lowercasing all words.
- Removing punctuation and stopwords.
- Stemming the words to their root form.
  
The `TfidfVectorizer` is then used to transform the cleaned text into numerical features that can be used by the Random Forest classifier.

### 2. Model Training

The model is initially trained using a default **Random Forest Classifier**:

```python
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

The classifier is then improved by applying **GridSearchCV** to fine-tune the hyperparameters and find the optimal combination for better accuracy:

```python
param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30]}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)
```

### 3. Evaluation

The model's performance is evaluated on the test set using **classification report** and **accuracy score**. Example metrics include:

- **Initial Model Accuracy**: 
  - Accuracy: ~96%
  
- **Optimized Model Accuracy** (after hyperparameter tuning):
  - Accuracy: **98.35748%**

The improvement in accuracy after hyperparameter tuning is due to better selection of `n_estimators` and `max_depth`, which control the number of trees and their depth, leading to more robust predictions.

```python
y_pred_optimized = optimized_model.predict(X_test)
print(classification_report(y_test, y_pred_optimized))
print("Accuracy:", accuracy_score(y_test, y_pred_optimized))
```

## Why Hyperparameter Tuning Improves Accuracy

Hyperparameter tuning allows the model to find the most effective combination of parameters, which results in better performance. In this case:
- **n_estimators**: Controls the number of decision trees. More trees generally improve accuracy but may increase computation time.
- **max_depth**: Determines how deep each tree grows. Limiting depth can prevent overfitting while still capturing relevant patterns in the data.

By using **GridSearchCV**, we test multiple combinations of these parameters and select the best-performing one, leading to improved accuracy.

## Results

- **Initial Accuracy**: ~96%  
- **Optimized Accuracy (after GridSearchCV)**: **98.35748%**

With hyperparameter tuning, the optimized model outperforms the initial model, delivering better results in classifying spam emails.

## Contributing

If you'd like to contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a pull request.

---
