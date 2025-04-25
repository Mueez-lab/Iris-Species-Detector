## 📁 Dataset

The dataset contains **150 rows** and **5 columns**:

- `sepal_length`
- `sepal_width`
- `petal_length`
- `petal_width`
- `species` (Target: Iris-setosa, Iris-versicolor, Iris-virginica)

---

## 📊 Features & Engineering

New features were created to enhance model performance:

- `sepal_ratio` = `sepal_length / sepal_width`
- `petal_area` = `petal_length * petal_width`

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn (for visualization)
- Scikit-learn (`sklearn`)

---

## 🚀 Model Used

- **RandomForestRegressor** from `sklearn.ensemble`
- Hyperparameter tuning using `GridSearchCV`

---

## 🧪 Performance

- Achieved R² score of **~0.98** on test data.
- Model was tuned with parameters:
  - `n_estimators`: [50, 100, 150]
  - `max_depth`: [None, 4, 8]
  - `min_samples_split`: [2, 4]

---

## 📂 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com//Mueez-lab/Iris-Species-Detector-rf.git
   cd iris-classification-rf

pip install -r requirements.txt

Run the script:

    python iris_random_forest.py

📈 Visualizations

    Histogram of each feature

    Correlation heatmap

    Feature importance from Random Forest

📌 Notes

    Label encoding was used to convert species names to numeric.

    Missing values (if any) were dropped.

    Feature engineering was applied without dropping original features for model flexibility.

💡 Future Improvements

    Try other models (Logistic Regression, XGBoost)

    Implement classification instead of regression

    Deploy using Streamlit or Flask

👨‍💻 Author

    Your Name
    GitHub

📜 License

This project is licensed under the MIT License.


---
