# 🧠 Fetal Health Classification with PyTorch

This project implements a **multi-class classification model** using **PyTorch** to predict fetal health status based on cardiotocography (CTG) features. The dataset is sourced from Kaggle and includes clinical measurements related to fetal heart rate and uterine activity.

---

## 📊 Dataset

- **Source:** Kaggle – Fetal Health Classification
- **Samples:** 2126
- **Features:** 21 numerical features
- **Target:** `fetal_health`

### Class Encoding

| Original Label | Encoded Label | Description |
|---------------|--------------|-------------|
| 1 | 0 | Normal |
| 2 | 1 | Suspect |
| 3 | 2 | Pathological |

Labels were shifted to start from zero to be compatible with PyTorch’s `CrossEntropyLoss`.

---

## 🔧 Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- PyTorch

---

## 🧪 Data Preprocessing

- Stratified train / validation / test split  
  - Train: 72%  
  - Validation: 8%  
  - Test: 20%
- Feature standardization using `StandardScaler`
- Conversion to PyTorch tensors
- Data loading using `TensorDataset` and `DataLoader`

---

## 🤖 Model Architecture

A simple **multiclass logistic regression** implemented in PyTorch.

```python
model = nn.Sequential(
    nn.Linear(21, 3),
    nn.Softmax(dim=1)
)
```

**Architecture Summary:**

* Input layer: 21 features
* Output layer: 3 classes
* Activation: Softmax

---

## ⚙️ Training Configuration

* **Loss Function:** CrossEntropyLoss
* **Optimizer:** SGD
  * Learning Rate: 0.1
  * Momentum: 0.9
* **Batch Size:** 100

---

## 📈 Objective

* Build a baseline neural network classifier
* Practice PyTorch data pipelines
* Perform multiclass classification
* Prepare a foundation for more advanced neural networks

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies
3. Download the dataset from Kaggle
4. Run the notebook or script

```bash
pip install numpy pandas matplotlib scikit-learn torch
```

---

## 🔮 Future Improvements

* Add hidden layers (MLP)
* Remove Softmax and use raw logits (best practice)
* Add evaluation metrics (accuracy, confusion matrix)
* Use Adam optimizer
* Hyperparameter tuning

---

## 📜 License

This project is intended for educational and learning purposes.
