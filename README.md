
# Cyberbullying Detection Using Machine Learning

![Cyberbullying Detection](https://img.shields.io/badge/Project-Cyberbullying%20Detection-blue)
![Python Version](https://img.shields.io/badge/Python-3.8+-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📖 Project Overview

**Cyberbullying Detection Using Machine Learning** is a project aimed at identifying and classifying cyberbullying in online comments. The project compares various machine learning classifiers to identify the most effective model for detecting **harassment** and **hate speech** in text data.

The models tested include:
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier
- AdaBoost Classifier
- Stochastic Gradient Descent (SGD) Classifier
- Linear SVC
- Bagging Classifier
- Decision Tree Classifier

## 🚀 Features
- **Multiple Classifiers**: Comparison of different models to find the most effective.
- **Text Preprocessing**: Uses **TF-IDF Vectorization** for text representation.
- **Evaluation Metrics**: Results are shown using metrics such as **Accuracy**, **Precision**, **Recall**, and **F1-Score**.
- **Real-World Application**: Addresses the issue of cyberbullying on social media.

## 🔧 Installation

### Clone the Repository
To get started, clone this repository to your local machine:

```bash
git clone https://github.com/SriSathwik1905/CyberBullying.git
cd CyberBullying
```

### Install Dependencies

Use `pip` to install the required libraries:

```bash
pip install -r requirements.txt
```

Alternatively, install manually:

```bash
pip install scikit-learn pandas numpy matplotlib
```

## 🧑‍💻 Usage

### Training the Models

Once you've set up the environment, you can train the models using the following command:

```bash
python train_models.py
```

### Evaluating the Models

After training, evaluate the performance of the models:

```bash
python evaluate_models.py
```

### Results Overview

The model performances (accuracy, precision, recall, F1-score) will be output, helping you compare their effectiveness for cyberbullying detection.

## 📊 Results

### Model Performance

| Classifier               | Accuracy (%) | Precision (0) | Precision (1) | Precision (2) | Recall (0) | Recall (1) | Recall (2) | F1-Score (0) | F1-Score (1) | F1-Score (2) |
|--------------------------|--------------|---------------|---------------|---------------|------------|------------|------------|--------------|--------------|--------------|
| **AdaBoost Classifier**   | 64           | 0.77          | 0.60          | 0.55          | 0.68       | 0.79       | 0.38       | 0.72         | 0.68         | 0.45         |
| **Gradient Boosting**     | 64           | 0.80          | 0.59          | 0.56          | 0.64       | 0.81       | 0.41       | 0.71         | 0.68         | 0.47         |
| **Random Forest**         | 64           | 0.74          | 0.60          | 0.56          | 0.73       | 0.79       | 0.34       | 0.73         | 0.68         | 0.42         |
| **SGD Classifier**        | 65           | 0.72          | 0.62          | 0.59          | 0.75       | 0.79       | 0.36       | 0.74         | 0.70         | 0.44         |
| **Logistic Regression**   | 63           | 0.73          | 0.60          | 0.55          | 0.68       | 0.78       | 0.38       | 0.71         | 0.68         | 0.45         |
| **Linear SVC**            | 62           | 0.68          | 0.63          | 0.50          | 0.69       | 0.70       | 0.41       | 0.69         | 0.66         | 0.45         |
| **Bagging Classifier**    | 63           | 0.72          | 0.63          | 0.51          | 0.71       | 0.72       | 0.42       | 0.71         | 0.67         | 0.46         |
| **Decision Tree**         | 56           | 0.64          | 0.59          | 0.43          | 0.64       | 0.61       | 0.41       | 0.64         | 0.60         | 0.42         |

The **SGD Classifier** outperformed others with an accuracy of **65%**, while **Decision Tree** performed the worst.

## 🧠 Future Directions

1. **Deep Learning**: Implement more complex models like **RNNs** and **LSTMs** for improved text understanding.
2. **NLP Enhancements**: Explore **contextual embedding models** (e.g., **BERT**) for better detection of hate speech.
3. **Real-time Detection**: Integrate the models into **social media platforms** for live monitoring.
4. **Ensemble Learning**: Use **Stacking** and **Boosting** for combining model strengths.

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

