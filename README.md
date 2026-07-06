# SMS Spam Detection using Simple RNN

## Overview

This project implements an SMS Spam Detection system using a **Simple Recurrent Neural Network (RNN)** built with TensorFlow/Keras. The model classifies SMS messages into two categories: **Spam** and **Ham (Not Spam)**. A user-friendly **Streamlit** web application is provided for real-time message prediction.

---

## Features

* Detects whether an SMS message is Spam or Ham.
* Uses text preprocessing and tokenization.
* Implements a Many-to-One Simple RNN architecture.
* Interactive web interface built with Streamlit.
* Automatically trains the model if no saved model is found.
* Saves the trained model and tokenizer for future predictions.

---

## Technologies Used

* Python
* TensorFlow / Keras
* Streamlit
* Pandas
* Scikit-learn
* Pickle
* Regular Expressions (Regex)

---

## Dataset

The project uses the **SMS Spam Collection Dataset**.

**Dataset File:** `spam.csv`

Required columns:

| Column | Description      |
| ------ | ---------------- |
| v1     | Label (ham/spam) |
| v2     | SMS Message      |

During preprocessing, these columns are renamed to:

* `label`
* `text`

---

## Project Structure

```
RNN/
│
├── app.py                 # Main Streamlit application
├── spam.csv               # SMS dataset
├── spam_model.keras       # Trained model (generated after training)
├── tokenizer.pkl          # Saved tokenizer (generated after training)
├── README.md
└── requirements.txt
```

---

## Model Architecture

```
Input Text
      │
      ▼
Text Cleaning
      │
      ▼
Tokenizer
      │
      ▼
Padding Sequences
      │
      ▼
Embedding Layer
      │
      ▼
Simple RNN Layer
      │
      ▼
Dense Layer (ReLU)
      │
      ▼
Output Layer (Sigmoid)
      │
      ▼
Spam / Ham Prediction
```

---

## Installation

Clone the repository:

```bash
git clone <https://github.com/Saikoushik14/SMS_Spam_Detector>
```

Move into the project folder:

```bash
cd RNN
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at:

```
http://localhost:8501
```

---

## Working

1. Load the SMS dataset.
2. Clean the text using regular expressions.
3. Convert text into numerical sequences using the Keras Tokenizer.
4. Pad sequences to a fixed length.
5. Train a Simple RNN model.
6. Save the trained model and tokenizer.
7. Predict whether new SMS messages are Spam or Ham through the Streamlit interface.

---

## Example Inputs

### Spam

```
Congratulations! You have won a FREE iPhone. Click here to claim your prize now.
```

### Ham

```
Hi, are we meeting at 5 PM today?
```

---

## Output

Example:

```
Prediction : Spam

Confidence : 99.42%
```

or

```
Prediction : Ham

Confidence : 98.75%
```

---

## Future Improvements

* Add LSTM and GRU models for comparison.
* Improve UI with charts and prediction history.
* Support multiple languages.
* Deploy the application on Streamlit Cloud.
* Visualize training accuracy and loss.

---

## Author

**Sai Koushik Kasula**

B.Tech – Data Science

Anurag University

---

## License

This project is developed for educational and learning purposes.
