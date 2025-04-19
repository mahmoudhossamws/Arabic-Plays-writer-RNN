
# 🎭 Shakespearean Text Generator – Character-Level LSTM Model

A **deep learning text generation** project using **TensorFlow** and **Keras**, trained on the works of *William Shakespeare*. The model learns to generate text character by character in the style of Shakespeare's iconic language and storytelling.

---

## 📚 Project Overview

This project feeds an LSTM-based RNN with Shakespeare's writings and trains it to learn the patterns, structure, and language. After training, it can generate new, *Shakespearean-style* text based on a user-provided prompt.

> 📘 This implementation is based on the excellent [FreeCodeCamp tutorial](https://www.youtube.com/watch?v=tPYj3fFJGjk&t=21762s/) for text generation using TensorFlow.

---

## ✨ Key Features

- 🔡 **Character-level modeling** – Understands and generates text one character at a time.
- 🧠 **LSTM neural network** – Learns long-term dependencies in text.
- 💾 **Checkpoint system** – Saves weights across training epochs.
- ✍️ **Interactive text generation** – Enter a starting phrase and watch the bard come alive.

---

## 🛠️ Technologies Used

- **Language:** Python  
- **Libraries:** TensorFlow, Keras, NumPy  
- **Concepts:** RNNs, LSTM, Character Encoding, Text Preprocessing  

---

## ▶️ How to Run

1. Make sure you have Python 3 and the required packages installed:
   ```bash
   pip install tensorflow numpy
   ```

2. Place `book.txt` (Shakespeare text) in the same directory.

3. Run the script:
   ```bash
   python project.py
   ```

4. After training finishes, you can generate text by entering a starting prompt.

---

## 📸 Screenshots

<p float="left">
  <img src="training.png" width="600" />
  <img src="example.png" width="600" />
</p>

---

## 📬 Contact

If you have any questions or suggestions, feel free to reach out:

📧 mahmoudhossam@aucegypt.edu

---
