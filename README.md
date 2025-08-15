# Aneurysm Predictor

A machine learning model developed for the **RSNA Intracranial Aneurysm Detection** Kaggle competition.  
Originally intended for competition submission, but due to a **system error** on Kaggle, this notebook could not be submitted.  
I’m sharing it here to keep a public record of my projects.

---

## 📌 About This Project
This model uses **Logistic Regression** to predict intracranial aneurysms based on features from medical imaging metadata.  
It was trained on the Kaggle competition dataset:

🔗 **Competition Link:** [RSNA Intracranial Aneurysm Detection](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection)  
📂 **Dataset:** Included in this repository for convenience (in case Kaggle access is unavailable).

---

## 🧠 What is an Aneurysm?
An **aneurysm** is a condition where a blood vessel in the brain swells or balloons, which can lead to severe health risks.  
Symptoms may include:
- Headache  
- Fever  
- Double vision  
- Other neurological changes  

---

## 📊 Dataset Features
Some example features from the dataset include:
- **PatientAge**
- **PatientSex**
- **Left Infraclinoid Internal**
- **Right Infraclinoid Internal**
- **Modality**
- …and more

---

## 📈 Model Performance
**Classification Report:**
  0       0.98      0.34      0.50       858
  1       0.03      0.74      0.06        23

    accuracy                           0.35       881
   macro avg       0.50      0.54      0.28       881
weighted avg       0.95      0.35      0.49       881

ROC-AUC score : 0.5133779264214047
## ⚠️ Disclaimer
This project was originally created for a Kaggle competition. Due to a submission system error, it was not evaluated on Kaggle’s leaderboard.  
It is shared here for learning and reference purposes.
