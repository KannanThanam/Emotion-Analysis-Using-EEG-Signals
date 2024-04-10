# Emotion Analysis using EEG Signals

Welcome to my project repository! This project focuses on analyzing emotions using EEG signals.

## Requirements

- [Python 3.7](https://www.python.org/downloads/release/python-370/) - You will need to have Python 3.7 installed on your system.
- [Thonny IDE](https://thonny.org/) - Thonny is a beginner-friendly IDE for Python.

## Python Libraries

To install additional libraries required for this project, you can use pip, the Python package installer. Open your terminal or command prompt and run the following command:

```bash
pip install <library-name>
```

# Dataset

Download the dataset [here](https://drive.google.com/drive/folders/13jF8KnoJr6n61YOMvRebPv6iPVfjlhMU?usp=drive_link)
 - Rename the folder name ***Dataset*** to ***Data***

# Run Project

This project focuses on analyzing emotions using EEG signals.

- We have used CNN, KNN and XGBoost

## KNN
To run KNN download ***model.sav*** file [here](https://drive.google.com/file/d/1qQNEHYDGa5ycOnZ97DEpn4aNnXRnESJY/view?usp=sharing) - Request access.
Then Place the ***model.sav*** file in the project folder.

- Open ***KNN Train.py*** file run it in Thonny.
- It Shows Confusion Matrix as a result.

- Then run ***KNN Test.py***
- Opens an interface where you can select an image and result will be the emotion.

## CNN
To run CNN download ***image_features.csv*** file [here](https://drive.google.com/file/d/19ZJkJI5RJOrNfQ0UV4Blb4mIEs2ZAjJn/view?usp=drive_link) - Request access.
Then Place the ***image_features.csv*** file in the project folder.

- Open ***CNN Train.py*** file run it in Thonny.
- It Shows Confusion Matrix as a result.

- Then run ***CNN Test.py***
- Opens an interface where you can select an image and result will be the emotion.

## XGBoost
To run XGBoost download ***xgboost_model.joblib*** file [here](https://drive.google.com/file/d/19sPzh1SFg3o_kEXA4mUMwzgJDH5Kky9y/view?usp=drive_link) - Request access.
Then Place the ***xgboost_model.joblib*** file in the project folder.
- First run ***LG.py*** file in Thonny. *(It will update ***image features.csv*** file for 1st time alone)*
- Open ***XGBoost Train.py*** file run it in Thonny.
- It Shows Confusion Matrix as a result.

- Then run ***XGBoost Test.py***
- Opens an interface where you can select an image and result will be the emotion.

# Final Demo

Run ***final.py***
