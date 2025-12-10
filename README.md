# Dam Power Discharge Prediction🌊

<p align="center">
    <img src="https://github.com/user-attachments/assets/65c9d601-35ed-44b6-8c63-f126a3bea14a" width="45%" />
  <img src="https://github.com/user-attachments/assets/2e80c22a-e65c-45a7-8227-05eb6fe44c35" width="45%" />
</p>
Demonstration Video: https://youtu.be/LBqvcdxHdpE

This project uses dam operation data from K-water to predict turbine discharge (Q) used for hydropower generation. Turbine discharge is a critical variable in hydropower systems, as it directly determines power output according to the fundamental hydropower equation:

P = ρ · g · Q · H · η

Because discharge (Q) is a primary driver of power generation, accurately predicting turbine discharge enables indirect estimation of hydropower output and supports more efficient dam operation and energy planning. This project applies machine learning models to forecast turbine discharge from hydrological and operational variables, emphasizing practical value for real-world dam management rather than purely theoretical modeling.
* * *

## Project Structure

    project_root/
    │
    │── model.py                 # Main training and evaluation script
    │── .gitignore               # Excludes large CSV files and virtual environments
    │── requirements.txt
    │── README.md

* * *

## Environment

This project was developed and tested in the following environment:

* Python 3.10+
* Operating System: Windows 10/11
* Required Libraries:
  * pandas
  * numpy
  * matplotlib
  * scikit-learn

* * *

## Installation & Setup

### 1) Clone the repository

    git clone https://github.com/leeyerin2697/Dam-power-prediction.git
    cd Dam-power-prediction

### 2) Create virtual environment

    python -m venv venv

Activate the virtual environment:

Windows:

    venv\Scripts\activate

macOS / Linux:

    source venv/bin/activate

### 3) Install required packages

    pip install -r requirements.txt


* * *

## Dataset

Place the following file in the project root directory:

    한국수자원공사_수문현황정보_일별.csv
    https://www.data.go.kr/data/15083335/fileData.do

This dataset is not included in the repository due to GitHub file size limits.

### Feature Columns

* water_level (저수위)
* storage_volume (저수량)
* inflow_rate (유입량)
* total_discharge (총방류량)
* rainfall (강수량)
* cumulative_rainfall (금년누가강우량)
* storage_ratio (저수율)

### Target Column

* power_discharge (발전방류량)

* * *

## How to Run

Run the script from the project root directory.

    python model.py

* * *

## Pipeline Flow

1. Load CSV data
2. Rename Korean columns to English
3. Remove missing and invalid values
4. Sample large datasets (up to 50,000 rows)
5. Train machine learning models
   * Linear Regression
   * Polynomial Regression
   * Random Forest
6. Evaluate models using MSE, RMSE, MAE, and R²
7. Visualize results using matplotlib
8. Hyperparameter experiment(random forest)

* * *

## Output Example


## Prediction Results

<p float="left">
  <img src="https://github.com/user-attachments/assets/1f42cf9d-1adb-4f11-86b3-056408ec8901" width="300"/>
  <img src="https://github.com/user-attachments/assets/e86ac76d-b2c2-4ad2-b8ad-71c7e31b267f" width="300"/>
  <img src="https://github.com/user-attachments/assets/12e67e52-3ff0-4fb5-b9ea-505ff1de8bae" width="300"/>
</p>

---

## Model Performance

<p float="left">
  <img src="https://github.com/user-attachments/assets/73bfe73e-2ac5-482d-b910-99f63fc7c4ab" width="300"/>
  <img src="https://github.com/user-attachments/assets/a50060f5-636f-44df-bdcd-6bc1862fcfb5" width="300"/>
  <img src="https://github.com/user-attachments/assets/1681e687-633e-4f47-a297-a1cc83fc1436" width="300"/>
</p>

---

## Feature Importance

<p float="left">
  <img src="https://github.com/user-attachments/assets/738a0ff5-832b-49b1-b5c5-13a1229206dd" width="350"/>
</p>

---

When executed, the console will display:

* Model evaluation results (MSE, RMSE, MAE, R²)
* Hyperparameter tuning results
* Feature importance rankings

Graphs will also be displayed for:

* Model comparison
* Actual vs Predicted values
* Feature importance

## Discussion

The Random Forest model significantly outperformed linear and polynomial models, demonstrating that simple linear approaches are insufficient for predicting turbine discharge under real-world dam operating conditions. The results indicate that turbine discharge (Q) cannot be explained by a single factor, but is influenced by a combination of hydrological and operational variables, including water level and storage volume.

Key observations:

* Linear regression models showed limited ability to represent the nonlinear behavior of dam operations.

* Ensemble-based methods, particularly Random Forest, achieved substantially lower prediction errors.

These findings are important because turbine discharge is a key variable in the hydropower equation (P = ρ · g · Q · H · η). By improving the prediction of Q, this study demonstrates the potential to indirectly estimate hydropower generation and provide useful insights for dam operation and energy management.
* * *

## Notes

* Large CSV files are excluded using `.gitignore`.
* This project tracks only source code and documentation in GitHub.

* * *

## Need Help?

If you would like improvements or additional features added to this project, feel free to ask!
leeyerin2697@kentech.ac.kr
