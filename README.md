# Dam Power Discharge Prediction🌊

<p align="center">
    <img src="https://github.com/user-attachments/assets/65c9d601-35ed-44b6-8c63-f126a3bea14a" width="45%" />
  <img src="https://github.com/user-attachments/assets/2e80c22a-e65c-45a7-8227-05eb6fe44c35" width="45%" />
</p>

This project uses dam operation data from K-water to examine whether the commonly assumed linear relationship between water discharge and hydropower output holds in real-world conditions.
Traditional hydropower theory is based on the equation P = ρ · g · Q · H · η, which implies P ∝ Q under constant head and efficiency.
By applying machine learning models, this project explores nonlinear relationships between hydrological and operational variables, emphasizing interpretability and real-world applicability over pure predictive accuracy.

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

* * *

## Output Example
<p align="center">
  <img src="https://github.com/user-attachments/assets/4810c494-dca5-453f-8c64-d209e32221e8" width="32%" />
  <img src="https://github.com/user-attachments/assets/8887b0c7-5c91-41fb-87ff-981a1496db3d" width="32%" />
  <img src="https://github.com/user-attachments/assets/67c5b010-7fad-4dbc-ad43-76c2c71bf70d" width="32%" />
</p>

When executed, the console will display:

* Model evaluation results (MSE, RMSE, MAE, R²)
* Hyperparameter tuning results
* Feature importance rankings

Graphs will also be displayed for:

* Model comparison
* Actual vs Predicted values
* Feature importance

## Discussion

The Random Forest model demonstrated substantially better performance than linear and polynomial models, highlighting the limitations of the commonly assumed linear relationship in hydropower discharge modeling. The results confirm that while discharge plays a dominant role, reservoir conditions such as water level and storage volume exert significant influence on dam discharge behavior.

Key observations:

* Linear models failed to adequately represent the complex structure of the data.

* Ensemble-based models, particularly Random Forest, achieved significantly lower prediction errors.

These findings support the central objective of this project: to demonstrate that dam discharge is governed by nonlinear interactions among hydrological and operational variables, and that purely linear assumptions are insufficient for realistic hydropower modeling.

* * *

## Notes

* Large CSV files are excluded using `.gitignore`.
* This project tracks only source code and documentation in GitHub.

* * *

## Need Help?

If you would like improvements or additional features added to this project, feel free to ask!
leeyerin2697@kentech.ac.kr
