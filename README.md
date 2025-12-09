# Dam Power Discharge Prediction🌊

This project predicts hydropower discharge using dam operation data from Korea Water Resources Corporation (K-water).

This project was created to explore how machine learning can support dam operation and hydropower management by predicting power discharge based on real hydrological and operational data.
Hydropower generation is highly dependent on water level, storage conditions, inflow, and rainfall. However, traditional operation often relies on rule-based or manual decision-making. By applying data-driven models, this project aims to:

* Improve understanding of the relationship between hydrological variables and power discharge
* Support more efficient dam operation strategies
* Demonstrate the practical use of machine learning in water resource engineering

Rather than focusing only on model accuracy, this project emphasizes **interpretability, comparison of different models, and real-world applicability**.

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

### 2) (Optional) Create virtual environment

    python -m venv venv

Activate the virtual environment:

Windows:

    venv\Scripts\activate

macOS / Linux:

    source venv/bin/activate

### 3) Install required packages

    pip install -r requirements.txt

If the file does not exist, install manually:

    pip install pandas numpy matplotlib scikit-learn

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

When executed, the console will display:

* Model evaluation results (MSE, RMSE, MAE, R²)
* Hyperparameter tuning results
* Feature importance rankings

Graphs will also be displayed for:

* Model comparison
* Actual vs Predicted values
* Feature importance

* * *

## Notes

* Large CSV files are excluded using `.gitignore`.
* This project tracks only source code and documentation in GitHub.

* * *

## Need Help?

If you would like improvements or additional features added to this project, feel free to ask!
