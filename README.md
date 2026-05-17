# AIE683 – ML Forecasting Inflation with PCA

This repository contains code, datasets, and outputs for the course project **Efficient and Robust Macroeconomic Forecasting under Resource Constraints**.  
The project applies **Principal Component Analysis (PCA)** to macroeconomic datasets for inflation forecasting, implemented in both **R** and **Python**.

Always standardize datasets before PCA.
Align all series to quarterly frequency.
Outputs are reproducible across R and Python implementations.

This paper investigates dimensionality reduction techniques in macroeconomic forecasting under resource constraints. Drawing on principal component analysis (PCA) and dynamic factor models, we examine how high-dimensional macroeconomic datasets over 100 variables. 
The study evaluates the trade-off between efficiency and information loss, using nowcasting and forecasting approaches.

R code fetching from IMF databases..
You can fetch data from the International Monetary Fund (IMF) directly in R using the imfapi package, which provides a user-friendly four-step workflow to explore dataflows, dimensions, and codelists before executing your data pull

Pyhton by manual csv datas loading..
# Load data exported from csv datas in my local folder path
df = pd.read_csv(   r"D:\Articles@arf\AIE683-ML-Forecasting-Inflation_PCA_R,Py\AIE683-ML-Forecasting-Inflation_PCA_R,Py\final_data2.csv"      )


## 📂 Project Structure
─ PCA_to_forecast_ICLR.pdf           # Paper draft (ICLR format)
├── README.md                         # Project documentation
├── data_raw.csv                      # Original raw dataset
├── data_clean.csv                    # Cleaned dataset (missing values handled)
├── data_excel_treated.csv            # Excel-preprocessed dataset
├── data_treated.csv                  # Treated dataset (first version)
├── data_treated2.csv                 # Treated dataset (second version)
├── wide_data.csv                     # Wide-format dataset for PCA
├── yield_curve.csv                   # Yield curve data (bond maturities)
├── X_filled.csv                      # Filled dataset (missing values interpolated)
├── X_filled_python.csv               # Python-prepared filled dataset
├── X_scaled_python.csv               # Python-prepared standardized dataset
├── evds_pca.csv                      # PCA-ready dataset from EVDS source
├── final_data.csv                    # Final dataset (main version)
├── final_data2.csv                   # Final dataset (alternative version)
├── na_ratio.csv                      # Missing value ratio analysis
├── pc_df.csv                           # Principal component dataframe
├── pc_scores_python.csv                # PCA scores (Python output)
├── pca_explained_variance_python.csv   # Explained variance ratios (Python output)
├── pca_loadings_python.csv             # PCA loadings (Python output)
├── gdp_pca.RData                       # Saved R PCA object
├── gdp_pca1.R                          # R script for PCA
├── gdp_pca_python.py                   # Python script for PCA



## ⚙️ Requirements
### Python
- Python 3.9+
- Libraries:
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

Install dependencies:
pip install -r requirements.txt

# Python PCA Workflow – Explanation
This script performs **Principal Component Analysis (PCA)** on a macroeconomic dataset to extract latent factors for inflation forecasting.  
It is designed to preprocess raw data, standardize variables, and generate interpretable PCA outputs such as explained variance, loadings, and component scores.

Step-by-Step Explanation
1. Import Libraries
import pandas as pd                                            # pandas: for data handling
from sklearn.preprocessing import StandardScaler               # scikit-learn: StandardScaler (standardization) and PCA (dimensionality reduction)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt                                # matplotlib: for visualization (scree plot, component plots)

2. Load Data
df = pd.read_csv("final_data2.csv")                            # Reads the dataset exported from R (final_data2.csv).
print(df.head())                                               # Displays the first few rows to confirm structure.

3. Preprocessing
X = df.drop(columns=["time"])                                # Remove time column
X = X.apply(pd.to_numeric, errors="coerce")                  # Ensure numeric types
X = X.loc[:, X.isna().mean() < 0.35]                         # Keep variables with <35% missing
X = X.interpolate(method="linear")                           # Fill missing values , Fills gaps using linear interpolation
X = X.dropna()                                               # Remove remaining NA

4. Standardization - This step is critical for meaningful PCA results.
scaler = StandardScaler()                                    # Standardizes each variable (mean = 0, variance = 1).
X_scaled = scaler.fit_transform(X)                           # Prevents variables with large scales (e.g., GDP vs CPI) from dominating PCA.


5. PCA Computation
pca = PCA()                                                # Fits PCA on the standardized dataset. 
scores = pca.fit_transform(X_scaled)                       # scores are the principal component values (latent factors)

6. Explained Variance
print(pca.explained_variance_ratio_)                        # Shows how much variance each principal component explains. Used to decide how many components to retain (e.g., first 2–3 explain most variance).


7. Scree Plot                                        # Visualizes explained variance by component. Helps identify the “elbow point” where additional components add little value.
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1),
         pca.explained_variance_ratio_, marker="o")
plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.grid(True)
plt.show()


8. Loadings                                               # Shows how each original variable contributes to each principal component. Example: CPI and PPI may load heavily on PC1 (inflation factor).

loadings = pd.DataFrame(pca.components_.T,
                        index=X.columns,
                        columns=[f"PC{i+1}" for i in range(len(X.columns))])
print(loadings.iloc[:, 0:3].round(2))



9. PCA Factors                                         # Extracts the first two principal components as time series. These latent factors can be used in forecasting models (e.g., OLS regression).

pc_df = pd.DataFrame({
    "PC1": scores[:, 0],
    "PC2": scores[:, 1]
})
print(pc_df.head())


10. Plot PC1                                        # Visualizes the first principal component over time. Helps interpret macroeconomic dynamics (e.g., inflationary pressure trends).

plt.plot(pc_df["PC1"])
plt.title("PC1")
plt.xlabel("Observation")
plt.ylabel("PC1")
plt.grid(True)
plt.show()


Python Outputs
    # Explained variance ratios → importance of each component
    # Scree plot → visual guide for component selection
    # Loadings table → variable contributions to components
    # PC1/PC2 time series → latent factors for forecasting models

Notes : Always standardize before PCA.   Handle missing values carefully (interpolation, dropping).  Interpret loadings to connect components with economic meaning.




# R PCA Workflow – Explanation,  R4.0+

This R script performs **Principal Component Analysis (PCA)** on macroeconomic datasets retrieved from IMF and EVDS sources.  
It preprocesses raw time series, applies growth transformations, standardizes variables, and generates interpretable PCA outputs such as scree plots, loadings, and component scores.

Step-by-Step Explanation
Packages:
stats
ggplot2
readr

 1. Load Libraries
library(rdbnomics)          # Fetch IMF/IFS data          # rdbnomics: connects to IMF/IFS datasets.
library(dplyr)              # Data manipulation           # dplyr: tidy data manipulation.
library(zoo)                # Quarterly time handling     # zoo: handles quarterly time indices.
library(readr)              # CSV import                  # readr/lubridate: import and parse bond yield CSVs.
library(lubridate)          # Date parsing

2. Define Series     # Defines IMF/IFS series codes for imports, exports, GDP, inflation, production, and labor indicators
   
series <- c(
  import = "IMF/IFS/Q.TR.NM_R_SA_XDC",
  export = "IMF/IFS/Q.TR.NX_R_SA_XDC",
  deposit_rate = "IMF/IFS/Q.TR.FIDR_PA",
  policy_rate  = "IMF/IFS/Q.TR.FPOLM_PA",
  fiscal_balance = "IMF/IFS/Q.TR.GG_GXOBP_G01_XDC",
  real_gdp = "IMF/IFS/Q.TR.NGDP_R_SA_XDC",
  cpi_inflation_qoq = "IMF/IFS/Q.TR.PCPI_PC_PP_PT",
  ppi_index = "IMF/IFS/Q.TR.PPPI_IX",
  electricity_prod_growth = "IMF/IFS/Q.TR.AIPEE_PC_PP_PT",
  manufacturing_prod_growth = "IMF/IFS/Q.TR.AIPMA_PC_PP_PT",
  mining_prod_growth = "IMF/IFS/Q.TR.AIPMI_PC_PP_PT",
  industrial_prod_sa = "IMF/IFS/Q.TR.AIP_PC_PP_SA_PT",
  unemployment = "IMF/IFS/Q.TR.LUR_PT",
  labor_index = "IMF/IFS/Q.TR.LE_IX",
  labor_force = "IMF/IFS/Q.TR.LLF_PE_PC_PP_PT"
)


3. Fetch and Clean Data            # Retrieves IMF data. Converts periods to quarterly format. Filters for 1998–2025. Produces a clean panel with time, variable, and value.

data_raw <- rdb(ids = unname(series))
data_clean <- data_raw %>%
  filter(!is.na(value)) %>%
  mutate(
    period2 = gsub("-", " ", original_period),
    time = as.yearqtr(period2, format = "%Y Q%q"),
    variable = names(series)[match(series_code,
                                   sub("IMF/IFS/", "", unname(series)))]
  ) %>%
  filter(time >= as.yearqtr("1998 Q1"),
         time <= as.yearqtr("2025 Q4")) %>%
  select(time, variable, value)


4. Wide Format Transformation             # Converts long format into wide format (variables as columns). Ensures chronological order.
   wide_data <- data_clean %>%
  pivot_wider(names_from = variable, values_from = value) %>%
  arrange(time)

5. Growth Transformations                  # Applies log‑difference growth rates (×100). Stabilizes variance and makes series stationary. Drops incomplete rows.
growth_vars <- c("import","export","real_gdp","ppi_index","labor_index")

data_treated <- wide_data %>%
  mutate(across(all_of(growth_vars),
                ~ 100 * (log(.) - log(lag(.))),
                .names = "{.col}")) %>%
  filter(complete.cases(.))


6. Bond Yield Integration            # Reads bond yield CSVs (2y, 3y, 5y, 10y).
   bond_2y <- read_yield_csv("Turkey 2-Year Bond Yield Historical Data.csv","bond_2y")
...
data_treated2 <- data_treated %>%
  left_join(yield_curve, by = "time") %>%
  arrange(time)

7. EVDS Data Integration            # Adds EVDS macro variables (debt, FX, money supply, equities, credit, FDI).

data_excel_treated <- evds_pca %>%
  mutate(across(all_of(growth_vars_excel),
                ~ 100 * (log(.) - log(lag(.))),
                .names = "{.col}"),
         budget_deficit = 100 * budget_deficit / gdp_nom,
         cariacik = 100 * cariacik / gdp_nom)


8. Final Dataset                    # Combines IMF, bond yields, and EVDS data.

final_data <- data_treated2 %>%
  full_join(data_excel_treated, by = "time") %>%
  mutate(fiscal_balance = 100 * fiscal_balance / gdp_nom)

final_data2 <- final_data %>% select(-gdp_nom)


9. Handle Missing Values                    #  Drops variables with >35% missing. Fills gaps using linear interpolation. Removes remaining NA rows.
    X <- final_data2 %>% select(-time)
X <- X %>% select(where(~ mean(is.na(.)) < 0.35))

X_filled <- X %>%
  mutate(across(everything(), ~ na.approx(., na.rm = FALSE))) %>%
  na.omit()

10. Standardization                    # Standardizes each variable (mean = 0, variance = 1). Ensures comparability across indicators.

X_scaled <- scale(X_filled)


11. PCA                        # Displays loadings for first 3 components.

pca <- prcomp(X_scaled)
summary(pca)
plot(pca, type = "l", main = "Scree Plot")
round(pca$rotation[,1:3], 2)


12. PCA Factors                    # Plots PC1 over time to visualize latent inflationary dynamics.
    pc_df <- data.frame(
  time = final_data2$time[(nrow(final_data2)-nrow(X_filled)+1):nrow(final_data2)],
  PC1 = pca$x[,1],
  PC2 = pca$x[,2]
)

plot(pc_df$time, pc_df$PC1, type = "l")


R code, Outputs:    
# Scree plot → variance explained by components
# Loadings matrix → variable contributions to PCs
# PC1/PC2 time series → latent factors for forecasting models
# gdp_pca.RData → saved PCA object for reuse
