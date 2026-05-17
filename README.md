AIE 683 # Machine-Learning-_Final Study
This paper investigates dimensionality reduction techniques in macroeconomic forecasting under resource constraints. Drawing on principal component analysis (PCA) and dynamic factor models, we examine how high-dimensional macroeconomic datasets over 100 variables. 
The study evaluates the trade-off between efficiency and information loss, using nowcasting and forecasting approaches.

R code fetching from IMF databases..
You can fetch data from the International Monetary Fund (IMF) directly in R using the imfapi package, which provides a user-friendly four-step workflow to explore dataflows, dimensions, and codelists before executing your data pull


Pyhton by manual csv datas loading..
# Load data exported from csv datas in my local folder path
df = pd.read_csv(
    r"D:\Articles@arf\AIE683-ML-Forecasting-Inflation_PCA_R,Py\AIE683-ML-Forecasting-Inflation_PCA_R,Py\final_data2.csv"
)
