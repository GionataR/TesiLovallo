import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate, GridSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from myfun import plot_learning_curve
from sklearn.model_selection import validation_curve
from sklearn.datasets import load_digits

#------------------------------------------------------------------------------
n_jobs = -1 #Attention, all available processors will be used
#------------------------------------------------------------------------------
# import data
data = pd.read_csv("C:\\Users\\Dino2\\Desktop\\DS\\cumulative.csv")
#------------------------------------------------------------------------------
LABEL="LABEL"
#------------------------------------------------------------------------------
# Analisys constant
TEST_SCORE = "test_score"
TRAIN_SCORE = "train_score"
AVERAGE = "Average"
PCT_STANDARD_DEVIATION = "% Standard Deviation"
STANDARD_DEVIATION = "Standard Deviation"

#------------------------------------------------------------------------------
#Columns Definition 
ROWID="rowid"
KEPID="kepid"
KEPOI_NAME="kepoi_name"
KEPLER_NAME="kepler_name"
KOI_DISPOSITION="koi_disposition"
KOI_PDISPOSITION="koi_pdisposition"
KOI_SCORE="koi_score"
KOI_FPFLAG_NT="koi_fpflag_nt"
KOI_FPFLAG_SS="koi_fpflag_ss"
KOI_FPFLAG_CO="koi_fpflag_co"
KOI_FPFLAG_EC="koi_fpflag_ec"
KOI_PERIOD="koi_period"
KOI_PERIOD_ERR1="koi_period_err1"
KOI_PERIOD_ERR2="koi_period_err2"
KOI_TIME0BK="koi_time0bk"
KOI_TIME0BK_ERR1="koi_time0bk_err1"
KOI_TIME0BK_ERR2="koi_time0bk_err2"
KOI_IMPACT="koi_impact"
KOI_IMPACT_ERR1="koi_impact_err1"
KOI_IMPACT_ERR2="koi_impact_err2"
KOI_DURATION="koi_duration"
KOI_DURATION_ERR1="koi_duration_err1"
KOI_DURATION_ERR2="koi_duration_err2"
KOI_DEPTH="koi_depth"
KOI_DEPTH_ERR1="koi_depth_err1"
KOI_DEPTH_ERR2="koi_depth_err2"
KOI_PRAD="koi_prad"
KOI_PRAD_ERR1="koi_prad_err1"
KOI_PRAD_ERR2="koi_prad_err2"
KOI_TEQ="koi_teq"
KOI_TEQ_ERR1="koi_teq_err1"
KOI_TEQ_ERR2="koi_teq_err2"
KOI_INSOL="koi_insol"
KOI_INSOL_ERR1="koi_insol_err1"
KOI_INSOL_ERR2="koi_insol_err2"
KOI_MODEL_SNR="koi_model_snr"
KOI_TCE_PLNT_NUM="koi_tce_plnt_num"
KOI_TCE_DELIVNAME="koi_tce_delivname"
KOI_STEFF="koi_steff"
KOI_STEFF_ERR1="koi_steff_err1"
KOI_STEFF_ERR2="koi_steff_err2"
KOI_SLOGG="koi_slogg"
KOI_SLOGG_ERR1="koi_slogg_err1"
KOI_SLOGG_ERR2="koi_slogg_err2"
KOI_SRAD="koi_srad"
KOI_SRAD_ERR1="koi_srad_err1"
KOI_SRAD_ERR2="koi_srad_err2"
RA="ra"
DEC="dec"
KOI_KEPMAG="koi_kepmag"
#------------------------------------------------------------------------------
#Definition of numeric variables
num_variables=[ROWID,KEPID,KOI_SCORE,KOI_FPFLAG_NT,KOI_FPFLAG_SS,KOI_FPFLAG_CO,KOI_FPFLAG_EC,
               KOI_PERIOD,KOI_PERIOD_ERR1,KOI_PERIOD_ERR2,KOI_TIME0BK,KOI_TIME0BK_ERR1,KOI_TIME0BK_ERR2,
               KOI_IMPACT,KOI_IMPACT_ERR1,KOI_IMPACT_ERR2,KOI_DURATION,KOI_DURATION_ERR1,KOI_DURATION_ERR2,
               KOI_DEPTH,KOI_DEPTH_ERR1,KOI_DEPTH_ERR2,KOI_PRAD, KOI_PRAD_ERR1,KOI_PRAD_ERR2,
               KOI_TEQ,KOI_TEQ_ERR1,KOI_TEQ_ERR2,KOI_INSOL,KOI_INSOL_ERR1,KOI_INSOL_ERR2,KOI_MODEL_SNR,
               KOI_TCE_PLNT_NUM,KOI_TCE_DELIVNAME,KOI_STEFF,KOI_STEFF_ERR1,KOI_STEFF_ERR2,
               KOI_SLOGG,KOI_SLOGG_ERR1,KOI_SLOGG_ERR2,KOI_SRAD,KOI_SRAD_ERR1,KOI_SRAD_ERR2,RA,DEC,KOI_KEPMAG]
#------------------------------------------------------------------------------
#Cleaning Data 


data.drop(
  
    [KEPLER_NAME,KOI_TEQ_ERR1,KOI_TEQ_ERR2,KOI_TCE_DELIVNAME,ROWID,KEPID,KEPOI_NAME,KEPLER_NAME], axis=1, inplace=True

    )
    
#------------------------------------------------------------------------------
#Replace missing data with the mean
data[KOI_SCORE].fillna(data[KOI_SCORE].mean(), inplace=True)
data[KOI_PERIOD_ERR1].fillna(data[KOI_PERIOD_ERR1].mean(), inplace=True)
data[KOI_PERIOD_ERR2].fillna(data[KOI_PERIOD_ERR2].mean(), inplace=True)
data[KOI_TIME0BK_ERR1].fillna(data[KOI_TIME0BK_ERR1].mean(), inplace=True)
data[KOI_TIME0BK_ERR2].fillna(data[KOI_TIME0BK_ERR2].mean(), inplace=True)
data[KOI_IMPACT].fillna(data[KOI_IMPACT].mean(), inplace=True)
data[KOI_IMPACT_ERR1].fillna(data[KOI_IMPACT_ERR1].mean(), inplace=True)
data[KOI_IMPACT_ERR2].fillna(data[KOI_IMPACT_ERR2].mean(), inplace=True)
data[KOI_DURATION].fillna(data[KOI_DURATION].mean(), inplace=True)
data[KOI_DURATION_ERR1].fillna(data[KOI_DURATION_ERR1].mean(), inplace=True)
data[KOI_DURATION_ERR2].fillna(data[KOI_DURATION_ERR2].mean(), inplace=True)
data[KOI_DEPTH].fillna(data[KOI_DEPTH].mean(), inplace=True)
data[KOI_DEPTH_ERR1].fillna(data[KOI_DEPTH_ERR1].mean(), inplace=True)
data[KOI_DEPTH_ERR2].fillna(data[KOI_DEPTH_ERR2].mean(), inplace=True)
data[KOI_PRAD_ERR1].fillna(data[KOI_PRAD_ERR1].mean(), inplace=True)
data[KOI_PRAD_ERR2].fillna(data[KOI_PRAD_ERR2].mean(), inplace=True)
data[KOI_PRAD].fillna(data[KOI_PRAD].mean(), inplace=True)
data[KOI_TEQ].fillna(data[KOI_TEQ].mean(), inplace=True)
data[KOI_INSOL].fillna(data[KOI_INSOL].mean(), inplace=True)
data[KOI_INSOL_ERR1].fillna(data[KOI_INSOL_ERR1].mean(), inplace=True)
data[KOI_INSOL_ERR2].fillna(data[KOI_INSOL_ERR2].mean(), inplace=True)
data[KOI_MODEL_SNR].fillna(data[KOI_MODEL_SNR].mean(), inplace=True)
data[KOI_TCE_PLNT_NUM].fillna(data[KOI_TCE_PLNT_NUM].mean(), inplace=True)

data[KOI_STEFF].fillna(data[KOI_STEFF].mean(), inplace=True)
data[KOI_STEFF_ERR1].fillna(data[KOI_STEFF_ERR1].mean(), inplace=True)
data[KOI_STEFF_ERR2].fillna(data[KOI_STEFF_ERR2].mean(), inplace=True)
data[KOI_SLOGG].fillna(data[KOI_SLOGG].mean(), inplace=True)
data[KOI_SLOGG_ERR1].fillna(data[KOI_SLOGG_ERR1].mean(), inplace=True)
data[KOI_SLOGG_ERR2].fillna(data[KOI_SLOGG_ERR2].mean(), inplace=True)
data[KOI_SRAD].fillna(data[KOI_SRAD].mean(), inplace=True)
data[KOI_SRAD_ERR1].fillna(data[KOI_SRAD_ERR1].mean(), inplace=True)
data[KOI_SRAD_ERR2].fillna(data[KOI_SRAD_ERR2].mean(), inplace=True)
data[RA].fillna(data[RA].mean(), inplace=True)
data[DEC].fillna(data[DEC].mean(), inplace=True)
data[KOI_KEPMAG].fillna(data[KOI_KEPMAG].mean(), inplace=True)




plt.figure()
sns.heatmap(data.isnull())
plt.show()

#------------------------------------------------------------------------------
#Drop NA Columns

#Plot of correlation Matrix with pearson method
corr_df = data
corr = corr_df.corr(method='pearson')
plt.figure()
plt.title("Correlation Plot")
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True), square= True)
plt.show()
#------------------------------------------------------------------------------
#Get dummies

data = pd.get_dummies(data,
                      columns=[KOI_DISPOSITION,KOI_PDISPOSITION], drop_first=True)
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# #Test and train data splitting 
y = data[KOI_SCORE]

x = data.drop([KOI_SCORE], axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

# #Definition of the models to include in the analysis ad optimization process 
models = {
    "RandomForest": RandomForestRegressor(n_estimators=200),
    "Gradient Boosting": GradientBoostingRegressor(),
    #"K Neighbors": KNeighborsRegressor(),
    #"Decision Tree": DecisionTreeRegressor(),
    "Neural Network": MLPRegressor((20, 20, 20), max_iter=1000,random_state=1)
 }

train_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, PCT_STANDARD_DEVIATION])
test_accuracies = pd.DataFrame(index=models.keys(), columns=[AVERAGE, PCT_STANDARD_DEVIATION])

# #------------------------------------------------------------------------------
for model_name, model in models.items():
    cv = cross_validate(model, x, y, cv=50, n_jobs=-1)
    avg_train_accuracy=(np.mean(cross_val_score(model, x_train, y_train, cv=50, n_jobs=-1)))
    std_train_accuracy = (np.std(cross_val_score(model, x_train, y_train, cv=50, n_jobs=-1)))
    pct_std_train_accu = std_train_accuracy / avg_train_accuracy * 100
    train_accuracies.loc[model_name] = [avg_train_accuracy, pct_std_train_accu]

    avg_test_accuracy=(np.mean(cross_val_score(model, x_test, y_test, cv=50, n_jobs=-1)))
    std_test_accuracy = (np.std(cross_val_score(model, x_test, y_test, cv=50, n_jobs=-1)))

    pct_std_test_accu = std_test_accuracy / avg_test_accuracy * 100
    test_accuracies.loc[model_name] = [avg_test_accuracy, pct_std_test_accu]
    plot_learning_curve(model, model_name, x, y, cv=50, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 20))
    model.fit(x_train, y_train)

print("Train accuracies:\n", train_accuracies.astype("float").round(2))
print("Test accuracies:\n", test_accuracies.astype("float").round(2))




