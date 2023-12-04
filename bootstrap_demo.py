import pandas as pd
import numpy as np
import torch
import fun_Data_bootstrap_wrapper
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------
# Define data generation parameters 
#-----------------------------------------------------------------------------------------------
params = {} #Initialize empty dictionary 
output_parameters = {}

params["T"] = 5 #Time horizon, in years
params["N_rb"] = 60  #Nr of equally-spaced periods of return data to generate in [0,T]
params["delta_t"] = params["T"] / params["N_rb"]    # Time interval

params["N_d_train"] = int(10) #Nr of data return sample paths to bootstrap

# automatic flag to see if running on GPU machine.
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
params["device"] = device

#set seed for both pytorch and numpy
params["random_seed"] = 42
np.random.seed(params["random_seed"])
torch.manual_seed(params["random_seed"])

#--------------------------------------
# ASSET BASKET: Specify basket of candidate assets, and REAL or NOMINAL data
params["asset_basket_id"] =  "Basic_stock_bond"   # Pre-defined basket of underlying candidate assets 
                                             # See fun_Data_timeseries_basket_ASSETS.py for other asset basket options, and to add new asset baskets. 

params["real_or_nominal"] = "real" # "real" or "nominal" for asset data for wealth process: if "real", the asset data will be deflated by CPI

params["add_cash_TrueFalse"] = False

#-----------------------------------------------------------------------------------------------
# Gathering Historical Market data: 
#   Data required for bootstrapping is extracted, processed (e.g. inflation adjusted)
#   and prepared for bootstrapping here.
#-----------------------------------------------------------------------------------------------

params = fun_Data_bootstrap_wrapper.wrap_append_market_data(
                            params = params,  #params dictionary as in main code
                            data_read_yyyymm_start = 192607, #Start date to use for historical market data, set to None for data set start
                            data_read_yyyymm_end = 202212,  #End date to use for historical market data, set to None for data set end
                            data_read_input_folder = "Market_data", #folder name (relative path)
                            data_read_input_file = "market_data_example", #just the filename, no suffix
                            data_read_input_file_type = ".xlsx",  # file suffix
                            data_read_delta_t = 1 / 12,  # time interval for returns data (monthly returns means data_delta_t=1/12)
                            data_read_returns_format = "percentages",  # 'percentages' = already multiplied by 100 but without added % sign
                                                                        # 'decimals' is percentages in decimal form
                            data_read_skiprows = 15 , # nr of rows of file to skip before start reading
                            data_read_index_col = 0,  # Column INDEX of file with yyyymm to use as index
                            data_read_header = 0,  # INDEX of row AFTER "skiprows" to use as column names
                            data_read_na_values = "nan" # how missing values are identified in the data
                            )
#          params["bootstrap_source_data"]: (new field) pandas.DataFrame with time series ready for bootstrapping:
#                                           1) Inflation adjusted if necessary,
#                                           2) Trade signals and asset returns merged
#                                           3) NaNs removed (at start due to trade signal calculation)
#               for a given month, asset obs are at END of month, trade signals at BEGINNING of month

#Output bootstrap source data to Excel, if needed
output_bootstrap_source_data = False
if output_bootstrap_source_data:
    df_temp = params["bootstrap_source_data"]
    df_temp.to_excel("bootstrap_source_data.xlsx")
    

#-----------------------------------------------------------------------------------------------
# MARKET DATA GENERATOR: 
#-----------------------------------------------------------------------------------------------

params["output_csv_data_training_testing"] = False  #if True, write out data to .csv files

# ----------------------------------------
# Data bootstrapping
# - Append bootstrapped data to "params" dictionary as params["Y_train"]

params = fun_Data_bootstrap_wrapper.wrap_run_bootstrap(
    train_test_Flag = "train",                  # "train" or "test"
    params = params,                            # params dictionary as in main code
    data_bootstrap_yyyymm_start = 192601,       # start month to use subset of data for bootstrapping, CHECK DATA!
    data_bootstrap_yyyymm_end = 202212,         # end month to use subset of data for bootstrapping, CHECK DATA!
    data_bootstrap_exp_block_size = 3,  # Expected block size in terms of frequency of market returns data
                                                # e.g. = X means expected block size is X months of returns
                                                # if market returns data is monthly
    data_bootstrap_fixed_block = False,
    data_bootstrap_delta_t = 1 / 12             # time interval for returns data (monthly returns means data_delta_t=1/12)
)
# ----------------------------------------

#Plot indices, as demonstration

asset_names = params["asset_basket"]["basket_columns"]
returns = {}
indices = {}

for i,asset in enumerate(asset_names):
    returns[asset] = pd.DataFrame(params["Y_train"][:,:,i]).transpose()
    indices[asset] = returns[asset].cumprod().mul(100)
    
tick_step_yr = 1
xticks_loc = np.arange(0, params["N_rb"]+1, int(params["N_rb"]/params["T"])*tick_step_yr)
xticks_labels = np.arange(0,params["T"]+1, tick_step_yr)


f, (ax0, ax1, ax2) = plt.subplots(1, 3, sharey=True, figsize=(16, 5))
ax0.plot(indices[asset_names[0]])
ax0.set_title("Resampled 30-Day T-Bill",fontsize = 14)
ax0.set_xticks(xticks_loc, xticks_labels)

ax1.plot(indices[asset_names[1]])
ax1.set_title("Resampled 10-Year U.S. Bond",fontsize = 14)
ax1.set_xticks(xticks_loc, xticks_labels)

ax2.plot(indices[asset_names[2]])
ax2.set_title("Resampled SP500 Index",fontsize = 14)
ax2.set_xticks(xticks_loc, xticks_labels)

ax0.set_ylabel("Index",fontsize = 14)
ax1.set_xlabel("Year",fontsize = 14)

plt.tight_layout(pad=3)
f.savefig("example_plot.png")

check = 0