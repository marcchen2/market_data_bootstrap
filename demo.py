

import pandas as pd
import numpy as np
import os
from pathlib import Path
import re
import json
import torch

#Import files needed (other files are imported within those files as needed)
import fun_Data_timeseries_basket
import fun_Data_bootstrap_wrapper


#-----------------------------------------------------------------------------------------------
# Portfolio problem: Main structural parameters
#-----------------------------------------------------------------------------------------------
params = {} #Initialize empty dictionary 
output_parameters = {}

params["T"] = 30 #Investment time horizon, in years
params["N_rb"] = 30  #Nr of equally-spaced rebalancing events in [0,T]
          #Withdrawals, cash injections, AND rebalancing at times t_n = (n-1)*(T/N_rb), for n=1,...,N_rb
params["delta_t"] = params["T"] / params["N_rb"]    # Rebalancing time interval


# automatic flag to see if running on GPU machine.
device = 'cuda' if torch.cuda.is_available() else 'cpu'  
params["device"] = device

#set seed for both pytorch and numpy
params["random_seed"] = 42
np.random.seed(params["random_seed"])
print("\n numpy seed: ", params["random_seed"], " \n")
torch.manual_seed(params["random_seed"])
print("\n pytorch seed: ", params["random_seed"], " \n")


#----------------------------------------------------------------------
# Main settings for data: This is used for creating both training and testing data. 
params["N_d_train"] = int(2000) #Nr of data return sample paths to bootstrap


#--------------------------------------
# ASSET BASKET: Specify basket of candidate assets, and REAL or NOMINAL data
params["asset_basket_id"] =  "Paper_Basic"   # Pre-defined basket of underlying candidate assets 
                                             # See fun_Data_timeseries_basket_ASSETS.py for other asset basket options, and to add new 
                                             # asset baskets. 

params["real_or_nominal"] = "real" # "real" or "nominal" for asset data for wealth process: if "real", the asset data will be deflated by CPI

params["add_cash_TrueFalse"] = False

#-----------------------------------------------------------------------------------------------
# Asset basket and Feature specification (also specify basket of trading signals, if applicable)
#-----------------------------------------------------------------------------------------------

#Construct asset basket:
# - this will also give COLUMN NAMES in the historical returns data to use
params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_construct(
                            basket_type="asset",
                            basket_id=params["asset_basket_id"],
                            add_cash_TrueFalse=params["add_cash_TrueFalse"],
                            real_or_nominal = params["real_or_nominal"] )


#-----------------------------------------------------------------------------------------------
# Gathering Historical Market data: 
#   Data required for bootstrapping is extracted, processed (e.g. inflation adjusted)
#   and prepared for bootstrapping here.
#-----------------------------------------------------------------------------------------------

# following fields appended/modified to params dictionary:
#          params["asset_basket"]: (existing field) modified by appending historical data
#                           and associated key stats (mean, stdev, corr matrix) to asset_basket
#          params["asset_basket_data_settings"]: new dictionary appended  historical data extraction settings for record
#          params["bootstrap_source_data"]: (new field) pandas.DataFrame with time series ready for bootstrapping:
#                                           1) Inflation adjusted if necessary,
#                                           2) Trade signals and asset returns merged
#                                           3) NaNs removed (at start due to trade signal calculation)
#               for a given month, asset obs are at END of month, trade signals at BEGINNING of month

# Note: if real_or_nominal = "real" (assets or trade signals), the inflation-adjusted returns time series will be constructed here

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
    df_temp.to_excel(output_parameters["code_title_prefix"] + "bootstrap_source_data.xlsx")
    

#-----------------------------------------------------------------------------------------------
# MARKET DATA GENERATOR: Source data for training\testing
#-----------------------------------------------------------------------------------------------

params["output_csv_data_training_testing"] = False  #if True, write out training/testing data to .csv files


# ----------------------------------------
# Data bootstrapping
# - Append bootstrapped data to "params" dictionary
blocksize = 3
print("Bootstrap block size: " + str(blocksize))
params = fun_Data_bootstrap_wrapper.wrap_run_bootstrap(
    train_test_Flag = "train",                  # "train" or "test"
    params = params,                            # params dictionary as in main code
    data_bootstrap_yyyymm_start = 192601,       # start month to use subset of data for bootstrapping, CHECK DATA!
    data_bootstrap_yyyymm_end = 202212,         # end month to use subset of data for bootstrapping, CHECK DATA!
    data_bootstrap_exp_block_size = blocksize,  # Expected block size in terms of frequency of market returns data
                                                # e.g. = X means expected block size is X months of returns
                                                # if market returns data is monthly
    data_bootstrap_fixed_block = False,
    data_bootstrap_delta_t = 1 / 12             # time interval for returns data (monthly returns means data_delta_t=1/12)
)

# ASSET return data: always appended (if market data required
#   params["Y_train"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
#       -- IMPORTANT: params["Y_train"][j, n, i] entries are basically (1 + return), so it is ready for multiplication with start value
#   params["Y_order_train"][i] = column name of asset i used for identification
# ----------------------------------------
check = 0