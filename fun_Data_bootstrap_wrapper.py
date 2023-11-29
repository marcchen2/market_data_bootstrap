#Objective: Wraps the bootstrap functions to reduce the length of main code


import copy
import fun_Data_read_and_process_market_data
import fun_Data_timeseries_basket
import fun_Data_bootstrap


def wrap_run_bootstrap(train_test_Flag, #"train" or "test"
                        params,  #params dictionary as in main code
                        data_bootstrap_yyyymm_start,    #start month to use subset of data for bootstrapping, CHECK DATA!
                        data_bootstrap_yyyymm_end,      #end month to use subset of data for bootstrapping, CHECK DATA!
                        data_bootstrap_exp_block_size,  #Expected block size in terms of frequency of market returns data
                                                                #e.g. = X means expected block size is X months of returns
                                                                # if market returns data is monthly
                        data_bootstrap_fixed_block,     #if False: use STATIONARY BLOCK BOOTSTRAP, if True, use FIXED block bootstrap
                        data_bootstrap_delta_t #time interval for returns data (monthly returns means data_delta_t=1/12)
                        ):

    #OUTPUT: params dictionary, with following fields appended/modified
    #        Note: Below, xx \in {"train", "test"}

    # params["bootstrap_settings_xx"]: inputs used to get the bootstrapping results
    # ASSET return data: always appended
    #   params["Y_xx"][j, n, i] = Return, along sample path j, over time period (t_n, t_n+1), for asset i
    #       -- IMPORTANT: params["Y_xx"][j, n, i] entries are basically (1 + return),
    #                           so it is ready for multiplication with start value
    #   params["Y_order_xx"][i] = column name of asset i used for identification
    #
    # TRADING SIGNAL data: only appended if params["use_trading_signals_TrueFalse"] == True
    #   params["TradSig_xx"][j, n, i] = Point-in time observation for trade signal i, along sample path j, at rebalancing time t_n;
    #                               can only rely on time series observations <= t_n
    #   params["TradSig_order_xx"][i] = column name of trade signal i used for identification

    params = copy.deepcopy(params)

    if train_test_Flag == "train":
        N_d = params["N_d_train"]

    elif train_test_Flag == "test":
        N_d = params["N_d_test"]

    # ----------------------------------------
    # Data bootstrap settings
    bootstrap_settings = {}   #initialize dictionary
    bootstrap_settings["N_d"] = N_d     #Number of datapoints in the training data set
    bootstrap_settings["purpose"] = train_test_Flag   #"train" or "test"
    bootstrap_settings["data_bootstrap_yyyymm_start"] = data_bootstrap_yyyymm_start #start month to use subset of data for bootstrapping, CHECK FILE!
    bootstrap_settings["data_bootstrap_yyyymm_end"] = data_bootstrap_yyyymm_end #end month to use subset of data for bootstrapping, CHECK FILE!

    bootstrap_settings["data_bootstrap_exp_block_size"] = data_bootstrap_exp_block_size    #Expected block size in terms of frequency of market returns data
                                        #e.g. = X means expected block size is X months of returns
                                        # if market returns data is monthly
                                     # Run Matlab code to get optimal expected blocksizes for comparison

    bootstrap_settings["data_bootstrap_fixed_block"] = data_bootstrap_fixed_block   #if False: use STATIONARY BLOCK BOOTSTRAP
                                    #           then blocksize will be sampled from geometric distribution
                                    #           with parameter (1 / bootstrap_exp_block_size)

    bootstrap_settings["data_bootstrap_delta_t"] = data_bootstrap_delta_t #time interval for returns data (monthly returns means data_delta_t=1/12)

    bootstrap_settings["data_bootstrap_n_tot"] = params["T"] / bootstrap_settings["data_bootstrap_delta_t"]  #Number of entries in each bootstrap sample
                                        #Matches frequency of market returns data


    # ----------------------------------------
    # APPEND bootstrap settings to "params" dictionary
    params["bootstrap_settings_" + train_test_Flag] = bootstrap_settings.copy()   #copy over

    del bootstrap_settings


    #----------------------------------------
    #APPEND bootstrapped data to "params" dictionary

    params = fun_Data_bootstrap.run_bootstrap(df_all = params["bootstrap_source_data"],
                                              bootstrap_settings= params["bootstrap_settings_" + train_test_Flag] ,
                                              params = params)



    return params


def wrap_append_market_data(params,  #params dictionary as in main code
                            data_read_yyyymm_start = None, #Start date to use for historical market data, set to None for data set start
                            data_read_yyyymm_end = None,  #End date to use for historical market data, set to None for data set end
                            data_read_input_folder = "",    #folder (relative path)
                            data_read_input_file = "",  #just the filename, no suffix
                            data_read_input_file_type = ".xlsx",  # file suffix
                            data_read_delta_t = 1 / 12,  # time interval for returns data (monthly returns means data_delta_t=1/12)
                            data_read_returns_format = "percentages",  # 'percentages' = already multiplied by 100 but without added % sign
                                                                        # 'decimals' is percentages in decimal form
                            data_read_skiprows = 0 , # nr of rows of file to skip before start reading
                            data_read_index_col = 0,  # Column INDEX of file with yyyymm to use as index
                            data_read_header = 0,  # INDEX of row AFTER "skiprows" to use as column names
                            data_read_na_values = "nan" , # how missing values are identified in the data
                            ):


    #OUTPUT:

    # -- if run_code == False: just returns params dictionary, no changes or modifications!

    # -- if run_code == True: params dictionary, with following fields appended/modified
    #          params["asset_basket"]: (existing field modified) modified by appending historical data
    #                           and associated key stats (mean, stdev, corr matrix) to asset_basket
    #          params["asset_basket_data_settings"]: new dictionary appended  historical data extraction settings for record
    #          params["trading_signal_basket"]:  (existing field modified) modified by appending historical data
    #                                   and associated key stats (mean, stdev, corr matrix) to trading_signal_basket
    #                                   Trading signals constructed *lagged* to avoid look-ahead
    #          params["trading_signal_basket_data_settings"]: new dictionary appended historical data extraction settings for record

    #          params["bootstrap_source_data"]: (new field) pandas.DataFrame with time series ready for bootstrapping:
    #                                           1) Inflation adjusted if necessary,
    #                                           2) Trade signals and asset returns merged
    #                                           3) NaNs removed (at start due to trade signal calculation)

    # Note: if real_or_nominal = "real" (assets or trade signals), the inflation-adjusted returns time series will be constructed here


    #Decide if we need to run this code: Only need to run if market data is required
    

    # -----------------------------------------------------------------------------------------------
    # Underlying assets: Read and process market data
    # -----------------------------------------------------------------------------------------------

    #IMPORTANT:
    #   Market data provided always assumed to be NOMINAL returns
    #   - if params["real_or_nominal"] = "real", the inflation-adjusted returns time series will be constructed here
    #   Training and testing datasets (constructed subsequently) will use SUBSETS of the yyyymm_start and yyyymm_end selected here


    data_settings = {}

    # Market data file import and process settings
    data_settings["data_read_yyyymm_start"] = data_read_yyyymm_start  #Start date to use for historical market data, set to None for data set start
    data_settings["data_read_yyyymm_end"] = data_read_yyyymm_end    #End date to use for historical market data, set to None for data set end
    data_settings["data_read_input_folder"] = data_read_input_folder
    data_settings["data_read_input_file"] = data_read_input_file
    data_settings["data_read_input_file_type"] = data_read_input_file_type  #suffix
    data_settings["data_read_delta_t"] = data_read_delta_t #time interval for returns data (monthly returns means data_delta_t=1/12)
    data_settings["data_read_returns_format"] = data_read_returns_format  #'percentages' = already multiplied by 100 but without added % sign
                                                        #'decimals' is percentages in decimal form
    data_settings["data_read_skiprows"] = data_read_skiprows   #nr of rows of file to skip before start reading
    data_settings["data_read_index_col"] = data_read_index_col #Column INDEX of file with yyyymm to use as index
    data_settings["data_read_header"] = data_read_header #INDEX of row AFTER "skiprows" to use as column names
    data_settings["data_read_na_values"] = data_read_na_values #how missing values are identified in the data

    data_settings["real_or_nominal"] = params["real_or_nominal"]    #if "real", will process the (nominal) market data
                                                                        # to obtain inflation-adjusted  returns

# Read and process data
    data_returns = fun_Data_read_and_process_market_data.read_and_process_market_data(data_settings = data_settings,
                                                                                        timeseries_basket=params["asset_basket"])


    #Append historical data and associated key stats (mean, stdev, corr matrix) to asset_basket
    params["asset_basket"] = fun_Data_timeseries_basket.timeseries_basket_append_info(data_df = data_returns,
                                                                                timeseries_basket= params["asset_basket"])
    #Append historical data settings for record
    params["asset_basket_data_settings"] = data_settings.copy()

    #clean up
    del data_settings, data_returns



# -----------------------------------------------------------------------------------------------
    # Prepare final market dataset for bootstrapping: Merge and clean
    # -----------------------------------------------------------------------------------------------

    # SOURCE data for bootstrapping
    df_asset_returns = params["asset_basket"]["data_df"]
    df_all = df_asset_returns.copy()

        # Clean up
    del df_asset_returns
    
     # Identify rows with nan values, and remove
        # - this will be the INITIAL rows only, due to calculation of trading_signals
    df_all_isnan = (df_all.isna()).sum(axis=1)
    df_all_isnan = df_all_isnan[df_all_isnan > 0]
    print("-----------------------------------------------")
    print("Dates to be REMOVED due to missing values (should be at *start* only due to trading signals):")
    print(df_all_isnan.index)

    # Remove rows with nan values
    df_all = df_all.loc[[idx for idx in df_all.index if idx not in df_all_isnan.index]].copy()

    print("-----------------------------------------------")
    print("Dates from available data for bootstrapping:")
    print("Start: " + str(min(df_all.index)))
    print("End: " + str(max(df_all.index)))
    print("-----------------------------------------------")

    # Append to params
    params["bootstrap_source_data"] = df_all



    return params

