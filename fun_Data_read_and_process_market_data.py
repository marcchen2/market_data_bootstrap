import pandas as pd
import fun_Data_timeseries_basket


def read_and_process_market_data(data_settings, timeseries_basket):
    #Objective: extract market returns for the ASSET basket dictionary given by "timeseries_basket"
    #timeseries_basket["basket_type"] MUST BE "asset"

    # IMPORTANT:
    #   Market data provided always assumed to be NOMINAL returns
    #   - if params["real_or_nominal"] = "real", the inflation-adjusted returns time series will be constructed here
    #   Training and testing datasets (constructed subsequently) will use SUBSETS of the yyyymm_start and yyyymm_end selected here

    #OUTPUTS:
    # data_returns = pandas DataFrame with index (yyyymm) and columns corresponding to
    yyyymm_start = data_settings["data_read_yyyymm_start"]  #Start date to use for historical market data, set to None for data set start
    yyyymm_end = data_settings["data_read_yyyymm_end"]    #End date to use for historical market data, set to None for data set end
    input_folder = data_settings["data_read_input_folder"]
    input_file = data_settings["data_read_input_file"]
    input_file_type = data_settings["data_read_input_file_type"] #suffix
    delta_t = data_settings["data_read_delta_t"]  #time interval for returns data (monthly returns means data_delta_t=1/12)
    returns_format = data_settings["data_read_returns_format"]  #'percentages' = already multiplied by 100 but without added % sign
                                                        #'decimals' is percentages in decimal form
    skiprows = data_settings["data_read_skiprows"]  #nr of rows of file to skip before start reading
    index_col = data_settings["data_read_index_col"] #Column INDEX of file with yyyymm to use as index
    header = data_settings["data_read_header"] #INDEX of row AFTER "skiprows" to use as column names
    na_values = data_settings["data_read_na_values"]  #how missing values are identified in the data

    real_or_nominal = data_settings["real_or_nominal"]  #do we want real or nominal market data


    if timeseries_basket["basket_type"] == "asset":
        asset_basket_id = timeseries_basket["basket_id"]   #basket_id

        #Check if cash needs to be added
        if "Cash" in timeseries_basket["basket_timeseries_names"]:
            add_cash_TrueFalse = True
        else:
            add_cash_TrueFalse = False

        # Read .xlsx file with returns
        data_returns_nominal = pd.read_excel(input_folder + "/" + input_file  + input_file_type,
                                     skiprows=skiprows,
                                     index_col=index_col,
                                     header=header, 
                                     na_values=na_values, engine='openpyxl') #if xlrd gives error, add option: engine='openpyxl'

        if returns_format == "percentages":
            data_returns_nominal = data_returns_nominal.copy()/100      #Always work with decimal form


        #CHECK:
        print(data_returns_nominal.head())
        print(data_returns_nominal.tail())


        #Get column names for underlying data which is always assumed to be NOMINAL returns
        timeseries_basket_NOMINAL = fun_Data_timeseries_basket.\
            timeseries_basket_construct(basket_type = timeseries_basket["basket_type"], #must be ASSET basket
                                        basket_id=asset_basket_id,
                                        real_or_nominal = "nominal",
                                        add_cash_TrueFalse = add_cash_TrueFalse)

        data_returns_columns = timeseries_basket_NOMINAL["basket_columns"]

        if real_or_nominal == "real": #append CPI because it will be needed for deflating the returns
            data_returns_columns.append("CPI_nom_ret")

        #Drop columns and rows and check:
        data_returns_nominal.drop(columns= [col for col in data_returns_nominal if col not in data_returns_columns],
                          inplace = True)

        if yyyymm_end is not None:
            data_returns_nominal = data_returns_nominal.loc[data_returns_nominal.index <= yyyymm_end].copy()
        if yyyymm_start is not None:
            data_returns_nominal = data_returns_nominal.loc[data_returns_nominal.index >= yyyymm_start].copy()

        #CHECK:
        print(data_returns_nominal.head())
        print(data_returns_nominal.tail())


        #If real data is to be used, calculate real (inflation-adjusted) returns
        if real_or_nominal == "real":

            #Construct indices for assets and CPI
            data_returns_indices = construct_indices_from_returns(df = data_returns_nominal)
            data_returns_indices.drop(columns="date_for_plt", inplace=True) #No need for plot

            #Calculate inflation_adjusted returns
            data_returns_real = pd.DataFrame(index= data_returns_indices.index) #initialize

            for timeseries in timeseries_basket_NOMINAL["basket_timeseries_names"]:

                #Step 1: Inflation-adjust indices
                data_returns_indices[timeseries+"_real_ret_ind"] = 100. * data_returns_indices[timeseries+"_nom_ret_ind"]         \
                                                                / data_returns_indices["CPI_nom_ret_ind"]

                #Step 2: Calculate inflation-adjusted returns using pct_change method (will be *monthly* returns)
                data_returns_real[timeseries+"_real_ret"] = data_returns_indices[timeseries+"_real_ret_ind"].pct_change(periods=1)

            #Step 3: Drop the first row with "na" values
            data_returns_real.drop(labels=data_returns_real.iloc[0].name, inplace=True)


        #Set output
        if real_or_nominal == "nominal":
            data_returns = data_returns_nominal
        elif real_or_nominal == "real":
            data_returns = data_returns_real

    else:
        raise ValueError("PVSerror in 'read_and_process_market_data': code can only process returns for asset baskets")


    return data_returns


def construct_indices_from_returns(df, #pd.DataFrame with returns time series; index in format yyyymm
                                    yyyymm_start = None,    #if None, just runs from start of data until <= yyyymm_end or end of data
                                    yyyymm_end = None,  #if None, just runs from dates >=yyyymm_start until end of data
                                    columns_to_keep = None, #if None, keep all columns
                                    wealth_start = 100.0 #wealth invested at the *end* of month [yyyymm_start MINUS 1 month]
                                   ):

    #Inputs:
    # df = pd.DataFrame with RETURNS time series in DECIMAL form (i.e. already divided by 100)

    # Outputs: pd.DataFrame = 'df_index' with time series of INDEX values
    #   df_index COLUMN names in "columns_to_keep" will have "_ind" appended
    #   df_index ROWS will have first row appended at end of month [yyyymm_end MINUS 1 month]
    # return df_index

    df = df.copy()


    #----------------------------------------------------------------------------
    # Keep only dates and columns of interest


    #Select dates of interest
    if yyyymm_end is not None:
        df = df.loc[df.index <= yyyymm_end].copy()
    if yyyymm_start is not None:
        df = df.loc[df.index >= yyyymm_start].copy()

    #Keep only columns of interest
    if columns_to_keep is not None:
        df.drop(columns = [col for col in df.columns if col not in columns_to_keep], inplace=True)


    #Add date for plotting (only used for plotting)
    # - Technically wrong, since it gives START date of month rather than end date
    #   but this doesn't matter for plotting purposes
    dates_as_datetime = pd.to_datetime(df.index, format='%Y%m') #only used for plotting
    df["date_for_plt"] = dates_as_datetime


    #------------------------------------------------------------
    # Add FIRST ROW (with "zero" returns) for index construction
    # - construct dataframe with indices
    #   Investing 'wealth_start' over [yyyymm_start, yyyymm_end]

    #Create first row for index construction
    df_index_firstrow = pd.DataFrame()
    df_index_firstrow = df[0:1].copy() #copy first row to get column headings
    df_index_firstrow["date_for_plt"] = df_index_firstrow["date_for_plt"] - pd.DateOffset(months=1)   #subtract one month to get start right
    df_index_firstrow["yyyymm"] = df_index_firstrow["date_for_plt"].iloc[0].year * 100 + df_index_firstrow["date_for_plt"].iloc[0].month
    df_index_firstrow.set_index("yyyymm", drop = True, inplace=True)

    for col in df_index_firstrow.columns:
        if col != "date_for_plt":
            df_index_firstrow[col] = 0.0  #set to zero to make code work below

    #Insert first row
    df_index = pd.concat([df_index_firstrow, df], axis=0, join="outer", ignore_index=False)

    del df_index_firstrow


    #------------------------------------------------------------
    #Construct indices starting at wealth_start at the beginning of start_yyyymm

    for col in df_index.columns:      #Loop through df columns
        if col != "date_for_plt":
            # Replace returns columns with index columns
            df_index[col + "_ind"] = wealth_start * (1+df_index[col]).cumprod()
            df_index.drop(columns=[col], inplace=True)

    print("Indices constructed with column names:")
    print(df_index.columns)

    return df_index