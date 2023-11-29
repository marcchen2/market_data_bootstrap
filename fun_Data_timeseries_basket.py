import fun_Data_timeseries_basket_ASSETS

#Objective: constructs timeseries baskets: used for both ASSETS and TRADING SIGNAL (feature vector) constructions
#IMPORTANT: If RETURNS, this code assumes the  historical returns time series in DECIMAL form, i.e. already divided by 100


def timeseries_basket_construct( basket_type, #"asset" or "trading_signal"
                                basket_id, #   Basket ID of asset_basket or trading_signal_basket
                                real_or_nominal,     #whether nominal or real historical returns should be used.
                                add_cash_TrueFalse = False, #if True, cash is added as an asset
                                underlying_asset_basket_id = None   #Used only for TRADING SIGNAL baskets
                            ):

    #OUTPUT: dictionary 'timeseries_basket'

    timeseries_basket = {}  #dictionary with timeseries basket info

    # ------------------------------------------------------------------------------------------------
    # Construct timeseries basket

    if basket_type == "asset":
        timeseries_basket = fun_Data_timeseries_basket_ASSETS.asset_basket(basket_type = basket_type, basket_id = basket_id,
                                     real_or_nominal = real_or_nominal, add_cash_TrueFalse = add_cash_TrueFalse)

   


    # ------------------------------------------------------------------------------------------------
    # Print info
    print("\n")
    print("############# Defined " + basket_type + " basket #################")
    print("timeseries_basket.keys() = " )
    print(timeseries_basket.keys() )
    print("timeseries_basket['basket_type'] = " + timeseries_basket['basket_type'])
    print("timeseries_basket['basket_id'] = " + timeseries_basket['basket_id'])
    print("timeseries_basket['basket_desc'] = " + timeseries_basket['basket_desc'])
    print("timeseries_basket['basket_columns'] = ")
    print(timeseries_basket['basket_columns'])
    print("############# End: defined " + basket_type + "  basket #################")


    return timeseries_basket



def timeseries_basket_append_info(data_df,    #pd.DataFrame with historical time series (if returns, in DECIMAL form)
                       timeseries_basket   #timeseries basket dictionary to which to append info
                        ):

    #Append historical returns data to timeseries_basket

    #Dataframes with key information
    data_df_mean = data_df.mean(axis=0)
    data_df_stdev = data_df.std(axis=0, ddof=1)
    data_df_corr = data_df.corr(method = "pearson")


    timeseries_basket.update({"data_df" : data_df})
    timeseries_basket.update({"data_df_mean": data_df_mean})
    timeseries_basket.update({"data_df_stdev": data_df_stdev})
    timeseries_basket.update({"data_df_corr": data_df_corr})

    print("############# Updated: defined " + timeseries_basket["basket_type"] + " basket #################")
    print("timeseries_basket['data_df_mean'] = ")
    print(timeseries_basket['data_df_mean'])
    print("\n")
    print("timeseries_basket['data_df_stdev'] = ")
    print(timeseries_basket['data_df_stdev'])
    print("\n")
    print("timeseries_basket['data_df_corr'] = ")
    print(timeseries_basket['data_df_corr'])
    print("\n")
    print("timeseries_basket.keys() = " )
    print(timeseries_basket.keys() )
    print("############# End: updated: defined " + timeseries_basket["basket_type"] + " basket #################")

    return timeseries_basket




