
def asset_basket(basket_type,   #"asset"
                 basket_id, #   Basket ID of pre-defined portfolios we want to consider, see below
                 real_or_nominal,     #whether nominal or real historical returns should be used.
                 add_cash_TrueFalse = False  #True if cash needs to be added as an asset
                ):
    
    #OBJECTIVE: Constructs pre-defined asset baskets (groups of assets to incorporate in the portfolio)
    #RETURNS dictionary "asset_basket"

    asset_basket = {}

    #------------------------------------------------------------------------------------------------
    #Pre-defined portfolios; "basket_id" will be the identifier
    
    if basket_id == "Basic_stock_bond":
        #description
        basket_desc = "Basic portfolio for paper: T30, B10 and VWD"

        #SHORT label for e.g. figures
        basket_label = "T30, B10 and VWD"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD"]

    elif basket_id == "2_Factor_plus_basic":
        #description
        basket_desc = "Factor2 portfolio for paper: Basic, size and value"

        #SHORT label for e.g. figures
        basket_label = "Factor2 portfolio"

        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["T30", "B10", "VWD", "Size_Lo30", "Value_Hi30"]
    
    elif basket_id == "5_Factor_plusEWD":
        
        #description
        basket_desc = "5_Factor_plusEWD"
        
        #SHORT label for e.g. figures
        basket_label = "5_Factor_plusEWD"
                
        #timeseries names, ONE FOR EVERY ASSET to be used as identifiers (e.g. column headings PREFIX) for timeseries
        basket_timeseries_names = ["Size_Lo30", "Value_Hi30", "Mom_Hi30", "Vol_Lo20", "Div_Hi30",  "T30", "B10", "VWD", "EWD"]
    

    # ------------------------------------------------------------------------------------------------
    # Add CASH as an asset if needed
    if add_cash_TrueFalse:  #if add_cash_TrueFalse == True, append "Cash" to timeseries names
        basket_timeseries_names.insert(0,"Cash")    #Cash will always be the FIRST asset (asset [0] if it has to be included)

    #Get column names from historical data
    basket_columns =  asset_data_column_names(basket_type, basket_timeseries_names, real_or_nominal,
                                                   returns_or_indices = "returns")

    # ------------------------------------------------------------------------------------------------
    # Construct asset basket
    asset_basket = {"basket_type": basket_type,
                         "basket_id": basket_id,
                         "basket_desc": basket_desc,
                         "basket_label": basket_label,
                         "basket_columns": basket_columns,
                         "basket_timeseries_names": basket_timeseries_names
                         }

    return asset_basket


def asset_data_column_names( basket_type, #"asset"
                                basket_timeseries_names,
                               real_or_nominal,
                               returns_or_indices   #"returns" or "indices"
                                 ):

    #OBJECTIVE: returns data column names associated with timeseries in basket_timeseries_names according to the format:
    # where timeseries is the timeseries name
    # timeseries_nom_ret: timeseries nominal returns
    # timeseries_real_ret: timeseries real returns
    # timeseries_nom_ret_ind: index formed based on timeseries nominal returns
    # timeseries_real_ret_ind: index formed based on timeseries real returns

    #RETURNS: column_names = LIST
    if basket_type == "asset":
        if real_or_nominal == "nominal":
            if returns_or_indices == "returns":
                column_names = [i+"_nom_ret" for i in basket_timeseries_names]
            elif returns_or_indices == "indices":
                column_names = [i + "_nom_ret_ind" for i in basket_timeseries_names]

        elif real_or_nominal == "real":
            if returns_or_indices == "returns":
                column_names = [i+"_real_ret" for i in basket_timeseries_names]
            elif returns_or_indices == "indices":
                column_names = [i + "_real_ret_ind" for i in basket_timeseries_names]

    else:
        raise ValueError("PVS error in asset_data_column_names: only gives conventions for asset return columns.")


    return column_names
