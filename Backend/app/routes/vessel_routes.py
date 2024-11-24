
from io import StringIO
from ..utils.data_processing import get_json_data,get_last_three_year_data, load_vessel_data, det_cat_data, det_subcat_data, get_pd_data, get_dd_subcat,plotly_monthly_quartiles, plotly_yearly_quartiles
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from ..models import user_models
from ..database import get_db
from ..dependencies import get_current_user
from ..schemas import vessel_schemas
from dotenv import load_dotenv
from functools import wraps
import os
import numpy as np
import json
import redis

router = APIRouter()
load_dotenv()
# Retrieve Redis connection settings from environment
redis_host = os.getenv("REDIS_HOST", "localhost")  # Default to 'localhost' if not set
redis_port = os.getenv("REDIS_PORT", "6379")  # Default to '6379' if not set

# Connect to Redis (make sure Redis server is running)
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0, decode_responses=True)


def redis_cache(ttl_seconds: int = 3600):
    """Decorator for caching function results in Redis."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Use a unique key based on the function name and parameters
            cache_key = f"{func.__name__}_{json.dumps(args)}_{json.dumps(kwargs)}"
            
            # Check if result is in cache
            cached_result = redis_client.get(cache_key)
            if cached_result:
                print("Returning cached result")
                return pd.read_json(StringIO(cached_result))

            # Compute the result and cache it
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, ttl_seconds, result.to_json())
            return result
        return wrapper
    return decorator

@router.get("/vessel/vessel_types/")
def get_vessel_types():
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    return vessel_particulars['VESSEL TYPE'].unique().tolist()

@router.get("/vessel/vessel_subtypes/")
def get_vessel_subtypes(vessel_type: str):
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    subtypes = vessel_particulars[vessel_particulars['VESSEL TYPE'] == vessel_type]['VESSEL SUBTYPE'].unique()
    return subtypes.tolist()

@router.post("/vessel/filter_report_data")
def filter_report_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()
    
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()

    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)
    
    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)
    vessels_selected_count=filtered_result['COST_CENTER'].nunique()

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
    
    cat_df = det_cat_data(filtered_result).reset_index()
    subcat_df = det_subcat_data(filtered_result).reset_index()
    
    manning = ['CREW WAGES', 'CREW EXPENSES', 'VICTUALLING EXPENSES']
    tech = ['STORES', 'SPARES', 'REPAIRS & MAINTENANCE', 'MISCELLANEOUS', 'LUBE OIL CONSUMPTION']
    fees = ['MANAGEMENT FEES']
    admin = ['VESSEL BANK CHARGE', 'ADMINISTRATIVE EXPENSES']
    
    budgeted_expenses = manning + tech + fees + admin
    non_budget = ['INSURANCE', 'P&I/H&M EXPENSES', 'CAPITAL EXPENDITURE', 'NON-BUDGETED EXPENSES', 'VOYAGE/CHARTERERS EXPENSES', 'EXTRA ORDINARY ITEMS', 'VESSEL UPGRADING COSTS', 'SHIP SOFTWARE']
    event_cats = ['PRE-DELIVERY EXPENSES', 'DRYDOCKING EXPENSES']
    all_cat = budgeted_expenses + non_budget + event_cats
    
    # Assign order based on the dictionary
    cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    cat_df = cat_df.sort_values(by='order')
    cat_df.drop('order', axis=1, inplace=True)
    subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    subcat_df = subcat_df.sort_values(by='order')
    subcat_df.drop('order', axis=1, inplace=True)


    def group_budget(x):
        if x in manning:
            return "Manning"
        elif x in tech:
            return "Technical"
        elif x in fees:
            return "Management"
        elif x in admin:
            return "Administrative Expenses"
        elif x in non_budget:
            return "ADDITIONAL CATEGORIES"
        elif x in event_cats:
            return "EVENT CATEGORIES"
        
    def split_cats(x):
        x=x.split(",")
        return x[0].strip('(')
    
    ### report view data proccessing
    ## 1.1 Budget category
    budget_cat_df = cat_df[cat_df.CATEGORIES.isin(budgeted_expenses)]
    budget_cat_df = budget_cat_df.copy()

    budget_cat_df.loc[:, 'Header'] = budget_cat_df['CATEGORIES'].apply(group_budget)
    budget_cat_df = budget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
    budget_cat_df['order'] = budget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    budget_cat_df = budget_cat_df.sort_values(by='order')
 

    df2 = budget_cat_df.copy()
    wo_lo = df2[df2.CATEGORIES != 'LUBE OIL CONSUMPTION'].reset_index().set_index((['Header', 'CATEGORIES']))
    wo_lo = wo_lo.sum(numeric_only=True)
    wo_lo = wo_lo.reset_index(name='Total OPEX').transpose().tail(1)
    wo_lo['CATEGORIES'] = 'Without Lube Oil'
    wo_lo = wo_lo.rename(columns={0:'median_50perc_population', 
                            1:'optimal_63perc_population',
                            2:'top_75perc_population',
                            3 : 'order'}).reset_index(names='Header')
 
    opex_per_day = pd.DataFrame()
    opex_per_day['median_50perc_population'] = wo_lo['median_50perc_population']//30
    opex_per_day['optimal_63perc_population'] = wo_lo['optimal_63perc_population']//30
    opex_per_day['top_75perc_population'] = wo_lo['top_75perc_population']//30
    opex_per_day['order'] = 99
    opex_per_day['CATEGORIES'] = 'Without Lube Oil'
    opex_per_day['Header'] = 'OPEX/DAY'
    
    wo_lo = pd.concat([wo_lo, opex_per_day])
    w_lo = df2.copy().reset_index().set_index((['Header', 'CATEGORIES']))
    w_lo = w_lo.sum(numeric_only=True)
    w_lo = w_lo.reset_index(name='Total OPEX').transpose().tail(1)
    w_lo['CATEGORIES'] = 'With Lube Oil'
    
    w_lo = w_lo.rename(columns={0:'median_50perc_population', 
                            1:'optimal_63perc_population',
                            2:'top_75perc_population',
                            3 : 'order'}).reset_index(names='Header')
    
    opex_per_day = pd.DataFrame()
    opex_per_day['median_50perc_population'] = w_lo['median_50perc_population']//30
    opex_per_day['optimal_63perc_population'] = w_lo['optimal_63perc_population']//30
    opex_per_day['top_75perc_population'] = w_lo['top_75perc_population']//30
    opex_per_day['order'] = 100
    opex_per_day['CATEGORIES'] = 'With Lube Oil'
    opex_per_day['Header'] = 'OPEX/DAY'
    
    w_lo = pd.concat([w_lo, opex_per_day])
    total_df = pd.concat([w_lo, wo_lo]).sort_values(by=['order']).reset_index(drop=True)
    budget_cat_df = pd.concat([budget_cat_df.reset_index(), total_df]).sort_values(by='order', ascending=True)
    budget_cat_data = get_json_data(budget_cat_df)  

    def calculate_summary_fields(data):
        for category in data:
            # Check if 'records' field is present and contains multiple entries
            if 'records' in category and len(category['records']) > 1:
                # Initialize total values to 0
                total_median = 0
                total_optimal = 0
                total_top = 0
                # Sum up the values from each record
                for record in category['records']:
                    total_median += record.get('median_50perc_population', 0) or 0
                    total_optimal += record.get('optimal_63perc_population', 0) or 0
                    total_top += record.get('top_75perc_population', 0) or 0
                # Update the main category totals if they were originally None
                category['median_50perc_population'] = total_median if category['median_50perc_population'] is None else category['median_50perc_population']
                category['optimal_63perc_population'] = total_optimal if category['optimal_63perc_population'] is None else category['optimal_63perc_population']
                category['top_75perc_population'] = total_top if category['top_75perc_population'] is None else category['top_75perc_population']
        return data

    budget_cat_data = calculate_summary_fields(budget_cat_data)

 
    ## 1.2 Budget Sub-category
    budget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(budgeted_expenses)]
    budget_subcatdf = budget_subcatdf.copy()
    
    budget_subcatdf.loc[:, 'Header'] = budget_subcatdf['CATEGORIES'].apply(group_budget)
    budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
    budget_subcatdf['order'] = budget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    budget_subcatdf = budget_subcatdf.sort_values(by='order')
    budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
    budget_subcat_data = get_json_data(budget_subcatdf, flag='subcat')
    
    ## 2.1 Additinal category
    nonbudget_cat_df = cat_df[cat_df.CATEGORIES.isin(non_budget)]
    nonbudget_cat_df['Header'] = nonbudget_cat_df['CATEGORIES'].apply(group_budget)
    nonbudget_cat_df = nonbudget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
    nonbudget_cat_df['order'] = nonbudget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    nonbudget_cat_df = nonbudget_cat_df.sort_values(by='order', ascending=True)
    non_budget_cat_data = get_json_data(nonbudget_cat_df)

    ## 2.2 Additinal Sub-category
    nonbudget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(non_budget)]
    nonbudget_subcatdf['Header'] = nonbudget_subcatdf['CATEGORIES'].apply(group_budget)
    nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
    nonbudget_subcatdf['order'] = nonbudget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    nonbudget_subcatdf = nonbudget_subcatdf.sort_values(by='order')
    nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
    nonbudget_subcat_data = get_json_data(nonbudget_subcatdf, flag='subcat')
                
    ### Event Category
    event_cat_df = filtered_result[filtered_result.CATEGORIES.isin(event_cats)].reset_index(drop=True)
    
    df_dd = event_cat_df[event_cat_df['CATEGORIES'] == 'DRYDOCKING EXPENSES'].reset_index(drop=True)
    df_pd = event_cat_df[event_cat_df['CATEGORIES'] == 'PRE-DELIVERY EXPENSES'].reset_index(drop=True)
        
 
    try:
        cat_seg_, subcat_seg_, cat_seg, subcat_seg = get_pd_data(df_pd)
    except Exception as e:
        print(f"Error processing get_dd_subcat.",e)
        dd_subcat_filtered_df, subcat_df_seg = pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

    try:
        dd_subcat_filtered_df, subcat_df_seg = get_dd_subcat(df_dd)
    except Exception as e:
        print(f"Error processing get_dd_subcat.",e)
        dd_subcat_filtered_df, subcat_df_seg = pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames


    numeric_cols = subcat_df_seg.select_dtypes(include='number').columns
    subcat_df_seg[numeric_cols] = subcat_df_seg[numeric_cols].astype(int)
    
    event_subcat_df1 = pd.concat([subcat_seg, subcat_df_seg]).reset_index()

    event_subcat_df = event_subcat_df1.copy()
 
    try:
        # Check if 'CATEGORIES' is in columns
        if 'CATEGORIES' not in event_subcat_df.columns:
            raise KeyError("The 'CATEGORIES' column is missing from the DataFrame.")

        event_subcat_df['Header'] = event_subcat_df['CATEGORIES'].apply(group_budget)
        event_subcatdf = (
            event_subcat_df.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
            .astype('int')
            .reset_index()
        )
        event_subcatdf['order'] = event_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
        event_subcatdf = event_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).reset_index()
        event_subcatdf['Header'] = event_subcatdf['CATEGORIES']
        event_subcatdf['CATEGORIES'] = event_subcatdf['ACCOUNT_CODE'] + "; " + event_subcatdf['SUB_CATEGORIES']

    except KeyError as e:
        print(f"KeyError occurred: {e}")
        # Handle empty DataFrame if needed
        event_subcatdf = pd.DataFrame()  # Return an empty DataFrame or any other default behavior
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


             
    event_subcat_data = get_json_data(event_subcatdf, flag='cat')
            

    
    # # Budget Category processing
    # cat_df = get_cat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(budgeted_expenses)])
    # cat_df['stats_model_optimal_budget'] = cat_df['stats_model_optimal_budget'].astype(int)
    # print("110")
    # cat_df = cat_df.set_index('CATEGORIES').join(budget_cat_df.set_index('CATEGORIES'), lsuffix='_cat_df', rsuffix='_budget_cat_df', how='outer').reset_index()
    # temp = cat_df[~cat_df.CATEGORIES.isin(['With Lube Oil', 'Without Lube Oil'])].reset_index(drop=True)
    
    # w_lo_tot = temp['stats_model_optimal_budget'].sum()
    # wo_lo_tot = temp[temp.CATEGORIES != 'LUBE OIL CONSUMPTION']['stats_model_optimal_budget'].sum()
    
    # conditions = (cat_df['CATEGORIES'] == 'With Lube Oil') & (cat_df['Header'] == 'Total OPEX')
    # cat_df.loc[conditions, 'stats_model_optimal_budget'] = w_lo_tot
    
    # conditions = (cat_df['CATEGORIES'] == 'Without Lube Oil') & (cat_df['Header'] == 'Total OPEX')
    # cat_df.loc[conditions, 'stats_model_optimal_budget'] = wo_lo_tot
    
    # conditions = (cat_df['CATEGORIES'] == 'With Lube Oil') & (cat_df['Header'] == 'OPEX/DAY')
    # cat_df.loc[conditions, 'stats_model_optimal_budget'] = w_lo_tot/30
    
    # conditions = (cat_df['CATEGORIES'] == 'Without Lube Oil') & (cat_df['Header'] == 'OPEX/DAY')
    # cat_df.loc[conditions, 'stats_model_optimal_budget'] = wo_lo_tot/30
    
    # budget_cat_df = cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
    # budget_cat_df = budget_cat_df.sort_values(by='order', ascending=True)
    # budget_cat_data = get_json_data(budget_cat_df)

    # subcat_df = get_subcat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(budgeted_expenses)])
    # subcat_df['CATEGORIES'] = subcat_df.reset_index()['SUB CATEGORIES'].apply(split_cats)
    
    # subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    # subcat_df = subcat_df.sort_values(by='order')
    # subcat_df['stats_model_optimal_budget'] = subcat_df['stats_model_optimal_budget'].astype(int)
    # budget_subcatdf = budget_subcatdf.reset_index()
    
    # scats = []
    # for index, row in budget_subcatdf[['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']].iterrows():
    #     scats.append("('{}', '{}', '{}')".format(row['CATEGORIES'], row['ACCOUNT_CODE'], row['SUB_CATEGORIES']))
    
    # budget_subcatdf['SUB CATEGORIES'] = pd.Series(scats)
    # budget_subcatdf = budget_subcatdf.merge(subcat_df, on='SUB CATEGORIES')
    # budget_subcatdf = budget_subcatdf.rename(columns={'SUB CATEGORIES' : 'CATEGORIES'})
    # budget_subcat_data = get_json_data(budget_subcatdf)
    # print("22")
    # # Optional Category processing
    # cat_df = get_cat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(non_budget)])
    # cat_df['stats_model_optimal_budget'] = cat_df['stats_model_optimal_budget'].astype(int)
    # cat_df = cat_df.merge(nonbudget_cat_df, on='CATEGORIES')
    # cat_df['Header'] = cat_df.CATEGORIES.apply(group_budget)
    # cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    # cat_df = cat_df.sort_values(by='order')
    # nonbudget_cat_df = cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')    
    # nonbudget_cat_data = get_json_data(nonbudget_cat_df)
    
    # subcat_df = get_subcat_optimal_mean(filtered_result[filtered_result.CATEGORIES.isin(non_budget)])
    # subcat_df['stats_model_optimal_budget'] = subcat_df['stats_model_optimal_budget'].astype(int)
    # nonbudget_subcatdf = nonbudget_subcatdf.reset_index()
    
    # scats = []
    # for index, row in nonbudget_subcatdf[['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']].iterrows():
    #     scats.append("('{}', '{}', '{}')".format(row['CATEGORIES'], row['ACCOUNT_CODE'], row['SUB_CATEGORIES']))
    
    # nonbudget_subcatdf['SUB CATEGORIES'] = pd.Series(scats)
    # nonbudget_subcatdf = nonbudget_subcatdf.merge(subcat_df, on='SUB CATEGORIES')
    # nonbudget_subcatdf = nonbudget_subcatdf.rename(columns={'SUB CATEGORIES' : 'CATEGORIES'})
    
    # nonbudget_subcat_data = get_json_data(nonbudget_subcatdf)

    # print("33")
    # # Event Categories proccessing
    # event_DF = pd.concat([df_dd, df_pd])  
    # subcat_df = get_event_subcat_optimal_mean(event_DF.rename(columns={'Expense' : 'EXPENSE'}))
    # subcat_df['stats_model_optimal_budget'] = subcat_df['stats_model_optimal_budget'].astype(int)
    # subcat_df['ACCOUNT_CODE'] = subcat_df['SUB CATEGORIES'].apply(lambda x: x.strip().split(";")[1])

    # subcat_df = pd.merge(event_subcatdf, subcat_df, on='ACCOUNT_CODE', how='inner').drop(columns=['order'])
    
    # # Assuming you know the column names or have a way to identify them
    # numeric_cols = [col for col in subcat_df.columns if subcat_df[col].dtype in ['int32', 'float32']]
    # numeric_sum = subcat_df[numeric_cols].sum()
    # numeric_sum['Header']='TOTAL EVENT EXPENSES'

    # event_subcat_data = get_json_data(subcat_df, flag='cat')
    # event_subcat_data.append(numeric_sum.to_dict())

    ### Trend Analysis
    plotly_monthly_quartiles_data= plotly_monthly_quartiles(filtered_result)
    plotly_yearly_quartiles_data= plotly_yearly_quartiles(filtered_result)

    # new_modified_data=new_modified.get_optimal_mean(filtered_result)
    # print("77")
    # whole_year_data=whole_year.get_optimal_mean(filtered_result)
    # print("88")


    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "vessel_cat_options": handle_nan_inf(convert_numpy_data(vessel_cat_options)),
        "vessel_subcat_options": handle_nan_inf(convert_numpy_data(vessel_subcat_options)),
        "selected_vessels_option": handle_nan_inf(convert_numpy_data(selected_vessels_option)),
        "vessels_selected_count":handle_nan_inf(convert_numpy_data(vessels_selected_count)),
        "budget_cat_data": handle_nan_inf(convert_numpy_data(budget_cat_data)),
        "budget_subcat_data": handle_nan_inf(convert_numpy_data(budget_subcat_data)),
        "non_budget_cat_data": handle_nan_inf(convert_numpy_data(non_budget_cat_data)),
        "nonbudget_subcat_data": handle_nan_inf(convert_numpy_data(nonbudget_subcat_data)),
        "event_subcat_data": handle_nan_inf(convert_numpy_data(event_subcat_data)),
        "plotly_monthly_quartiles_data": handle_nan_inf(convert_numpy_data(plotly_monthly_quartiles_data)),
        "plotly_yearly_quartiles_data": handle_nan_inf(convert_numpy_data(plotly_yearly_quartiles_data))
 
    }

    return response



@router.post("/vessel/inputs")
def filter_report_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()
    
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()

    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)

    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)
    vessels_selected_count=filtered_result['COST_CENTER'].nunique()

    # Group by Vessel Age and count unique vessels (based on 'VESSEL NAME' or another unique identifier)
    vessel_age_counts = merged_df.groupby('VESSEL AGE')['COST_CENTER'].nunique().reset_index(name='COUNT')

    # Filter out only even ages
    vessel_age_counts = vessel_age_counts[vessel_age_counts['VESSEL AGE'] % 2 == 0]

    # Convert the result to a dictionary to return as JSON, only for even ages
    age_count_data = {
        'x': vessel_age_counts['VESSEL AGE'].tolist(),  # X-axis: Only even ages of vessels
        'y': vessel_age_counts['COUNT'].tolist()  }# Y-axis: Count of unique vessels per age

    
    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "vessel_cat_options": handle_nan_inf(convert_numpy_data(vessel_cat_options)),
        "vessel_subcat_options": handle_nan_inf(convert_numpy_data(vessel_subcat_options)),
        "selected_vessels_option": handle_nan_inf(convert_numpy_data(selected_vessels_option)),
        "vessels_selected_count":handle_nan_inf(convert_numpy_data(vessels_selected_count)),
        "age_count_data": handle_nan_inf(convert_numpy_data(age_count_data))
    }

    return response


@router.post("/vessel/filter_budget_report_data")
def filter_budget_report_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()


    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)
    
    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
    
    cat_df = det_cat_data(filtered_result).reset_index()
    subcat_df = det_subcat_data(filtered_result).reset_index()
    
    manning = ['CREW WAGES', 'CREW EXPENSES', 'VICTUALLING EXPENSES']
    tech = ['STORES', 'SPARES', 'REPAIRS & MAINTENANCE', 'MISCELLANEOUS', 'LUBE OIL CONSUMPTION']
    fees = ['MANAGEMENT FEES']
    admin = ['VESSEL BANK CHARGE', 'ADMINISTRATIVE EXPENSES']
    
    budgeted_expenses = manning + tech + fees + admin
    non_budget = ['INSURANCE', 'P&I/H&M EXPENSES', 'CAPITAL EXPENDITURE', 'NON-BUDGETED EXPENSES', 'VOYAGE/CHARTERERS EXPENSES', 'EXTRA ORDINARY ITEMS', 'VESSEL UPGRADING COSTS', 'SHIP SOFTWARE']
    event_cats = ['PRE-DELIVERY EXPENSES', 'DRYDOCKING EXPENSES']
    all_cat = budgeted_expenses + non_budget + event_cats
    
    # Assign order based on the dictionary
    cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    cat_df = cat_df.sort_values(by='order')
    cat_df.drop('order', axis=1, inplace=True)
    subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    subcat_df = subcat_df.sort_values(by='order')
    subcat_df.drop('order', axis=1, inplace=True)


    def group_budget(x):
        if x in manning:
            return "Manning"
        elif x in tech:
            return "Technical"
        elif x in fees:
            return "Management"
        elif x in admin:
            return "Administrative Expenses"
        elif x in non_budget:
            return "ADDITIONAL CATEGORIES"
        elif x in event_cats:
            return "EVENT CATEGORIES"
        
    ### report view data proccessing
    ## 1.1 Budget category
    budget_cat_df = cat_df[cat_df.CATEGORIES.isin(budgeted_expenses)]
    budget_cat_df = budget_cat_df.copy()

    budget_cat_df.loc[:, 'Header'] = budget_cat_df['CATEGORIES'].apply(group_budget)
    budget_cat_df = budget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
    budget_cat_df['order'] = budget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    budget_cat_df = budget_cat_df.sort_values(by='order')
 

    df2 = budget_cat_df.copy()
    wo_lo = df2[df2.CATEGORIES != 'LUBE OIL CONSUMPTION'].reset_index().set_index((['Header', 'CATEGORIES']))
    wo_lo = wo_lo.sum(numeric_only=True)
    wo_lo = wo_lo.reset_index(name='Total OPEX').transpose().tail(1)
    wo_lo['CATEGORIES'] = 'Without Lube Oil'
    wo_lo = wo_lo.rename(columns={0:'median_50perc_population', 
                            1:'optimal_63perc_population',
                            2:'top_75perc_population',
                            3 : 'order'}).reset_index(names='Header')
 
    opex_per_day = pd.DataFrame()
    opex_per_day['median_50perc_population'] = wo_lo['median_50perc_population']//30
    opex_per_day['optimal_63perc_population'] = wo_lo['optimal_63perc_population']//30
    opex_per_day['top_75perc_population'] = wo_lo['top_75perc_population']//30
    opex_per_day['order'] = 99
    opex_per_day['CATEGORIES'] = 'Without Lube Oil'
    opex_per_day['Header'] = 'OPEX/DAY'
    
    wo_lo = pd.concat([wo_lo, opex_per_day])
    w_lo = df2.copy().reset_index().set_index((['Header', 'CATEGORIES']))
    w_lo = w_lo.sum(numeric_only=True)
    w_lo = w_lo.reset_index(name='Total OPEX').transpose().tail(1)
    w_lo['CATEGORIES'] = 'With Lube Oil'
    
    w_lo = w_lo.rename(columns={0:'median_50perc_population', 
                            1:'optimal_63perc_population',
                            2:'top_75perc_population',
                            3 : 'order'}).reset_index(names='Header')
    
    opex_per_day = pd.DataFrame()
    opex_per_day['median_50perc_population'] = w_lo['median_50perc_population']//30
    opex_per_day['optimal_63perc_population'] = w_lo['optimal_63perc_population']//30
    opex_per_day['top_75perc_population'] = w_lo['top_75perc_population']//30
    opex_per_day['order'] = 100
    opex_per_day['CATEGORIES'] = 'With Lube Oil'
    opex_per_day['Header'] = 'OPEX/DAY'
    
    w_lo = pd.concat([w_lo, opex_per_day])
    total_df = pd.concat([w_lo, wo_lo]).sort_values(by=['order']).reset_index(drop=True)
    budget_cat_df = pd.concat([budget_cat_df.reset_index(), total_df]).sort_values(by='order', ascending=True)
    budget_cat_data = get_json_data(budget_cat_df)  

    def calculate_summary_fields(data):
        for category in data:
            # Check if 'records' field is present and contains multiple entries
            if 'records' in category and len(category['records']) > 1:
                # Initialize total values to 0
                total_median = 0
                total_optimal = 0
                total_top = 0
                # Sum up the values from each record
                for record in category['records']:
                    total_median += record.get('median_50perc_population', 0) or 0
                    total_optimal += record.get('optimal_63perc_population', 0) or 0
                    total_top += record.get('top_75perc_population', 0) or 0
                # Update the main category totals if they were originally None
                category['median_50perc_population'] = total_median if category['median_50perc_population'] is None else category['median_50perc_population']
                category['optimal_63perc_population'] = total_optimal if category['optimal_63perc_population'] is None else category['optimal_63perc_population']
                category['top_75perc_population'] = total_top if category['top_75perc_population'] is None else category['top_75perc_population']
        return data

    budget_cat_data = calculate_summary_fields(budget_cat_data)

 
    ## 1.2 Budget Sub-category
    budget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(budgeted_expenses)]
    budget_subcatdf = budget_subcatdf.copy()

    budget_subcatdf.loc[:, 'Header'] = budget_subcatdf['CATEGORIES'].apply(group_budget)
    budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
    budget_subcatdf['order'] = budget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    budget_subcatdf = budget_subcatdf.sort_values(by='order')
    budget_subcatdf = budget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
    budget_subcat_data = get_json_data(budget_subcatdf, flag='subcat')
    
    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "budget_cat_data": handle_nan_inf(convert_numpy_data(budget_cat_data)),
        "budget_subcat_data": handle_nan_inf(convert_numpy_data(budget_subcat_data))
    }

    return response




@router.post("/vessel/filter_nonbudget_report_data")
def filter_nonbudget_report_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()


    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)
    
    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
    
    cat_df = det_cat_data(filtered_result).reset_index()
    subcat_df = det_subcat_data(filtered_result).reset_index()
    
    manning = ['CREW WAGES', 'CREW EXPENSES', 'VICTUALLING EXPENSES']
    tech = ['STORES', 'SPARES', 'REPAIRS & MAINTENANCE', 'MISCELLANEOUS', 'LUBE OIL CONSUMPTION']
    fees = ['MANAGEMENT FEES']
    admin = ['VESSEL BANK CHARGE', 'ADMINISTRATIVE EXPENSES']
    
    budgeted_expenses = manning + tech + fees + admin
    non_budget = ['INSURANCE', 'P&I/H&M EXPENSES', 'CAPITAL EXPENDITURE', 'NON-BUDGETED EXPENSES', 'VOYAGE/CHARTERERS EXPENSES', 'EXTRA ORDINARY ITEMS', 'VESSEL UPGRADING COSTS', 'SHIP SOFTWARE']
    event_cats = ['PRE-DELIVERY EXPENSES', 'DRYDOCKING EXPENSES']
    all_cat = budgeted_expenses + non_budget + event_cats
    
    # Assign order based on the dictionary
    cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    cat_df = cat_df.sort_values(by='order')
    cat_df.drop('order', axis=1, inplace=True)
    subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    subcat_df = subcat_df.sort_values(by='order')
    subcat_df.drop('order', axis=1, inplace=True)


    def group_budget(x):
        if x in manning:
            return "Manning"
        elif x in tech:
            return "Technical"
        elif x in fees:
            return "Management"
        elif x in admin:
            return "Administrative Expenses"
        elif x in non_budget:
            return "ADDITIONAL CATEGORIES"
        elif x in event_cats:
            return "EVENT CATEGORIES"
        

    ### report view data proccessing

    ## 2.1 Additinal category
    nonbudget_cat_df = cat_df[cat_df.CATEGORIES.isin(non_budget)]
    nonbudget_cat_df['Header'] = nonbudget_cat_df['CATEGORIES'].apply(group_budget)
    nonbudget_cat_df = nonbudget_cat_df.set_index(['Header', 'CATEGORIES']).astype('int').reset_index().set_index('Header')
    nonbudget_cat_df['order'] = nonbudget_cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    nonbudget_cat_df = nonbudget_cat_df.sort_values(by='order', ascending=True)
    non_budget_cat_data = get_json_data(nonbudget_cat_df)

    ## 2.2 Additinal Sub-category
    nonbudget_subcatdf = subcat_df[subcat_df.CATEGORIES.isin(non_budget)]
    nonbudget_subcatdf['Header'] = nonbudget_subcatdf['CATEGORIES'].apply(group_budget)
    nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).astype('int').reset_index()
    nonbudget_subcatdf['order'] = nonbudget_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    nonbudget_subcatdf = nonbudget_subcatdf.sort_values(by='order')
    nonbudget_subcatdf = nonbudget_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
    nonbudget_subcat_data = get_json_data(nonbudget_subcatdf, flag='subcat')
                

    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "non_budget_cat_data": handle_nan_inf(convert_numpy_data(non_budget_cat_data)),
        "nonbudget_subcat_data": handle_nan_inf(convert_numpy_data(nonbudget_subcat_data))
    }

    return response




@router.post("/vessel/filter_event_report_data")
def filter_event_report_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()


    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)
    
    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
    
    cat_df = det_cat_data(filtered_result).reset_index()
    subcat_df = det_subcat_data(filtered_result).reset_index()
    
    manning = ['CREW WAGES', 'CREW EXPENSES', 'VICTUALLING EXPENSES']
    tech = ['STORES', 'SPARES', 'REPAIRS & MAINTENANCE', 'MISCELLANEOUS', 'LUBE OIL CONSUMPTION']
    fees = ['MANAGEMENT FEES']
    admin = ['VESSEL BANK CHARGE', 'ADMINISTRATIVE EXPENSES']
    
    budgeted_expenses = manning + tech + fees + admin
    non_budget = ['INSURANCE', 'P&I/H&M EXPENSES', 'CAPITAL EXPENDITURE', 'NON-BUDGETED EXPENSES', 'VOYAGE/CHARTERERS EXPENSES', 'EXTRA ORDINARY ITEMS', 'VESSEL UPGRADING COSTS', 'SHIP SOFTWARE']
    event_cats = ['PRE-DELIVERY EXPENSES', 'DRYDOCKING EXPENSES']
    all_cat = budgeted_expenses + non_budget + event_cats
    
    # Assign order based on the dictionary
    cat_df['order'] = cat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    cat_df = cat_df.sort_values(by='order')
    cat_df.drop('order', axis=1, inplace=True)
    subcat_df['order'] = subcat_df['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
    subcat_df = subcat_df.sort_values(by='order')
    subcat_df.drop('order', axis=1, inplace=True)


    def group_budget(x):
        if x in manning:
            return "Manning"
        elif x in tech:
            return "Technical"
        elif x in fees:
            return "Management"
        elif x in admin:
            return "Administrative Expenses"
        elif x in non_budget:
            return "ADDITIONAL CATEGORIES"
        elif x in event_cats:
            return "EVENT CATEGORIES"
    
    ### report view data proccessing

    ### Event Category
    event_cat_df = filtered_result[filtered_result.CATEGORIES.isin(event_cats)].reset_index(drop=True)
    
    df_dd = event_cat_df[event_cat_df['CATEGORIES'] == 'DRYDOCKING EXPENSES'].reset_index(drop=True)
    df_pd = event_cat_df[event_cat_df['CATEGORIES'] == 'PRE-DELIVERY EXPENSES'].reset_index(drop=True)
        
 
    try:
        cat_seg_, subcat_seg_, cat_seg, subcat_seg = get_pd_data(df_pd)
    except Exception as e:
        print(f"Error processing get_dd_subcat.",e)
        dd_subcat_filtered_df, subcat_df_seg = pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames

    try:
        dd_subcat_filtered_df, subcat_df_seg = get_dd_subcat(df_dd)
    except Exception as e:
        print(f"Error processing get_dd_subcat.",e)
        dd_subcat_filtered_df, subcat_df_seg = pd.DataFrame(), pd.DataFrame()  # Return empty DataFrames


    numeric_cols = subcat_df_seg.select_dtypes(include='number').columns
    subcat_df_seg[numeric_cols] = subcat_df_seg[numeric_cols].astype(int)
    
    event_subcat_df1 = pd.concat([subcat_seg, subcat_df_seg]).reset_index()

    event_subcat_df = event_subcat_df1.copy()
 
    try:
        # Check if 'CATEGORIES' is in columns
        if 'CATEGORIES' not in event_subcat_df.columns:
            raise KeyError("The 'CATEGORIES' column is missing from the DataFrame.")

        event_subcat_df['Header'] = event_subcat_df['CATEGORIES'].apply(group_budget)
        event_subcatdf = (
            event_subcat_df.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])
            .astype('int')
            .reset_index()
        )
        event_subcatdf['order'] = event_subcatdf['CATEGORIES'].apply(lambda x: all_cat.index(x) if x in all_cat else len(all_cat))
        event_subcatdf = event_subcatdf.set_index(['Header', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).reset_index()
        event_subcatdf['Header'] = event_subcatdf['CATEGORIES']
        event_subcatdf['CATEGORIES'] = event_subcatdf['ACCOUNT_CODE'] + "; " + event_subcatdf['SUB_CATEGORIES']

    except KeyError as e:
        print(f"KeyError occurred: {e}")
        # Handle empty DataFrame if needed
        event_subcatdf = pd.DataFrame()  # Return an empty DataFrame or any other default behavior
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
     
    event_subcat_data = get_json_data(event_subcatdf, flag='cat')
            

    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "event_subcat_data": handle_nan_inf(convert_numpy_data(event_subcat_data))
    }

    return response




@router.post("/vessel/filter_trend_data")
def filter_trend_data(params: vessel_schemas.VesselFilterParams, db: Session = Depends(get_db), current_user: user_models.User = Depends(get_current_user)):
    last_3_years = get_last_three_year_data()

    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    vessel_particulars = load_vessel_data(VESSEL_PARTICULARS_URL)
    merged_df = pd.merge(last_3_years, vessel_particulars[['COST_CENTER', 'VESSEL NAME', 'BUILD YEAR']], on='COST_CENTER', how='left')
    
    # Add VESSEL AGE to merged_df
    merged_df['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['VESSEL AGE'] = pd.to_datetime(merged_df['DATE']).dt.year - merged_df['BUILD YEAR']
    last_3_years['CATEGORIES'] = last_3_years['CATEGORIES'].str.upper()
    merged_df['CATEGORIES'] = merged_df['CATEGORIES'].str.upper()


    # Filter DataFrame based on slicer values
    @redis_cache(ttl_seconds=3600)
    def filter_dataframe(vessel_type, vessel_subtype, vessel_age_start, vessel_age_end):

        if isinstance(vessel_type, tuple):
            vessel_type = vessel_type[0]  # Adjust this based on your needs
    
        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))
        ]

        filtered_df = vessel_particulars[
            (vessel_particulars['VESSEL TYPE'] == vessel_type) &
            (vessel_particulars['VESSEL SUBTYPE'].isin(vessel_subtype))]
        
        selected_vessels = merged_df[merged_df['COST_CENTER'].isin(filtered_df['COST_CENTER'].unique())].reset_index(drop=True)
        selected_vessels = selected_vessels[selected_vessels['VESSEL AGE'].between(vessel_age_start, vessel_age_end)]
        selected_vessels = selected_vessels.groupby(['PERIOD', 'COST_CENTER', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
        return selected_vessels
    
    # Call the function with a tuple instead of a list
    selected_vessels = filter_dataframe(
        params.vessel_type,
        tuple(params.vessel_subtype),
        params.vessel_age_start,
        params.vessel_age_end
    )
    # selected_vessels=filter_dataframe(params.vessel_type,params.vessel_subtype,params.vessel_age_start,params.vessel_age_end)
    
    # formatting the selected vessels
    filtered_result1=selected_vessels
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-999')]
    filtered_result1 = filtered_result1[~filtered_result1['ACCOUNT_CODE'].str.contains('.*-998')]

     # Categories field
    vessel_cat_options = filtered_result1['CATEGORIES'].unique()
    vessel_cat_options = ['Select All'] + vessel_cat_options.tolist()
    
    
    if 'Select All' in params.vessel_cat:
        vessel_cat = vessel_cat_options[1:]
    else:
        vessel_cat = params.vessel_cat
        
    filtered_result1 = filtered_result1[filtered_result1['CATEGORIES'].isin(vessel_cat)]
    # Sub Categories field
    vessel_subcat_options = filtered_result1['SUB_CATEGORIES'].unique()
    vessel_subcat_options = ['Select All'] + vessel_subcat_options.tolist()
     
    if 'Select All' in params.vessel_subcat:
        vessel_subcat = vessel_subcat_options[1:]
    else:
        vessel_subcat=params.vessel_subcat
        
    # Filter by selected subcategories
    filtered_result = filtered_result1[filtered_result1['SUB_CATEGORIES'].isin(vessel_subcat)].reset_index(drop=True)

    # Selected Vessels Dropdown
    selected_vessels = filtered_result['VESSEL NAME'].unique()
    selected_vessels_option = ['Select All'] + selected_vessels.tolist()

    # Filter DataFrame based on selected vessels
    if 'Select All' in params.selected_vessels_dropdown:
        selected_vessels_dropdown = selected_vessels_option[1:]
    else:
        selected_vessels_dropdown = params.selected_vessels_dropdown
    
    filtered_result = filtered_result[filtered_result['VESSEL NAME'].isin(selected_vessels_dropdown)].reset_index(drop=True)

    filtered_result['DATE'] = pd.to_datetime(filtered_result['PERIOD'], format='%Y%m').dt.date
    filtered_result['YEAR'] = pd.to_datetime(filtered_result['DATE']).dt.year.astype('str')
            

    ### Trend Analysis
    plotly_monthly_quartiles_data= plotly_monthly_quartiles(filtered_result)
    plotly_yearly_quartiles_data= plotly_yearly_quartiles(filtered_result)


    def convert_numpy_data(data):
        """Helper function to recursively convert numpy objects to native Python types."""
        if isinstance(data, np.float32):
            return float(data)
        if isinstance(data, np.ndarray):
            return data.tolist()
        if isinstance(data, dict):
            return {k: convert_numpy_data(v) for k, v in data.items()}
        if isinstance(data, list):
            return [convert_numpy_data(item) for item in data]
        return data  # Return data as-is if it's not numpy
    
    def handle_nan_inf(data):
        """Convert NaN and infinite values to None."""
        if isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
            return None
        if isinstance(data, dict):
            return {k: handle_nan_inf(v) for k, v in data.items()}
        if isinstance(data, list):
            return [handle_nan_inf(item) for item in data]
        return data


    # Example usage in your response
    response = {
        "plotly_monthly_quartiles_data": handle_nan_inf(convert_numpy_data(plotly_monthly_quartiles_data)),
        "plotly_yearly_quartiles_data": handle_nan_inf(convert_numpy_data(plotly_yearly_quartiles_data))
    }

    return response
