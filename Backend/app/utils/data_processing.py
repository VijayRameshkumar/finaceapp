from io import StringIO
import json
import numpy as np
import pandas as pd
import redis
from ..utils.get_data import get_expense_data
from datetime import timedelta
from dotenv import load_dotenv
import os
from concurrent.futures import ThreadPoolExecutor

# Connect to Redis (make sure Redis server is running)
load_dotenv()

# Retrieve Redis connection settings from environment
redis_host = os.getenv("REDIS_HOST", "localhost")  # Default to 'localhost' if not set
redis_port = os.getenv("REDIS_PORT", "6379")  # Default to '6379' if not set

# Connect to Redis (make sure Redis server is running)
redis_client = redis.StrictRedis(host=redis_host, port=redis_port, db=0, decode_responses=True)


def get_cache_key(file_path: str) -> str:
    """Generate a unique cache key based on the file path."""
    return f"vessel_data:{file_path}"

def load_vessel_data(file_path: str) -> pd.DataFrame:
    """Load vessel data from an Excel file, using Redis for caching."""
    try:
        # Check if data is in cache
        cache_key = get_cache_key(file_path)
        cached_data = redis_client.get(cache_key)

        if cached_data:
            print("Data loaded from cache")
            # Return the cached data as a DataFrame
            return pd.read_json(StringIO(cached_data))
        else:
            print("Data loaded from file")
            # Load the data from the Excel file
            df = pd.read_excel(file_path)

            # Cache the data in Redis (convert DataFrame to JSON before storing)
            redis_client.set(cache_key, df.to_json())

            return df
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def get_last_three_year_data():
    """Retrieve expense data for the last three years with Redis caching."""
    try:
        # Define a cache key
        cache_key = "expense_data:last_three_years"

        # Check if data is already in cache
        cached_data = redis_client.get(cache_key)

        if cached_data:
            print("Data loaded from cache")
            # Load the cached data as a DataFrame
            return pd.read_json(StringIO(cached_data))

        # If data is not cached, fetch it from get_expense_data
        print("Data loaded from function")
        data = get_expense_data()

        # Cache the result in Redis (convert DataFrame to JSON before storing)
        redis_client.setex(cache_key, timedelta(hours=1), data.to_json())

        return data

    except Exception as e:
        raise Exception(f"Error retrieving last three years' data: {e}")
    

def get_cat_quartiles(filtered_result1):
    # Filter out rows with Expense <= 0
    filtered_result1 = filtered_result1[filtered_result1['Expense'] > 0]
    
    # Group by PERIOD, VESSEL NAME, CATEGORIES and sum Expenses
    monthly_data = filtered_result1.groupby(['PERIOD', 'VESSEL NAME', 'CATEGORIES'])['Expense'].sum()
    
    # Calculate quartiles and other statistics
    monthly_data = monthly_data.groupby(['PERIOD', 'CATEGORIES']).agg(
        q1=lambda x: np.quantile(x, 0.25),
        q2=lambda x: np.quantile(x, 0.50),
        median=lambda x: np.quantile(x, 0.63),
        q3=lambda x: np.quantile(x, 0.75)
    )
    
    # Reset index to flatten the DataFrame
    percentiles = monthly_data.reset_index()
    
    # Group by CATEGORIES and calculate population percentiles
    percentiles = percentiles.groupby('CATEGORIES').agg(
        median_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    
    return percentiles

 
def det_cat_data(fr):
    # Call the get_cat_quartiles function to process the data
    df = get_cat_quartiles(fr)
    return df
 
 
def get_subcat_quartiles(filtered_result1):
    filtered_result1 = filtered_result1[filtered_result1.Expense > 0]
    monthly_data = filtered_result1.groupby(['PERIOD', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum()
    # Calculate percentiles for each month
    monthly_data = filtered_result1.groupby(['PERIOD', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].agg(
        q1=lambda x: np.quantile(x, 0.25),
        q2=lambda x: np.quantile(x, 0.50),
        median=lambda x: np.quantile(x, 0.63),
        q3=lambda x: np.quantile(x, 0.75)
    )
    
    # Extract dates and percentiles for plotting
    percentiles = monthly_data.reset_index()
    
    percentiles = percentiles.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=('q2', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('median', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('q3', lambda x: np.quantile(x, 0.75))
    )
    return percentiles

 
def det_subcat_data(fr):
    df = get_subcat_quartiles(fr)
    return df

def make_unique(columns):
    seen = {}
    new_columns = []
    for col in columns:
        if col in seen:
            seen[col] += 1
            new_columns.append(f"{col}.{seen[col]}")
        else:
            seen[col] = 0
            new_columns.append(col)
    return new_columns

def get_json_data(cat_df,order = ['Manning', 'Technical', 'Management', 'Administative Expenses'], flag='cat'):
    # Check if cat_df is empty
    if cat_df.empty:
        print("Warning: Input DataFrame 'cat_df' is empty.")
        return [] 
    if flag == 'cat':
        cat_df['median_50perc_population'] = cat_df['median_50perc_population'].astype(int)
        cat_df['optimal_63perc_population'] = cat_df['optimal_63perc_population'].astype(int)
        cat_df['top_75perc_population'] = cat_df['top_75perc_population'].astype(int)
        
        grp_tot_cat = cat_df.groupby(['Header']).sum().reset_index()
        
        condition = grp_tot_cat["Header"].isin(['Total OPEX', 'OPEX/DAY'])
        
        if condition.shape[0] > 0:
            grp_tot_cat.loc[condition, 'stats_model_optimal_budget'] = None
            grp_tot_cat.loc[condition, 'median_50perc_population'] = None
            grp_tot_cat.loc[condition, 'optimal_63perc_population'] = None
            grp_tot_cat.loc[condition, 'top_75perc_population'] = None
        
        grp_tot_cat['order'] = grp_tot_cat['Header'].apply(lambda x: order.index(x) if x in order else len(order))
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        grp_tot_cat = grp_tot_cat.set_index('Header')        
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        # Check for duplicate columns and rename them if necessary
        grp_tot_cat.columns = make_unique(grp_tot_cat.columns)
        
        grp_tot_cat = grp_tot_cat.T.to_json()
        
        ## cate vise sum
        json_data = {}
        
        # Group by 'Header' column
        grouped = cat_df.groupby('Header')
        
        # Iterate over groups
        for group_name, group_data in grouped:
            group_json = group_data.to_dict(orient='records')
            json_data[group_name] = group_json
            # json_output = json.dumps(json_data, indent=4)
        
        grp_tot_cat = json.loads(grp_tot_cat)
        for key in grp_tot_cat.keys():
            grp_tot_cat[key].update({"records": json_data[key]})
            grp_tot_cat[key].update({"Header": key})                
            
        
        data = []
        for key, value in grp_tot_cat.items():
            data.append(grp_tot_cat[key])
    else:
        grp_tot_cat = cat_df.reset_index().groupby(['Header']).sum().reset_index()
        grp_tot_cat['order'] = grp_tot_cat['Header'].apply(lambda x: order.index(x) if x in order else len(order))
        grp_tot_cat = grp_tot_cat.sort_values(by='order', ascending=True)
        # grp_tot_cat.drop('order', axis=1, inplace=True)
        grp_tot_cat = grp_tot_cat.set_index('Header')
        grp_tot_cat.columns = make_unique(grp_tot_cat.columns)
        
        grp_tot_cat = grp_tot_cat.T.to_json()
        
        ## cate vise sum
        json_data = {}
        cat_df = cat_df.reset_index()
        # Concatenate the columns into a new column 'CATEGORIES'
        cat_df['CATEGORIES'] = cat_df['CATEGORIES'] + ': (' + cat_df['ACCOUNT_CODE'] + ', ' + cat_df['SUB_CATEGORIES'] + ')'
        cat_df = cat_df.drop(columns=['ACCOUNT_CODE', 'SUB_CATEGORIES'])
        
        # Group by 'Header' column
        grouped = cat_df.groupby('Header')
        
        # Iterate over groups
        for group_name, group_data in grouped:
            group_json = group_data.to_dict(orient='records')
            json_data[group_name] = group_json
            json_output = json.dumps(json_data, indent=4)
        
        grp_tot_cat = json.loads(grp_tot_cat)
        for key in grp_tot_cat.keys():
            grp_tot_cat[key].update({"records": json_data[key]})
            grp_tot_cat[key].update({"Header": key})                
            
        
        data = []
        for key, value in grp_tot_cat.items():
            data.append(grp_tot_cat[key])
    # st.json(data)
    return data


def get_dd_cat(DF_DD):
    cost_centers = []
    expenses = []
    
    df_dd_cat = DF_DD.groupby(['VESSEL NAME', 'PERIOD', 'CATEGORIES']).Expense.sum().reset_index()
    df_dd_cat['DATE'] = df_dd_cat['PERIOD'].astype('str').apply(lambda x: f"{x[:4]}-{x[4:]}-01" if x else None)
    # df_dd_cat.groupby(['COST_CENTER', 'CATEGORIES', 'PERIOD', 'DATE'])['AMOUNT_USD'].sum().reset_index()
    cat_seg = df_dd_cat.groupby(['VESSEL NAME', 'CATEGORIES']).apply(func)

    for rec in cat_seg.reset_index(name='daterange').itertuples():
        cc = rec[1]
        for dd in rec[3]:
            temp = df_dd_cat[(df_dd_cat['VESSEL NAME'] == cc) & (df_dd_cat.DATE >= pd.to_datetime(dd[0]).strftime("%Y-%m-%d")) & (df_dd_cat.DATE <= pd.to_datetime(dd[1]).strftime("%Y-%m-%d"))]
            cost_centers.append(cc)
            expenses.append(temp.Expense.sum())
            
    cat_seg_event = pd.DataFrame()
    cat_seg_event['VESSEL NAME'] = pd.Series(cost_centers)
    cat_seg_event['EXPENSE'] = pd.Series(expenses)
    
 
    # filtered_df = cat_seg_event
    filtered_df = cat_seg_event[cat_seg_event.EXPENSE > 0.00]
    q1 = int(filtered_df['EXPENSE'].quantile(0.25))
    q2 = int(filtered_df['EXPENSE'].quantile(0.50))  # This is the median
    q3 = int(filtered_df['EXPENSE'].quantile(0.75))
    

    # Create a DataFrame with quartile values
    return filtered_df, pd.DataFrame({'Quartile': ['CATEGORIES', 'median_50perc_population', 'optimal_63perc_population', 'top_75perc_population'],
                                'Value': [rec[2], q1, q2, q3]}).set_index('Quartile').T.reset_index(drop=True)

def generate_segments(dates_series, interval_years=2, interval_months=6):
    segments = []
    dates_series.sort_values(inplace=True)
    current_date = dates_series.iloc[0]  # Start with the minimum date
    for date in dates_series.iloc[1:]:
        segment_end = current_date + timedelta(
            days=(interval_years * 365.25 + interval_months * 30.44) - 1
        )
        if date > segment_end:
            segments.append((current_date, segment_end))
            current_date = date
    segments.append((current_date, dates_series.iloc[-1]))  # Last segment
    return segments


def func(x):
    dates = pd.to_datetime(x['DATE']).copy()
    segments = generate_segments(dates)
    return segments


def get_dd_cat(DF_DD):
    cost_centers = []
    expenses = []
    
    df_dd_cat = DF_DD.groupby(['VESSEL NAME', 'PERIOD', 'CATEGORIES']).Expense.sum().reset_index()
    df_dd_cat['DATE'] = df_dd_cat['PERIOD'].astype('str').apply(lambda x: f"{x[:4]}-{x[4:]}-01" if x else None)
    # df_dd_cat.groupby(['COST_CENTER', 'CATEGORIES', 'PERIOD', 'DATE'])['AMOUNT_USD'].sum().reset_index()
    cat_seg = df_dd_cat.groupby(['VESSEL NAME', 'CATEGORIES']).apply(func)

    for rec in cat_seg.reset_index(name='daterange').itertuples():
        cc = rec[1]
        for dd in rec[3]:
            temp = df_dd_cat[(df_dd_cat['VESSEL NAME'] == cc) & (df_dd_cat.DATE >= pd.to_datetime(dd[0]).strftime("%Y-%m-%d")) & (df_dd_cat.DATE <= pd.to_datetime(dd[1]).strftime("%Y-%m-%d"))]
            cost_centers.append(cc)
            expenses.append(temp.Expense.sum())
            
    cat_seg_event = pd.DataFrame()
    cat_seg_event['VESSEL NAME'] = pd.Series(cost_centers)
    cat_seg_event['EXPENSE'] = pd.Series(expenses)
    
 
    # filtered_df = cat_seg_event
    filtered_df = cat_seg_event[cat_seg_event.EXPENSE > 0.00]
    q1 = int(filtered_df['EXPENSE'].quantile(0.25))
    q2 = int(filtered_df['EXPENSE'].quantile(0.50))  # This is the median
    q3 = int(filtered_df['EXPENSE'].quantile(0.75))
    

    # Create a DataFrame with quartile values
    return filtered_df, pd.DataFrame({'Quartile': ['CATEGORIES', 'median_50perc_population', 'optimal_63perc_population', 'top_75perc_population'],
                                'Value': [rec[2], q1, q2, q3]}).set_index('Quartile').T.reset_index(drop=True)


def get_dd_subcat(DF_DD):
    cost_centers = []
    expenses = []
    ac_codes=[]
    sub_cats = []
    if DF_DD.empty:
        print("Input DataFrame is empty.")
        return (pd.DataFrame(), pd.DataFrame())

    df_dd_cat = DF_DD.groupby(['VESSEL NAME', 'PERIOD', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).Expense.sum().reset_index()
    df_dd_cat['DATE'] = df_dd_cat['PERIOD'].astype('str').apply(lambda x: f"{x[:4]}-{x[4:]}-01" if x else None)
    # df_dd_cat.groupby(['COST_CENTER', 'CATEGORIES', 'PERIOD', 'DATE'])['AMOUNT_USD'].sum().reset_index()
    cat_seg = df_dd_cat.groupby(['VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).apply(func)

    for rec in cat_seg.reset_index(name='daterange').itertuples():
        cc = rec[1]
        for dd in rec[5]:
            temp = df_dd_cat[(df_dd_cat['VESSEL NAME'] == cc) & (df_dd_cat.DATE >= pd.to_datetime(dd[0]).strftime("%Y-%m-%d")) & (df_dd_cat.DATE <= pd.to_datetime(dd[1]).strftime("%Y-%m-%d"))]
            cost_centers.append(cc)
            expenses.append(temp.Expense.sum())
            ac_codes.append(rec[3])
            sub_cats.append(rec[4])
            
    subcat_seg_event = pd.DataFrame()
    subcat_seg_event['VESSEL NAME'] = pd.Series(cost_centers)
    subcat_seg_event['CATEGORIES'] = rec[2]
    subcat_seg_event['ACCOUNT_CODE'] = pd.Series(ac_codes)
    subcat_seg_event['SUB_CATEGORIES'] = pd.Series(sub_cats)
    subcat_seg_event['EXPENSE'] = pd.Series(expenses)
    
    filtered_df = subcat_seg_event[subcat_seg_event.EXPENSE > 0.00]
    # filtered_df = subcat_seg_event

    subcat_df_pd = filtered_df.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=('EXPENSE', lambda x: np.quantile(x, 0.50)),
        optimal_63perc_population=('EXPENSE', lambda x: np.quantile(x, 0.63)),
        top_75perc_population=('EXPENSE', lambda x: np.quantile(x, 0.75))
        )

    return filtered_df, subcat_df_pd

def get_pd_data(df_pd):

    if df_pd.empty:
        print("Input DataFrame is empty.")
        return (pd.DataFrame(),) * 4
    cat_df_pd_ = df_pd.groupby(['VESSEL NAME', 'CATEGORIES']).Expense.median()

    subcat_df_pd_ = df_pd.groupby(['VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).Expense.median()
    
    cat_df_pd = cat_df_pd_.groupby(['CATEGORIES']).agg(
        median_50perc_population=lambda x: np.quantile(x, 0.50),
        optimal_63perc_population=lambda x: np.quantile(x, 0.63),
        top_75perc_population=lambda x: np.quantile(x, 0.75)
        ).astype(int)
    
    subcat_df_pd = subcat_df_pd_.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES']).agg(
        median_50perc_population=lambda x: np.quantile(x, 0.50),
        optimal_63perc_population=lambda x: np.quantile(x, 0.63),
        top_75perc_population=lambda x: np.quantile(x, 0.75)
        ).astype(int)
    
    return cat_df_pd_, subcat_df_pd_, cat_df_pd, subcat_df_pd

def plotly_monthly_quartiles(data):
    try:
        # Group by 'PERIOD' and 'VESSEL NAME' to calculate expenses
        monthly_data = data.groupby(['PERIOD', 'VESSEL NAME'])['Expense'].sum()
        
        # Calculate percentiles (q1, q2, median, q3)
        percentiles = monthly_data.groupby(['PERIOD']).agg(
            q1=lambda x: np.quantile(x, 0.25),
            q2=lambda x: np.quantile(x, 0.50),
            median=lambda x: np.quantile(x, 0.63),
            q3=lambda x: np.quantile(x, 0.75)
        ).reset_index()
        
        # Convert 'PERIOD' to dates
        dates = pd.to_datetime(percentiles['PERIOD'], format='%Y%m').dt.date
        
        # Calculate median values for additional lines
        q1_median = percentiles['q2'].median()
        overall_median = percentiles['median'].median()
        q3_median = percentiles['q3'].median()

        # Prepare the data for frontend graphing
        response_data = {
            'dates': dates.tolist(),  # Convert dates to a list of strings for JSON serialization
            'q1': percentiles['q1'].tolist(),
            'q2': percentiles['q2'].tolist(),
            'median': percentiles['median'].tolist(),
            'q3': percentiles['q3'].tolist(),
            'overall_median': overall_median,
            'q1_median': q1_median,
            'q3_median': q3_median
        }
        
        return response_data
    
    except Exception as e:
        return {'error': str(e)}


def plotly_yearly_quartiles(data):
    try:
        # Ensure 'YEAR' is a string
        data['YEAR'] = data['YEAR'].astype('str')
        
        # Group by 'YEAR', 'PERIOD', and 'VESSEL NAME' to calculate yearly expenses
        yearly_data = data.groupby(['YEAR', 'PERIOD', 'VESSEL NAME'])['Expense'].sum().reset_index()

        # Calculate yearly percentiles (q1, q2, median, q3)
        percentiles = yearly_data.groupby(['YEAR'])['Expense'].agg(
            optimal_q1=lambda x: np.quantile(x, 0.25),
            optimal_q2=lambda x: np.quantile(x, 0.50),
            optimal_median=lambda x: np.quantile(x, 0.63),
            optimal_q3=lambda x: np.quantile(x, 0.75)
        ).reset_index()

        # Get years (for the x-axis)
        years = percentiles['YEAR'].tolist()

        # Calculate median values for additional lines
        q1_median = percentiles['optimal_q2'].median()
        overall_median = percentiles['optimal_median'].median()
        q3_median = percentiles['optimal_q3'].median()

        # Prepare the data for frontend graphing
        response_data = {
            'dates': years,  # Convert years to list for frontend
            'q1': percentiles['optimal_q1'].tolist(),
            'q2': percentiles['optimal_q2'].tolist(),
            'median': percentiles['optimal_median'].tolist(),
            'q3': percentiles['optimal_q3'].tolist(),
            'overall_median': overall_median,
            'q1_median': q1_median,
            'q3_median': q3_median
        }

        return response_data

    except Exception as e:
        return {'error': str(e)}
