from datetime import timedelta
from io import StringIO
from typing import Dict
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, norm
import redis
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

def cohen_d(group1, group2):
    """
    Calculate Cohen's d for two groups.
    """
    mean_diff = np.mean(group1) - np.mean(group2)
    pooled_std = np.sqrt((np.std(group1)**2 + np.std(group2)**2) / 2)
    return mean_diff / pooled_std

def optimize_mean(data_no_outliers, optimal_expected_mean, max_flag):
    """
    Optimize the expected mean considering both p-value and effect size.
    """
    min_combined_score = float('inf')  # Initialize the minimum combined score

    # Iterate through expected mean values within the range from median to mean
    for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 1000):
        _, p_value_candidate = ttest_1samp(data_no_outliers, popmean=expected_mean_candidate)
        p_value_candidate = max(p_value_candidate, 0.06)  # Ensure p-value doesn't go below 0.05

        # Calculate Cohen's d as the effect size
        effect_size_candidate = cohen_d(data_no_outliers, [expected_mean_candidate] * len(data_no_outliers))

        # Define weights for the optimization criteria
        p_value_weight = 0.5  # Adjust the weights as needed
        effect_size_weight = 0.5

        # Calculate the combined score as the weighted sum of the p-value and effect size
        combined_score = (
            p_value_weight * p_value_candidate + 
            effect_size_weight * abs(effect_size_candidate)
        )

        # Update the optimal values if the combined score is better
        if combined_score < min_combined_score:
            min_combined_score = combined_score
            optimal_expected_mean = expected_mean_candidate

    return optimal_expected_mean

 
def get_cat_optimal_mean(exl: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the optimal mean for each category in parallel using threading and cache the results.
    """
    # Define a cache key based on DataFrame hash or a constant if data is relatively static
    cache_key = "optimal_means_by_category"
    cached_data = redis_client.get(cache_key)

    # If data is found in the cache, load and return it as a DataFrame
    if cached_data:
        print("Data loaded from cache")
        return pd.read_json(StringIO(cached_data))

    # Prepare data for processing
    exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME', 'CATEGORIES'])['Expense'].sum().reset_index()
    exl = exl.loc[exl['Expense'] > 0]
    exl = exl.groupby(['YEAR', 'CATEGORIES'])['Expense'].median().reset_index()
    cats = exl['CATEGORIES'].unique()
    optimal_means: Dict[str, float] = {}

    # Function to optimize mean for a single category
    def optimize_category(cat: str) -> Dict[str, float]:
        data = exl[exl['CATEGORIES'] == cat]
        data = data['Expense'] if cat == 'Administrative Expenses' else data[data['Expense'] != 0]['Expense']
        transformed_data = data[~((data - np.mean(data)) > 3.5 * np.std(data))]

        # Calculate mean and median, then determine optimal expected mean
        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean = optimize_mean(transformed_data, optimal_expected_mean, max_flag)
        return {cat: optimal_expected_mean}

    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor() as executor:
        results = executor.map(optimize_category, cats)
        for result in results:
            optimal_means.update(result)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['stats_model_optimal_budget'])
    df.index.name = 'CATEGORIES'
    df = df.reset_index()

    # Cache the result in Redis with an expiration time (e.g., 1 hour)
    redis_client.setex(cache_key, timedelta(hours=1), df.to_json())

    return df

def calculate_geometric_mean(exl, level='CATEGORIES'):
    """
    Calculate the geometric mean for each category or subcategory.
    Args:
        exl (DataFrame): Input DataFrame containing expense data.
        level (str): Level at which to calculate the geometric mean ('CATEGORIES' or 'SUB_CATEGORIES').
    Returns:
        DataFrame: DataFrame containing the geometric mean for each category or subcategory.
    """
    # Group data by specified level and calculate geometric mean
    if level == 'CATEGORIES':
        # Calculate geometric mean for each category
        grouped_data = exl.groupby(['YEAR', 'CATEGORIES'])['Expense'].apply(lambda x: np.exp(np.mean(np.log(x + 1)))) - 1
    elif level == 'SUB_CATEGORIES':
        # Calculate geometric mean for each subcategory
        grouped_data = exl.groupby(['YEAR', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].apply(lambda x: np.exp(np.mean(np.log(x + 1)))) - 1
    else:
        raise ValueError("Invalid level. Use 'CATEGORIES' or 'SUB_CATEGORIES'.")

    # Convert series to DataFrame
    df = grouped_data.reset_index(name='Geometric Mean')
    
    return df

 
def get_event_cat_optimal_mean(exl: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the optimal mean for each category and cache results in Redis.
    """

    # Generate a unique cache key based on the input DataFrame
    cache_key = f"get_event_cat_optimal_mean_{hash(exl.to_json())}"

    # Check if the result is already in Redis
    cached_result = redis_client.get(cache_key)
    if cached_result:
        print("Returning cached result")
        return pd.read_json(StringIO(cached_result))

    # Filter data and initialize dictionary for optimal means
    exl = exl.loc[exl['EXPENSE'] > 0]
    cats = exl['CATEGORIES'].unique()
    optimal_means = {}

    # Function to optimize mean for a single category
    def optimize_category(cat: str) -> None:
        data = exl[exl['CATEGORIES'] == cat]
        data = data['EXPENSE'] if cat == 'Administrative Expenses' else data[data['EXPENSE'] != 0]['EXPENSE']
        
        transformed_data = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
        
        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)

        # Calculate the optimal expected mean
        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Call optimize_mean (assuming this function is defined elsewhere)
        optimal_expected_mean, _, _ = optimize_mean(transformed_data, optimal_expected_mean, max_flag)

        # Store optimal mean in dictionary
        optimal_means[cat] = optimal_expected_mean

    # Optimize for each category
    for cat in cats:
        optimize_category(cat)

    # Convert the dictionary to a DataFrame
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['stats_model_optimal_budget'])
    df.index.name = 'CATEGORIES'
    df = df.reset_index()

    # Cache the result in Redis with a 1-hour expiration
    redis_client.setex(cache_key, timedelta(hours=1), df.to_json())

    return df

def get_event_subcat_optimal_mean(exl):
    """
    Calculate the optimal mean for each category without using parallel processing.
    """
    # exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['Expense'].sum().reset_index()
    exl = exl.loc[exl['EXPENSE'] != 0]
    # exl = exl.groupby(['CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES'])['EXPENSE'].median().reset_index()
    # sub_cats = exl['SUB_CATEGORIES'].unique()
    sub_cats = exl.groupby(['CATEGORIES', 'ACCOUNT_CODE']).size().reset_index()
    optimal_means = dict()

    # Function to optimize mean for a single category
    def optimize_category(cat, ac_code):
        data = exl[(exl['CATEGORIES'] == cat) & (exl['ACCOUNT_CODE'] == ac_code)]
        data = data[data['EXPENSE'] > 0]
        data = data['EXPENSE']
        transformed_data = data[~((data - np.mean(data)) > 1 * np.std(data))]

        mean_value = np.mean(transformed_data)
        median_value = np.quantile(transformed_data, 0.75)
        std_dev = np.std(transformed_data)
        
        # mode_value = mode(transformed_data).mode[0]

        x = np.linspace(mean_value - 3 * std_dev, mean_value + 3 * std_dev, 1000)
        y = norm.pdf(x, mean_value, std_dev)

        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Optimize the expected mean
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(transformed_data, optimal_expected_mean, max_flag)
        
        # Store optimal mean in dictionary
        optimal_means["{};{}".format(cat, ac_code)] = optimal_expected_mean
        return optimal_means
    
    for _, row in sub_cats.iterrows():
        optimal_means.update(optimize_category(row['CATEGORIES'], row['ACCOUNT_CODE']))
        
    df = pd.DataFrame.from_dict(optimal_means, orient='index', columns=['stats_model_optimal_budget'])
    df.index.name = 'SUB CATEGORIES'
    df = df.reset_index()
    return df
