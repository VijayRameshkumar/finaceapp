import numpy as np
import pandas as pd
from scipy.stats import norm, ttest_1samp, mode
import warnings
warnings.filterwarnings('ignore')

#  Optimization Function 

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
    min_p_value = 1  # Initialize the minimum p-value
    min_effect_size = 0  # Initialize the minimum effect size

    # Iterate through expected mean values within the range from median to mean
    for expected_mean_candidate in np.linspace(optimal_expected_mean, max_flag, 10000):
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
            min_p_value = p_value_candidate
            min_effect_size = effect_size_candidate

    return optimal_expected_mean, min_p_value, min_effect_size

#################### Main ################################################
def get_optimal_mean(exl):
    unique_years = exl.YEAR.unique()
    exl = exl.groupby(['YEAR', 'PERIOD', 'VESSEL NAME'])['Expense'].sum().reset_index()
    
    results = []

    for yr in unique_years:
        # Filter data by year
        data = exl[exl['YEAR'] == yr]
        data = data[data['Expense'] > 0]['Expense']
        
        # Remove outliers
        data_no_outliers = data[~((data - np.mean(data)) > 3.5 * np.std(data))]
        
        # Calculate statistics: mean, median, mode, and std deviation
        mean_value = np.mean(data_no_outliers)
        median_value = np.median(data_no_outliers)
        std_dev = np.std(data_no_outliers)

        # Safely handle the mode result        
        mode_result = mode(data_no_outliers)
        mode_value = mode_result.mode[0] if hasattr(mode_result.mode, '__len__') and len(mode_result.mode) > 0 else None


        # Determine optimal mean based on conditions
        optimal_expected_mean = median_value if mean_value > median_value else mean_value
        max_flag = mean_value if mean_value > median_value else median_value

        # Call your optimization function
        optimal_expected_mean, min_p_value, min_effect_size = optimize_mean(data_no_outliers, optimal_expected_mean, max_flag)
        
        # Store the results for this year
        results.append({
            'year': yr,
            'mean': mean_value,
            'median': median_value,
            'mode': mode_value,
            'std_dev': std_dev,
            'optimal_mean': optimal_expected_mean,
            'p_value': min_p_value,
            'effect_size': min_effect_size
        })

    return results
