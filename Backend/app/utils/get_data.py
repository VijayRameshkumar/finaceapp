from dotenv import load_dotenv
import os
import snowflake.connector
import pandas as pd
import datetime

load_dotenv()

def get_data(query, database=None, schema=None):

    account = os.getenv('ACCOUNT')
    user = os.getenv('USER')
    password = os.getenv('PASSWORD')
    warehouse = os.getenv('WAREHOUSE')
    database = os.getenv('DATABASE')
    schema = os.getenv('SCHEMA')

    # Establish the connection
    conn = snowflake.connector.connect(
        user='DATABRICKS_USER',
        password= 'Syn_db!23',
        account= 'synergygrp-dwh',
        warehouse= 'COMPUTE_WH',
        database= 'CURATED_DB',
        schema= 'INCOME_STATEMENT'
    )

    # Create a cursor object
    cursor = conn.cursor()

    cursor.execute(query)

    # Fetch and print the result
    result = cursor.fetchall()

    # Close the cursor and connection
    cursor.close()
    conn.close()
    return result

def get_expense_data():
    
    VESSEL_PARTICULARS_URL=os.getenv('VESSEL_PARTICULARS_URL')
    unique_cc = pd.read_excel(VESSEL_PARTICULARS_URL, engine='openpyxl')['VESSEL CODE'].unique().tolist()
    unique_cc_str = ', '.join([f"'{cc}'" for cc in unique_cc])
    
    # Get the current year
    current_year = datetime.datetime.now().year

    # Calculate the start and end periods
    start_period = (current_year - 3) * 100 + 1  # January of the year 3 years ago
    end_period = (current_year - 1) * 100 + 12  # December of last year
    
    expense_query = f'''
    SELECT
        TO_VARIANT(DATE_TRUNC('MONTH', MS.POSTING_DATE))::VARCHAR AS "Year-Month",
        MS.COST_CENTER,
        MS.NODECODE AS "Categories",
        MS.ACCOUNT_CODE,
        MS.ACCOUNT_CODE_DESC AS "SUB_CATEGORIES",
        SUM(MS.AMOUNT_USD) AS "Expense"
    FROM
        CURATED_DB.VESSEL_ACCOUNTS.MANAGER_STATEMENT MS
    WHERE
        MS.BRANCHCODE = 'Operating Expenses' AND 
        MS.COST_CENTER IN ({unique_cc_str}) AND 
        MS.PERIOD >= {start_period} AND 
        MS.PERIOD <= {end_period}
    GROUP BY 
        "Year-Month", 
        MS.COST_CENTER, 
        MS.NODECODE, 
        MS.ACCOUNT_CODE, 
        MS.ACCOUNT_CODE_DESC
    ORDER BY 
        "Year-Month", 
        MS.COST_CENTER, 
        MS.NODECODE ASC
    '''
    
    expense_data = pd.DataFrame(get_data(expense_query), columns=['DATE', 'COST_CENTER', 'CATEGORIES', 'ACCOUNT_CODE', 'SUB_CATEGORIES', 'Expense'])
    
    # Efficiently handle date conversion and filtering
    expense_data['PERIOD'] = pd.to_datetime(expense_data['DATE']).dt.strftime('%Y%m').astype('int')
    expense_data['DATE'] = pd.to_datetime(expense_data['DATE']).dt.date
    expense_data['Expense'] = expense_data['Expense'].astype('int')

    # Filtering
    expense_data = expense_data[(expense_data['PERIOD'] >= start_period) & (expense_data['PERIOD'] <= end_period)]
    
    return expense_data