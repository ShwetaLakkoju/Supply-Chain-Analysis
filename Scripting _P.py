#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install scikit-learn')


# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from IPython.display import display
from scipy.stats.mstats import winsorize
import os


# In[ ]:


class DataPreProcessingAndEDA:
    """
    Class for preprocessing and performing exploratory data analysis on pandas DataFrames.
    Includes methods for data description, cleaning, and visualization.
    """

    @staticmethod
    def df_description(df):
        """
        Prints the shape and data types of the DataFrame.
        """
        print('Shape:', df.shape)
        print('Data types:', '\n', df.dtypes)

    @staticmethod
    def convert_col_names_to_lower_case(df):
        """
        Converts DataFrame column names to lower case.
        """
        return df.rename(columns={col: col.lower() for col in df.columns})

    @staticmethod
    def describe_data(df):
        """
        Displays descriptive statistics of DataFrame.
        """
        display(df.describe(include='all'))

    @staticmethod
    def data_info(df):
        """
        Displays information about DataFrame.
        """
        df.info(verbose=True, show_counts=True)

    @staticmethod
    def nulls_in_data(df):
        """
        Prints the number of nulls in each column of the DataFrame.
        """
        print(df.isnull().sum())

    @staticmethod
    def duplicate_rows_at_primary_key_level(df, level_of_the_data):
        """
        Identifies and prints duplicate rows based on the specified primary key columns.
        """
        duplicates = df[df.duplicated(subset=level_of_the_data, keep=False)]
        if duplicates.shape[0] > 0:
            print("DataFrame has duplicates, total rows with duplicates:", duplicates.shape[0])
        else:
            print("No duplicates in data")
        return duplicates

    @staticmethod
    def percentage_nulls_in_each_col(df):
        """
        Displays the percentage of nulls in each column of the DataFrame.
        """
        na_count = df.isna().sum()
        na_percent = (na_count / df.shape[0] * 100).apply("{:.2f}%".format)
        NA = pd.DataFrame({'NA Count': na_count, 'NA Percent': na_percent, 'Dtypes': df.dtypes})
        display(NA)

    @staticmethod
    def drop_cols(df, column_list_to_drop):
        """
        Drops specified columns from the DataFrame.
        """
        return df.drop(columns=column_list_to_drop)

    @staticmethod
    def df_columns_and_dtypes_into_list(df):
        """
        Returns a dictionary of DataFrame columns and their respective data types.
        """
        return dict(zip(df.columns, map(str, df.dtypes)))

    @staticmethod
    def unique_entries_count_each_column(df):
        """
        Returns a dictionary with column names as keys and the number of unique entries as values.
        """
        return {col: df[col].nunique() for col in df.columns}


    
    @staticmethod
    def value_counts_by_column(df, column_name):
        """
        Returns the counts of unique values for the specified column using value_counts().
        """
        return df[column_name].value_counts()


    @staticmethod
    def convert_column_to_string(df, column_name):
        """
        Converts the specified column to string type.
        """
        df[column_name] = df[column_name].astype(str)
        return df

    @staticmethod
    def convert_column_to_numeric(df, column_name, numeric_type='float64'):
        """
        Converts the specified column to a numeric type (int64 or float64).
        """
        if numeric_type == 'int64':
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0).astype('int64')
        else:
            df[column_name] = pd.to_numeric(df[column_name], errors='coerce').fillna(0.0).astype('float64')
        return df

    @staticmethod
    def check_distribution(df):
        """
        Checks the distribution of numerical columns by calculating skewness and kurtosis, and plotting histograms.
        """
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns
        results = []

        for col in num_cols:
            skewness = skew(df[col].dropna())
            kurt = kurtosis(df[col].dropna())
            results.append({'Variable': col, 'Skewness': skewness, 'Kurtosis': kurt})

            plt.figure(figsize=(10, 6))
            sns.histplot(df[col], kde=True, bins=30)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        return pd.DataFrame(results)

    @staticmethod
    def outlier_report(df):
        """
        Generates a report of potential and definite outliers in numerical columns based on Z-scores.
        """
        report = []
        num_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in num_cols:
            zs = zscore(df[col].dropna())
            possible_outliers = ((zs >= 2) & (zs < 3)).sum()
            definite_outliers = (zs >= 3).sum()
            report.append({
                'Numerical Variable': col,
                'Possible Outliers': possible_outliers,
                'Definite Outliers': definite_outliers,
                'Total Outliers': possible_outliers + definite_outliers
            })

        return pd.DataFrame(report)

    @staticmethod
    def analyze_freight_cost_distribution(df, x_var):
        """
        Analyzes and visualizes the distribution of 'freight cost (usd)' across categories of 'x_var'.
        """
        df['freight cost (usd)'] = pd.to_numeric(df['freight cost (usd)'], errors='coerce')
        valid_data = df.dropna(subset=['freight cost (usd)', x_var])

        plt.figure(figsize=(20, 10))
        sns.boxplot(x=x_var, y='freight cost (usd)', data=valid_data)
        plt.title(f'Distribution of Freight Cost (USD) by {x_var}')
        plt.xlabel(x_var)
        plt.ylabel('Freight Cost (USD)')
        plt.show()

        return valid_data.groupby(x_var)['freight cost (usd)'].describe()

    @staticmethod
    def plot_and_analyze_data(df, x_var):
        """
        Plots data and analyzes correlation between 'x_var' and 'freight cost (usd)'.
        """
        df['freight cost (usd)'] = pd.to_numeric(df['freight cost (usd)'], errors='coerce')
        df = df.dropna(subset=[x_var, 'freight cost (usd)'])

        if df[x_var].dtype in [np.float64, np.int64]:
            sns.boxplot(x=df[x_var])
            plt.show()

        sns.lmplot(x=x_var, y='freight cost (usd)', data=df)
        plt.show()

        return df[x_var].corr(df['freight cost (usd)'])

    @staticmethod
    def winsorize_dataframe(df, limits=(0.01, 0.01)):
        """
        Applies winsorization to all numerical columns in the DataFrame to limit extreme values.
        """
        winsorized_df = df.copy()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

        for col in numerical_cols:
            winsorized_df[col] = winsorize(df[col], limits=limits)

        return winsorized_df
    
    @staticmethod
    def convert_column_to_date(df, column_name, date_format=None):
        """
        Converts the specified column to datetime type. Optionally, a specific date_format can be provided.
        """
        df[column_name] = pd.to_datetime(df[column_name], format=date_format, errors='coerce')
        return df


# # **`Loading, Reading and Cleaning the Dataset`**

# In[ ]:


print("Previous Working Directory: ", os.getcwd())

desired_path = '/home/jovyan/library/ist652/spring24/project'

# Changing  the current working directory
os.chdir(desired_path)

print("Current Working Directory: ", os.getcwd())


# In[ ]:


df = pd.read_csv("./Supply_Chain_Shipment_Pricing_Dataset_20240325.csv")


# In[ ]:


df.head()


# In[ ]:


eda = DataPreProcessingAndEDA()


# In[ ]:


eda.data_info(df)


# **Insight**: From df.info() we can see that there are null values in the following columns - shipment mode, dosage, line item insurance.

# In[ ]:


eda.percentage_nulls_in_each_col(df)


# In[ ]:


eda.df_description(df)


# In[ ]:


eda.describe_data(df)


# In[ ]:


eda.nulls_in_data(df)


# In[ ]:


eda.df_columns_and_dtypes_into_list(df)


# In[ ]:


eda.duplicate_rows_at_primary_key_level(df, 'id')


# In[ ]:


eda.value_counts_by_column(df, 'fulfill via')
#Example


# In[ ]:


eda.unique_entries_count_each_column(df)


# In[ ]:


eda.value_counts_by_column(df, 'project code')
#This column's datatype need not be converted to any other format. 


# In[ ]:


eda.value_counts_by_column(df, 'pq #')
#No need to convert


# In[ ]:


eda.value_counts_by_column(df, 'po / so #')
#No need to convert


# In[ ]:


eda.value_counts_by_column(df, 'asn/dn #')
#No need to convert


# In[ ]:


eda.value_counts_by_column(df, 'country')
#No need to convert


# In[ ]:


eda.value_counts_by_column(df, 'managed by')


# In[ ]:


eda.value_counts_by_column(df, 'fulfill via')


# In[ ]:


eda.value_counts_by_column(df, 'vendor inco term')
#Need to change acronyms


# In[ ]:


eda.value_counts_by_column(df, 'shipment mode')
#no need to convert datatype


# In[ ]:


eda.value_counts_by_column(df, 'pq first sent to client date')
#Need to convert to date


# In[ ]:


eda.value_counts_by_column(df, 'po sent to vendor date')
#need to convert to date


# In[ ]:


eda.value_counts_by_column(df, 'scheduled delivery date')
#need to convert to date


# In[ ]:


eda.value_counts_by_column(df, 'delivered to client date')
#need to convert to date


# In[ ]:


eda.value_counts_by_column(df, 'delivery recorded date')
#need to convert to date


# In[ ]:


eda.value_counts_by_column(df, 'product group')
#no need to convert but should find acronyms


# In[ ]:


eda.value_counts_by_column(df, 'sub classification')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'vendor')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'item description')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'molecule/test type')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'brand')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'dosage')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'dosage form')


# In[ ]:


eda.value_counts_by_column(df, 'unit of measure (per pack)')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'line item quantity')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'line item value')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'pack price')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'unit price')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'manufacturing site')
#no need to convert


# In[ ]:


eda.value_counts_by_column(df, 'first line designation')


# In[ ]:


eda.value_counts_by_column(df, 'weight (kilograms)')
#should convert


# In[ ]:


eda.value_counts_by_column(df, 'freight cost (usd)')
#should convert


# In[ ]:


eda.value_counts_by_column(df, 'line item insurance (usd)')
#no need to convert


# **`Insights`**: In the above lines of code, we have found information on the characteristics of our dataset, found out information about null values, about duplicates in our dataset, about count of unique values and other basic information about the data. In the following lines of code, we will be changing data types, replacing acronyms wherever needed, and dropping unnecessary columns. 

# **Changing Datatypes**

# In[ ]:


eda.convert_column_to_numeric(df, 'weight (kilograms)', numeric_type='float64')


# In[ ]:


eda.convert_column_to_numeric(df, 'freight cost (usd)', numeric_type='float64')


# In[ ]:


eda.df_columns_and_dtypes_into_list(df)


# In[ ]:


eda.convert_column_to_date(df,'scheduled delivery date')


# In[ ]:


eda.convert_column_to_date(df,'scheduled delivery date')


# In[ ]:


eda.convert_column_to_date(df,'delivered to client date')


# In[ ]:


eda.convert_column_to_date(df,'delivery recorded date')


# In[ ]:


eda.df_columns_and_dtypes_into_list(df)


# In[ ]:


eda.nulls_in_data(df)


# In[ ]:


eda.value_counts_by_column(df, 'delivered to client date')


# In[ ]:


eda.value_counts_by_column(df, 'scheduled delivery date')


# In[ ]:


eda.value_counts_by_column(df, 'delivery recorded date')


# **Insights**: We have now converted the data types and have cross checked if the value counts have changed and we are good to go. In the next steps, we will remove the columns we dont need.

# **`Removing unnecessary columns`** 
# We will be removing the following columns as they would not be interesting to us from a research question perspective. 
# 1) project code
# 2) pq#
# 3) po / so #
# 4) asn/dn #
# 5) pq first sent to client date
# 6) po sent to vendor date
# 7) dosage

# In[ ]:


column_list_to_drop = ['project code', 'pq #', 'po / so #', 'asn/dn #', 'pq first sent to client date',
                       'po sent to vendor date', 'dosage']
df = eda.drop_cols(df, column_list_to_drop)


# In[ ]:


#checking the info., about columns after removing the columns
df.info()


# In[ ]:


eda.df_columns_and_dtypes_into_list(df)


# **Insights**: In the above lines of code, we have removed the columns that we would not be using. 
# Reasoning: Having id and columns like project code, pq#, po / so #, asn/dn # which are like secondary keys made no sense to us as we did not feel they would not affect cost much considering how many unique values each of them had. 
# The columns pq first sent to client date and po sent to vendor date got more to do with order placements, rather than being the ones driving cost - and the data in the columns seemed incomplete as it had a lot of invalid date entries like not available, hence we decided to drop them. Additionally, we were focussed more on operational and logistical insights rather than clinical insights hence we decided to drop dosage too. In the next lines of code, we will be replacing some acronyms with what they actually mean for better understanding and ease of access considering these terms would not be known by all.

# **`Dealing with abbreviations`**
# We will change the abbreviations for two columns - vendor inco group and product group into their elaborated names.

# In[ ]:


eda.value_counts_by_column(df, 'product group')


# From research across the internet, we found out the following elaborations
# 1) ACT: Artemisinin-based Combination Therapy
# 2) ANTM: Anti-malarial medicine
# 3) ARV: Anti-Retroviral Treatment
# 4) HRDT: HIV Rapid Diagnostic Test
# 5) MRDT: Malarial Rapid Diagnostic Test

# In[ ]:


abbreviations = {
    'ACT': 'Artemisinin-based Combination Therapy',
    'ANTM': 'Anti-malarial medicine',
    'ARV': 'Anti-Retroviral Treatment',
    'HRDT': 'HIV Rapid Diagnostic Test',
    'MRDT': 'Malarial Rapid Diagnostic Test'
}
df['product group'] = df['product group'].replace(abbreviations)


# In[ ]:


eda.value_counts_by_column(df, 'product group')


# In[ ]:


eda.value_counts_by_column(df, 'vendor inco term')


# **Insights**: The column vendor inco terms contains Incoterms ® or International Commercial Terms, which are the common language of international trade. Established by the International Chamber of Commerce (ICC), they are standard terms which are commonly incorporated into contracts for the trade of goods around the world. We will replace their abbreviations with their elaborations.

# In[ ]:


abbreviations1 = {
    'EXW': 'Ex Works',
    'FCA': 'Free Carrier',
    'CIP': 'Carriage and Insurance Paid To',
    'DAP': 'Delivery at Place',
    'DDU': 'Delivery at Place Unloaded',
    'DDP': 'Delivery Duty Paid',
    'CIF': 'Cost, Insurance and Freight',
    'N/A - From RDC': 'From Regional Distribution Center'
}
df['vendor inco term'] = df['vendor inco term'].replace(abbreviations1)


# In[ ]:


eda.value_counts_by_column(df, 'vendor inco term')


# In[ ]:


eda.value_counts_by_column(df, 'managed by')


# **Insight**: We have now changed the abbreviations to their elaborated forms. In the next steps, we will handle missing values. 

# # **`Handling Missing Values`**

# In[ ]:


eda.nulls_in_data(df)


# In[ ]:


eda.percentage_nulls_in_each_col(df)


# ## **Handling Missing Values by Imputing Insurance Column**
# 
# We have null values in two columns - 1) line item insurance (usd) - a numerical column for which we will be using KNN Imputer. 2) shipment mode - a categorical variable for which we will one hot encode it and then use KNN Imputer. 

# In[ ]:


KNN_Imputer = KNNImputer(n_neighbors=5, weights='uniform')
columns_to_impute = ['line item insurance (usd)']
imputed_data = KNN_Imputer.fit_transform(df[columns_to_impute])
df[columns_to_impute] = imputed_data


# In[ ]:


df['line item insurance (usd)'].describe()


# In[ ]:


eda.percentage_nulls_in_each_col(df)


# **`Handling nulls and imputing shipment mode - a categorical variable`**

# In[ ]:


df[['shipment mode']] = df[['shipment mode']].fillna('Unavailable')


# In[ ]:


eda.percentage_nulls_in_each_col(df)


# **Insights**: In the above lines of code, we have handled missing values by replacing them with the term 'unavailable'. We will now do a preliminary set of EDA without handling outliers. 

# ### Initial EDA without outlier handling ###

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'country')


# **Insights**: Country seems to play a good role with respect to cost, considering how varied the numbers are. 

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'managed by')


# **Insights**: The managed by contains just four categories and almost all of the entries are from the PMO - US, considering how almost 99% of the dataset is from PMO - US, it does not make a lot of difference with respect to cost.

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'fulfill via')


# **Insights**: There are just two categories in this column, both of them have similar counts, we can somewhat use this to predict cost, but there is not a lot of difference in mean or the medians of both categories. 

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'vendor inco term')


# **Insight**: The vendor inco terms seems to have some variance wrt cost.

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'shipment mode')


# In[ ]:


eda.analyze_freight_cost_distribution(df, 'product group')


# **Insights**: Nearly 85% of the dataset contains Anti Retroviral Treatment as product group, and another chunk contains HIV Rapid Diagnostic Test but there is considerable difference in the mean costs. 

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'sub classification')


# **Insights**: This column is also somewhat similar to product group as in there is one major category but the other ones have a difference in mean costs.

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'vendor')


# In[ ]:


eda.analyze_freight_cost_distribution(df, 'molecule/test type')


# In[ ]:


eda.analyze_freight_cost_distribution(df, 'brand')


# In[ ]:


eda.analyze_freight_cost_distribution(df, 'dosage form')


# In[ ]:


eda.analyze_freight_cost_distribution(df, 'manufacturing site')


# **Insights**: All of the above columns show variability in costs. 

# In[ ]:


eda.plot_and_analyze_data(df, 'unit of measure (per pack)')


# In[ ]:


eda.plot_and_analyze_data(df, 'line item quantity')


# In[ ]:


eda.plot_and_analyze_data(df, 'line item value')


# In[ ]:


eda.plot_and_analyze_data(df, 'pack price')


# In[ ]:


eda.plot_and_analyze_data(df, 'unit price')


# In[ ]:


eda.plot_and_analyze_data(df, 'weight (kilograms)')


# In[ ]:


eda.plot_and_analyze_data(df, 'line item insurance (usd)')


# **Insights**: Most of the numerical variables have weak correlations with cost. 

# ### Outlier Handling ###

# In[ ]:


eda.outlier_report(df)


# In[ ]:


df = eda.winsorize_dataframe(df, limits=(0.05, 0.05))


# **Insights**: We looked for outliers and winsorized the outliers to 5% and 95% of the mean. 

# ### Feature Engineering ###
# Our feature engineering is focussed mainly on countries, and the date columns. We will group countries by continent and then for dates, we will try to find differences in delivery dates to check if they affect costs. 

# In[ ]:


country_to_continent = {
    'Afghanistan': 'Asia',
    'Angola': 'Africa',
    'Belize': 'North America',
    'Benin': 'Africa',
    'Botswana': 'Africa',
    'Burkina Faso': 'Africa',
    'Burundi': 'Africa',
    'Cameroon': 'Africa',
    'Congo, DRC': 'Africa',
    'Côte d\'Ivoire': 'Africa',
    'Dominican Republic': 'North America',
    'Ethiopia': 'Africa',
    'Ghana': 'Africa',
    'Guatemala': 'North America',
    'Guinea': 'Africa',
    'Guyana': 'South America',
    'Haiti': 'North America',
    'Kazakhstan': 'Asia',
    'Kenya': 'Africa',
    'Kyrgyzstan': 'Asia',
    'Lebanon': 'Asia',
    'Lesotho': 'Africa',
    'Liberia': 'Africa',
    'Libya': 'Africa',
    'Malawi': 'Africa',
    'Mali': 'Africa',
    'Mozambique': 'Africa',
    'Namibia': 'Africa',
    'Nigeria': 'Africa',
    'Pakistan': 'Asia',
    'Rwanda': 'Africa',
    'Senegal': 'Africa',
    'Sierra Leone': 'Africa',
    'South Africa': 'Africa',
    'South Sudan': 'Africa',
    'Sudan': 'Africa',
    'Swaziland': 'Africa',
    'Tanzania': 'Africa',
    'Togo': 'Africa',
    'Uganda': 'Africa',
    'Vietnam': 'Asia',
    'Zambia': 'Africa',
    'Zimbabwe': 'Africa'
}


# In[ ]:


df['continent'] = df['country'].map(country_to_continent)
print(df[['country', 'continent']])


# In[ ]:


eda.value_counts_by_column(df, 'continent')


# **Insights**: We have now mapped the countries to their respective continents. 

# In[ ]:


eda.analyze_freight_cost_distribution(df, 'continent')


# **Insights**: Makes sense add points** Add insights**

# In[ ]:


df['days between scheduled and delivered'] = (df['delivered to client date'] - df['scheduled delivery date']).dt.days
df['days between delivered and recorded'] = (df['delivery recorded date'] - df['delivered to client date']).dt.days


# In[ ]:


eda.plot_and_analyze_data(df, 'days between delivered and recorded')


# In[ ]:


eda.plot_and_analyze_data(df, 'days between scheduled and delivered')


# In[ ]:


eda.percentage_nulls_in_each_col(df)


# ### We are now done with EDA, data processing. We now start to answer our research questions. ###

# In[ ]:


# Grouping by country and calculate mean costs
mean_costs_by_country = df.groupby('country')['freight cost (usd)'].mean().reset_index()

# Finding the median costs by country
median_costs_by_country = df.groupby('country')['freight cost (usd)'].median().reset_index()

print(mean_costs_by_country)


# In[ ]:


# Grouping by country and calculate mean costs
mean_costs_by_country = df.groupby('continent')['freight cost (usd)'].mean().reset_index()

# Finding the median costs by country
median_costs_by_country = df.groupby('continent')['freight cost (usd)'].median().reset_index()

print(mean_costs_by_country)


# In[ ]:


# Grouping by shipment mode and calculating mean costs
mean_costs_by_mode = df.groupby('shipment mode')['freight cost (usd)'].mean().reset_index()
mean_costs_by_mode


# In[ ]:


#grouping by product group and calculating the mean costs
mean_costs_by_product_group = df.groupby('product group')['freight cost (usd)'].mean().reset_index()


# In[ ]:


# Costs by Shipment Mode
plt.figure(figsize=(10, 6))
sns.barplot(x='shipment mode', y='freight cost (usd)', data=mean_costs_by_mode)
plt.title('Mean Freight Costs by Shipment Mode')
plt.xticks(rotation=45)  # Rotate labels if they overlap
plt.show()


# In[ ]:


#costs by product_group
plt.figure(figsize=(12, 8)) 
sns.barplot(x='product group', y='freight cost (usd)', data=mean_costs_by_product_group, palette='muted')
plt.title('Mean Freight Costs by Product Group')
plt.xlabel('Product Group')
plt.ylabel('Mean Freight Cost (USD)')
plt.xticks(rotation=45)  # Rotate the x-axis labels for better readability if necessary

plt.show()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='line item value', y='freight cost (usd)')
plt.title('Freight Cost (USD) vs. Line Item Value')
plt.xlabel('Line Item Value')
plt.ylabel('Freight Cost (USD)')
plt.show()


# In[ ]:


import pandas as pd
from scipy.stats import zscore
from scipy.stats.mstats import winsorize

# Function to generate an outlier report
def outlier_report(df):
    report = []
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        zs = zscore(df[col])
        possible_outliers = ((zs >= 2) & (zs < 3)).sum()
        definite_outliers = (zs >= 3).sum()
        report.append({
            'Numerical Variable': col,
            'Possible Outliers': possible_outliers,
            'Definite Outliers': definite_outliers,
            'Total Outliers': possible_outliers + definite_outliers
        })
    return pd.DataFrame(report)

# Function to apply winsorization to all numerical columns in the DataFrame
def winsorize_dataframe(df, limits=(0.01, 0.01)):
    winsorized_df = df.copy()  # Creating a copy to avoid changing the original DataFrame
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    
    for col in numerical_cols:
        # Applying winsorization
        winsorized_data = winsorize(df[col], limits=limits)
        winsorized_df[col] = winsorized_data
    
    return winsorized_df

# Applying the winsorization function to the DataFrame
df_winsorized = winsorize_dataframe(df, limits=(0.01, 0.01))



# In[ ]:


plot_and_analyze_data(df_winsorized, 'line item quantity')


# In[ ]:


plot_and_analyze_data(df_winsorized, 'weight (kilograms)')


# In[ ]:


plot_and_analyze_data(df_winsorized, 'unit of measure (per pack)')


# ## **Research Question 3:**
#     
# **Vendor and Manufacturing Analysis: Which vendors are most commonly used, and how do they correlate with shipment(freight) costs and shipment speed?**

# First, we wanted to take a look at how and if **vendor inco terms**  have an impact on freight/shipping costs.  For example, certain Incoterms might correlate with higher or lower shipping costs.

# **A liitle bit about "Vendor Inco terms"**
# 
# **"Vendor Incoterms"** refers to International Commercial Terms (Incoterms) which are predefined, internationally recognized terms published by the International Chamber of Commerce (ICC). They are used to communicate the tasks, costs, and risks associated with the transportation and delivery of goods in international trade.
# 
# **Ex Works (EXW):** The seller makes the goods available at their premises. The buyer is responsible for all transportation costs and risks, including loading the goods onto a vehicle and customs clearance for export.
# 
# **Free Carrier (FCA):** The seller hands over the goods, cleared for export, to the buyer's chosen carrier at a specified location. From that point, the buyer assumes the cost and risk of transportation.
# 
# **Delivery at Place (DAP):** The seller delivers the goods to a destination specified by the buyer, and assumes all risks and costs until the goods are ready for unloading. The buyer is responsible for unloading and import clearance.
# 
# **Delivery at Place Unloaded (DPU):** Similar to DAP, except the seller is responsible for unloading the goods at the destination. This is the only Incoterm that requires the seller to unload the goods.
# 
# **Delivery Duty Paid (DDP):** The seller delivers the goods to the buyer, cleared for import, and ready at the named destination. The seller bears all costs and risks involved in bringing the goods to the destination, including import duties and taxes.
# 
# **Carriage and Insurance Paid To (CIP):** The seller pays for the carriage and insurance to the named destination point, but risk passes to the buyer once the goods have been handed over to the first carrier.
# 
# **Cost, Insurance, and Freight (CIF):** The seller arranges and pays for transportation to the port of destination. The seller also provides insurance for the goods until they reach the destination port. However, the risk passes to the buyer as soon as the goods are loaded onto the shipping vessel.
# 
# **From Regional Distribution Center:** This term is not a standard Incoterm and could refer to a specific arrangement where the goods are shipped from a regional distribution center. The exact terms would depend on the contract or agreement in place.
# 
# Incoterms are key in international trade as they determine how far along the process the seller is required to bear the cost and risk, and at which point these responsibilities shift to the buyer. They are essential for clear communication in contracts of sale and can significantly affect the cost and risk profiles of transactions.
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


# Grouping data by 'vendor inco term' and calculate mean freight costs

grouped_data = df.groupby('vendor inco term')['freight cost (usd)'].agg(['mean', 'median', 'std', 'min', 'max'])

df_renamed = df.rename(columns={'freight cost (usd)': 'freight_cost_usd', 'vendor inco term': 'vendor_inco_term'})

# Visualize the freight cost distribution across different incoterms
# Use the renamed column names for the visualization.
sns.boxplot(x='vendor_inco_term', y='freight_cost_usd', data=df_renamed)
plt.xticks(rotation=45)
plt.show()


# #### **Insights** 
# 
# **Variability:** The incoterm "Ex Works" shows a significantly higher variability in freight costs compared to the others, as evidenced by the long box and the extended upper whisker. This indicates that for "Ex Works", the freight cost can vary widely from one shipment to another.
# 
# **Comparing Means and Medians:**  The incoterms with the highest average freight costs are "Ex Works" and "From Regional Distribution Center", which could be due to these terms requiring the buyer to arrange for transportation, possibly incurring additional costs for coordination and logistics.
# 
# The boxplot does not directly show correlation, which is a statistical measure of the strength and direction of a linear relationship between two variables. However, it does show that different incoterms are associated with different freight cost structures.
# 
# 

# ### **Next step in vendor analysis**
# 
# The next step involves identifying the most frequently utilized vendors and assessing whether there is any variation in freight costs associated with each vendor.
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


pd.set_option('display.max_rows', 100)
eda.value_counts_by_column(df, 'vendor')


# **Observation:** 
# 
# We've observed that there are 73 different vendors, with many having relatively low transaction volumes (i.e., 1-100 transactions). To enhance our analysis, we're filtering out the less frequently used vendors. By limiting the analysis to the most common vendors, we are trying to obtain a clearer understanding of the freight costs trends and outliers without the noise from vendors with very few records.

# In[ ]:


#Setting a threshold for the minimum count to consider a vendor 'frequent'
threshold = 100  

# Finding vendors that occur at least 'threshold' times
frequent_vendors = df['vendor'].value_counts()
frequent_vendors = frequent_vendors[frequent_vendors >= threshold].index.tolist()

#printing frequent vendors.

print(frequent_vendors)


# Filtering the DataFrame to only include these vendors
df_frequent_vendors = df[df['vendor'].isin(frequent_vendors)]

# Now analyze the freight costs for these vendors
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.boxplot(x='vendor', y='freight cost (usd)', data=df_frequent_vendors)
plt.xticks(rotation=90)  
plt.title('Freight Cost Distribution for Frequent Vendors')
plt.xlabel('Vendor')
plt.ylabel('Freight Cost (USD)')
plt.tight_layout() 
plt.show()


# In[ ]:


# Filter to include only the most frequent vendors, as identified earlier
frequent_vendors = df['vendor'].value_counts().head(15).index
df_frequent = df[df['vendor'].isin(frequent_vendors)]

# Create a new column in the DataFrame that combines vendor and continent
df_frequent['vendor_continent'] = df_frequent['vendor'] + " - " + df_frequent['continent']

# Plotting the boxplot
plt.figure(figsize=(15, 10))
sns.boxplot(data=df_frequent, x='vendor_continent', y='freight cost (usd)')
plt.xticks(rotation=90)
plt.title('Freight Cost Distribution for Frequent Vendors by Continent')
plt.xlabel('Vendor - Continent')
plt.ylabel('Freight Cost (USD)')
plt.grid(True)
plt.tight_layout()  # Adjust layout to fit the labels and title
plt.show()


# ### ** Vendors vs continents**

# In[ ]:


frequent_vendors = df['vendor'].value_counts().head(15).index
df_frequent = df[df['vendor'].isin(frequent_vendors)]

def plot_continent_vendors(df, continent_name):
    df_continent = df[df['continent'] == continent_name]
    
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_continent, x='vendor', y='freight cost (usd)')
    plt.xticks(rotation=90)
    plt.title(f'Freight Cost Distribution for Vendors in {continent_name}')
    plt.xlabel('Vendor')
    plt.ylabel('Freight Cost (USD)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# List of continents
continents = ['Africa', 'North America', 'Asia', 'South America']

# Generate a boxplot for each continent
for continent in continents:
    plot_continent_vendors(df, continent)


# ### **vendor vs product group**

# In[ ]:


eda.value_counts_by_column(df, 'product group')


# In[ ]:


def plot_product_group_freight_costs(df, product_group_name):
    df_group = df[df['product group'] == product_group_name]
    
    plt.figure(figsize=(15, 10))
    sns.boxplot(data=df_group, x='vendor', y='freight cost (usd)')
    plt.xticks(rotation=90)
    plt.title(f'Freight Cost Distribution for {product_group_name}')
    plt.xlabel('Vendor')
    plt.ylabel('Freight Cost (USD)')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to fit the labels and title
    plt.show()

# List of product groups based on your dataset
product_groups = ['Anti-Retroviral Treatment', 'HIV Rapid Diagnostic Test', 
                  'Anti-malarial medicine', 'Artemisinin-based Combination Therapy', 
                  'Malarial Rapid Diagnostic Test']

# Generate a boxplot for each product group
for product_group in product_groups:
    plot_product_group_freight_costs(df, product_group)


# ### Next step is to check if shipment speed differs from vendor to vendor.

# In[ ]:


df.info()


# In[ ]:


frequent_vendors


# In[ ]:


# Finding vendors that occur at least 'threshold' times
frequent_vendors = df['vendor'].value_counts()
frequent_vendors = frequent_vendors[frequent_vendors >= threshold].index.tolist()

#printing frequent vendors.

print(frequent_vendors)


# Filtering the DataFrame to only include these vendors
df_frequent_vendors = df[df['vendor'].isin(frequent_vendors)]

# Calculate the average 'days between scheduled and delivered' for each vendor
vendor_delivery_speed_avg = df_frequent_vendors.groupby('vendor')['days between scheduled and delivered'].mean().reset_index()

# Now create a boxplot for the average delivery speed of frequent vendors
plt.figure(figsize=(15, 10))  # Adjust the size as needed
sns.barplot(x='vendor', y='days between scheduled and delivered', data=vendor_delivery_speed_avg)
plt.xticks(rotation=90)  
plt.title('Average Delivery Speed Analysis by Vendor')
plt.xlabel('Vendor')
plt.ylabel('Average Delivery Speed (days)')
plt.tight_layout() 
plt.show()


# In[ ]:


df['days between scheduled and delivered']


# In[ ]:


def top_vendors_by_continent(df):
    top_vendors_by_continent = {}

    continents = df['continent'].unique()
    for continent in continents:
        continent_df = df[df['continent'] == continent]
        vendor_counts = continent_df['vendor'].value_counts()
        top_vendors = vendor_counts.head(5)
        top_vendors_by_continent[continent] = top_vendors

    return top_vendors_by_continent

def plot_top_vendors_by_continent(top_vendors_by_continent):
    for continent, top_vendors in top_vendors_by_continent.items():
        plt.figure(figsize=(10, 6))
        top_vendors.plot(kind='bar', color='skyblue')
        plt.title(f"Top 5 Vendors in {continent}")
        plt.xlabel("Vendor")
        plt.ylabel("Transaction Count")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

# Assuming your filtered DataFrame is named 'filtered_df'
top_vendors_by_continent = top_vendors_by_continent(df)
plot_top_vendors_by_continent(top_vendors_by_continent)


# In[ ]:


def top_vendors_by_shipping_cost_per_continent(df):
    top_vendors_by_continent = {}

    continents = df['continent'].unique()
    for continent in continents:
        continent_df = df[df['continent'] == continent]
        vendor_shipping_cost = continent_df.groupby('vendor')['freight cost (usd)'].sum()
        top_vendors = vendor_shipping_cost.nlargest(5)
        top_vendors_by_continent[continent] = top_vendors
        
        # Plotting bar graph for each continent
        plt.figure(figsize=(10, 6))
        top_vendors.plot(kind='bar', color='orange')
        plt.title(f"Top 5 Vendors by Shipping Cost in {continent}")
        plt.xlabel("Vendor")
        plt.ylabel("Total Shipping Cost (USD)")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    return top_vendors_by_continent

# Assuming your filtered DataFrame is named 'filtered_df'
top_vendors_by_shipping_cost_per_continent = top_vendors_by_shipping_cost_per_continent(df)
print(top_vendors_by_shipping_cost_per_continent)


SCMS from RDC  is the top vendor in the continents - Africa,North AMerica, SOuth AMerica. 


# In[ ]:


def top_vendors_by_frequency(df):
    # Counting occurrences of each vendor and getting top 5
    top_vendors = df['vendor'].value_counts().head(5)

    # Plotting horizontal bar graph
    plt.figure(figsize=(10, 6))
    ax = top_vendors.plot(kind='barh', color='skyblue')
    plt.title("Top 5 Most Frequent Vendors", fontweight='bold')
    plt.xlabel("Frequency", fontweight='bold')
    plt.ylabel("Vendor", fontweight='bold')

    # Adding value labels in the center of each bar and bolding the text
    for i in ax.patches:
        ax.text(i.get_width()/2, i.get_y() + 0.5*i.get_height(), str(round(i.get_width(), 2)),
                va='center', ha='center', color='black', fontweight='bold')

    plt.tight_layout()
    plt.show()

    return top_vendors

# Assuming your DataFrame is named 'df'
top_vendors = top_vendors_by_frequency(df)
print(top_vendors)


# In[ ]:


import matplotlib.pyplot as plt

def transactions_per_continent(df):
    # Counting the number of transactions (rows) per continent
    transaction_counts = df['continent'].value_counts()

    # Plotting pie chart
    fig, ax = plt.subplots(figsize=(8, 8))  # Create a figure and a set of subplots
    wedges, texts, autotexts = ax.pie(transaction_counts, labels=transaction_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired(range(len(transaction_counts))), wedgeprops={'linewidth': 2, 'edgecolor': 'black'})

    # Set font properties for all text
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    # Set overall title
    plt.title('Transactions Per Continent', fontweight='bold')

    # Add a bold outline to the circle
    ax.set_frame_on(True)  # Turn on the frame
    ax.spines['geo'].set_linewidth(2)  # Set the width of the outline
    ax.spines['geo'].set_color('black')  # Set the color of the outline

    # Show plot
    plt.show()

    return transaction_counts

# Assuming your DataFrame is named 'df' and has a 'continent' column with the continent names
# You would call the function like this:
transaction_counts = transactions_per_continent(df)
print(transaction_counts)


plt.savefig('/Users/USER/Downloads/Scripting Images/',transparent=True)  # Save with transparency
    plt.show()


# In[ ]:


def transactions_per_continent(df):
    # Counting the number of transactions (rows) per continent
    transaction_counts = df['continent'].value_counts()
    
    # Plotting pie chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(aspect="equal"))  # Ensures the plot is square
    wedges, texts, autotexts = ax.pie(transaction_counts, labels=transaction_counts.index, autopct='%1.1f%%', startangle=140,
                                      colors=plt.cm.Set3(range(len(transaction_counts))), # Using Set3 colormap for vibrant, distinct colors
                                      wedgeprops={'linewidth': 1, 'edgecolor': 'black'})

    # Set font properties for all text
    for text in texts:
        text.set_fontweight('bold')
    for autotext in autotexts:
        autotext.set_fontweight('bold')

    # Set overall title
    plt.title('Transactions Per Continent', fontweight='bold')

    # Making the spines (frame of the plot) bold and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_edgecolor('black')

    # Show plot
    plt.tight_layout()
    plt.show()

    # Save the figure
    fig.savefig('Desktop/transactions_per_continent.png')  # Adjust path as necessary

    return transaction_counts

# Assuming your DataFrame is named 'df' and has a 'continent' column with the continent names
# You would call the function like this:
transaction_counts = transactions_per_continent(df)
print(transaction_counts)


# In[ ]:


os.getcwd()


# In[ ]:


df.info()


# In[ ]:




