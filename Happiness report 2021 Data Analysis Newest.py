#!/usr/bin/env python
# coding: utf-8

# ## Data Analysis on World Happiness 
# 
# **Student Name:** Chin Wei Chung
# 
# **Student ID:** 77350057
# 
# **Date:** 25/5/2023

# ### Introduction
# In this project, we will explore and analyze the World Happiness Report dataset for the year 2021 using programming tools and techniques.
# 
# The World Happiness Report is a comprehensive survey that measures subjective well-being and happiness levels of countries around the world. It takes into account various factors such as GDP per capita, social support, life expectancy, freedom to make life choices, generosity, and perceptions of corruption. The report provides valuable insights into the factors that contribute to happiness and well-being on a global scale.
# 
# ### Research Question
# 
# Here are some research questions that we can explore in my project related to the World Happiness Report 2021:
# 
# - How do the happiness scores vary across different countries in the world?
# 
# - Which countries have the highest and lowest happiness scores in the World Happiness Report 2021?
# 
# - What are the main factors contributing to the happiness scores of different countries?
# 
# - How does the GDP per capita correlate with happiness scores? Is there a clear relationship?
# 
# - Are there any regional patterns or differences in happiness scores among countries?
# 
# - What is the relationship between social support and happiness scores? How significant is this factor?
# 
# - Does life expectancy have a strong impact on happiness levels? How does it vary across countries?
# 
# - Is there a relationship between freedom to make life choices and happiness scores?
# 
# - How does generosity among individuals in a country impact its overall happiness score?
# 
# - Does the perception of corruption affect the happiness levels of a country's residents?
# 
# These research questions will allow you to explore various aspects of the World Happiness Report 2021 dataset and gain insights into the factors influencing happiness levels across different countries. You can analyze the data, create visualizations, and draw conclusions to answer these research questions in your project.
# 
# ### Data Description
# 
# For my data analysis, i have used the World Happiness Report 2021 datasets obtained from:
# 
# - https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021
# 

# ### Importing packages required

# In[1]:


import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas
import holoviews as hv
import seaborn as sns
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
import ipywidgets as widgets
from IPython.display import display
pn.extension('tabulator', sizing_mode="stretch_width")
hv.extension('bokeh')

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Customizing Plot Style and Appearance

# In[2]:


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['figure.facecolor'] = '#FFE5B4'


# ### Importing the dataset

# In[3]:


data = pd.read_csv('C:\\Users\\ccalv\\Desktop\\ADS Sem 1\\Computer Programming\\Project\\world-happiness-report-2021.csv')


# In[4]:


data.head()


# ### Data Cleaning
# 
# - **Dropping Unwanted Columns:**
# We start by creating a list called data_columns that contains the names of the columns we want to keep in our dataset. These columns are the ones that provide valuable information for our analysis. Any columns not included in this list are considered unwanted and will be dropped from the dataset.
# 
# Next, we use the data_columns list to select only the desired columns from the original dataset, using the .copy() method to create a new copy of the DataFrame. This ensures that any modifications made to the new DataFrame do not affect the original dataset.
# 
# 
# - **Changing Variable Names:**
# After selecting the desired columns, we proceed to change the variable names to make them more descriptive and easier to work with. We use the .rename() method on the DataFrame data and provide a dictionary where the keys are the original column names, and the values are the new column names we want to assign.
# 
# By renaming the variables, we can improve the clarity of the dataset and make it easier to refer to specific columns during analysis.
# 
# Finally, we display the first few rows of the modified DataFrame, now named happy_df, using the .head() method to verify that the changes have been applied correctly. This portion of the data cleaning process ensures that we have a cleaned and transformed dataset with the desired columns and more user-friendly variable names, setting the stage for further analysis and visualization.

# #### Dropping unwanted columns from the dataset

# In[5]:


data_columns = ['Country name', 'Regional indicator', 'Happiness score', 'Logged GDP per capita', 'Social support', 'Healthy life expectancy', 'Freedom to make life choices', 'Generosity', 'Perceptions of corruption']


# In[6]:


data = data[data_columns].copy()


# #### changing the variable names

# In[7]:


happy_df = data.rename(columns={
    'Country name':'country_name', 
    'Regional indicator':'regional_indicator', 
    'Happiness score':'happiness_score', 
    'Logged GDP per capita':'logged_GDP_per_capita', 
    'Social support':'social_support', 
    'Healthy life expectancy':'healthy_life_expectancy', 
    'Freedom to make life choices':'freedom_to_make_life_choices', 
    'Generosity':'generosity', 
    'Perceptions of corruption':'perceptions_of_corruption'
})


# In[8]:


happy_df.head()


# ### Data Columns
# 
# - **country_name**: The name of the country included in the World Happiness Report 2021 dataset.
# 
# - **regional_indicator**: The regional indicator categorizing the country into a specific region based on geographical location or shared characteristics.
# 
# - **happiness_score**: The happiness score representing the subjective well-being and happiness level of a country.
# 
# - **logged_GDP_per_capita**: The logged GDP per capita, a measure of the economic output per person in the country adjusted for inflation and population.
# 
# - **social_support**: The perceived social support available to individuals in the country, indicating the extent to which social relationships contribute to happiness.
# 
# - **healthy_life_expectancy**: The healthy life expectancy at birth, representing the average number of years a newborn is expected to live in good health.
# 
# - **freedom_to_make_life_choices**: The perceived freedom to make life choices, indicating the level of personal agency and control over life decisions.
# 
# - **generosity**: The level of generosity of individuals in the country, measuring acts of kindness, altruism, and charitable behavior.
# 
# - **perceptions_of_corruption**: The perceptions of corruption in the country, reflecting the perceived level of governmental and institutional corruption.
# 
# 
# These variables provide crucial information for analyzing and understanding the factors influencing happiness scores across different countries. They represent various aspects such as economic well-being, social support, health, personal freedom, generosity, and perceptions of corruption. Analyzing these variables can help uncover patterns, relationships, and insights into the factors contributing to happiness levels worldwide.

# ### Counting Missing Values in DataFrame
# 
# To examine the presence of missing values in the `happy_df` DataFrame, we can use the `isnull()` function followed by the `sum()` function. This combination allows us to count the number of missing values in each column.
# 
# The resulting output shows the count of missing values for each variable in the `happy_df` DataFrame:
# 
# 

# In[9]:


happy_df.isnull().sum()


# ## Data Analysis

# ### World Happiness Index
# 
# This code uses the Plotly library to create a choropleth map based on the data in the happy_df dataframe.
# 
# - The `plot_ly()` function initializes the plot object.
# 
# 
# - `locations` specifies the column in the dataframe that contains the country names.
# 
# 
# - `locationmode` sets the mode for interpreting the locations as country names.
# 
# 
# - `z` represents the numeric variable to be color-coded on the map (in this case, the happiness score).
# 
# 
# - `type` is set to `'choropleth'` to create a choropleth map.
# 
# 
# - `colors` specifies the color scale to use for the map.
# 
# 
# - `colorbar` configures the colorbar on the map, with a specified title.
# 
# 
# - `showscale` is set to `TRUE` to display the color scale on the map.
# 
# 
# The `layout()` function is used to adjust the map's layout and appearance. Key settings include:
# 
# 
# - `title` sets the title of the map.
# 
# 
# - `geo` configures the geographic properties of the map, such as the background color, land color, projection type, ocean   color and lake color.
# 
# 
# - `margin` specifies the margins around the map plot.

# In[10]:


# Create a choropleth map using Plotly
fig = px.choropleth(happy_df, 
                    locations='country_name', 
                    locationmode='country names', 
                    color='happiness_score', 
                    color_continuous_scale=px.colors.sequential.Plasma,
                    scope='world')

# Adjust the map layout
fig.update_layout(
    title='World Happiness Index',
    geo=dict(
        bgcolor='rgba(0,0,0,0)',
        showland=True, 
        landcolor='rgb(217, 217, 217)',
        projection_type='equirectangular',
        showocean=False, 
        oceancolor='rgb(12, 28, 63)',
        showlakes=False, 
        lakecolor='rgb(12, 28, 63)'
    ),
    margin=dict(l=0, r=0, t=50, b=0)  # Adjust the top margin to make space for the title
)

# Display the map
fig.show()


# ### Correlation Heatmap
# 
# This code below calculates and visualizes the correlation matrix for the numeric columns in the happy_df dataframe using the seaborn library.
# 
# 
# - The code first selects only the numeric columns from the `happy_df` DataFrame using the `select_dtypes()` method. The `include=[float, int]` argument ensures that only columns with data types of `float` or `int` are considered as numeric.
# 
# 
# - The resulting numeric column names are stored in the `numeric_columns` variable.
# 
# 
# - The correlation matrix is calculated using the `corr()` method on the numeric columns of the `happy_df` DataFrame.
# 
# 
# - The resulting correlation matrix is stored in the `correlation_matrix` variable.
# 
# 
# - A new figure with a size of 10x8 inches is created using `plt.figure(figsize=(10, 8))`.
# 
# 
# - The `sns.heatmap()` function from the Seaborn library is called to create the heatmap of the correlation matrix.
# 
# 
# - The correlation matrix (`correlation_matrix`) is passed as the data for the heatmap.
# 
# 
# - `annot=True` adds numeric annotations to the heatmap cells, showing the correlation values.
# 
# 
# - The colormap `'coolwarm'` is used to represent the correlation values, with cool colors for negative correlations and warm colors for positive correlations.
# 
# 
# - The `plt.title()` function sets the title of the plot to 'Correlation Matrix - World Happiness Report 2021'.
# 
# 
# 
# 

# In[11]:


# Select only the numeric columns for correlation matrix calculation
numeric_columns = happy_df.select_dtypes(include=[float, int]).columns
correlation_matrix = happy_df[numeric_columns].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix - World Happiness Report 2021')
plt.show()


# ### Scatterplot between Happiness and GDP
# 
# This code creates a scatter plot to show the relationship between the Happiness Score and GDP per capita variables in the happy_df dataframe using the `matplotlib.pyplot` and `seaborn` packages.
# 
# 
# - `plt.rcParams['figure.figsize'] = (15, 7)`: This line sets the figure size for the plot using the rcParams dictionary in matplotlib.pyplot (plt). It specifies a width of 15 inches and a height of 7 inches for the plot.
# 
# - `plt.title('Plot between Happiness Score and GDP')`: This line sets the title of the plot to "Plot between Happiness Score and GDP" using the title() function from matplotlib.pyplot (plt). It adds a title to the plot.
# 
# - `sns.scatterplot(x = happy_df.happiness_score, y = happy_df.logged_GDP_per_capita, hue = happy_df.regional_indicator, s = 200)`: This line creates a scatter plot using the `scatterplot()` function from `seaborn` (`sns`). It specifies the x-axis data as the `happiness_score` column of the `happy_df` DataFrame, the y-axis data as the `logged_GDP_per_capita` column of the same DataFrame, and the hue (color) of the points based on the `regional_indicator` column. The `s` parameter sets the size of the points to 200.
# 
# - `plt.legend(loc = 'upper left', fontsize = '10')`: This line adds a legend to the plot using the `legend()` function from `matplotlib.pyplot` (`plt`). It specifies the location of the legend to be in the upper left corner of the plot and sets the font size of the legend text to 10.
# 
# - `plt.xlabel('Happiness Score')`: This line sets the x-axis label of the plot to "Happiness Score" using the `xlabel()` function from `matplotlib.pyplot` (`plt`). It adds a label to the x-axis.
# 
# - `plt.ylabel('GDP per capita')`: This line sets the y-axis label of the plot to "GDP per capita" using the `ylabel()` function from `matplotlib.pyplot` (`plt`). It adds a label to the y-axis.
# 
# 

# In[12]:


# Plot between happiness and GDP

plt.rcParams['figure.figsize'] = (15, 7)
plt.title('Plot between Happiness Score and GDP')
sns.scatterplot(x = happy_df.happiness_score, y = happy_df.logged_GDP_per_capita, hue = happy_df.regional_indicator, s = 200);

plt.legend(loc = 'upper left', fontsize = '10')
plt.xlabel('Happiness Score')
plt.ylabel('GDP per capita')


# ### Total Logged GDP Per Capita for each regional indicators
# 
# To calculate the total logged GDP per capita for each regional indicator in the `happy_df` DataFrame, the code utilizes the `groupby()` function in conjunction with the `sum()` function.
# 
# The resulting output displays the total logged GDP per capita for each regional indicator:
# 
# 
# 
# The code groups the DataFrame `happy_df` by the `regional_indicator` column and calculates the sum of the `logged_GDP_per_capita` column within each group. This provides the total logged GDP per capita for each regional indicator.
# 
# By analyzing the total logged GDP per capita across different regions, we can gain insights into the economic well-being and prosperity of various geographical areas covered in the World Happiness Report 2021 dataset.
# 

# In[13]:


gdp_region = happy_df.groupby('regional_indicator')['logged_GDP_per_capita'].sum()
print(gdp_region)


# ### Pie Chart showing the GDP by Regional Indicators
# 
# To visualize the distribution of GDP by region, the code utilizes the `plot.pie()` method from the `gdp_region` data obtained in the previous code block. This method generates a pie chart representing the proportion of GDP contributed by each region.
# 

# In[14]:


gdp_region.plot.pie(autopct = '%1.1f%%')
plt.title('GDP by Region')
plt.ylabel('')


# ### Total Countries Per region
# 
# To calculate the total number of countries within each regional indicator, the code below uses the `groupby()` function on the `happy_df` DataFrame. The `groupby()` function groups the data based on the 'regional_indicator' column.
# 
# The resulting output displays the count of countries for each regional indicator:
# 
# 
# The code groups the DataFrame `happy_df` by the 'regional_indicator' column and then selects the 'country_name' column within each group. The `count()` function is then applied to calculate the number of countries within each regional indicator.
# 
# By analyzing the total number of countries within each region, we gain insights into the distribution and representation of countries across different regional indicators in the World Happiness Report 2021 dataset.
# 
# 
# 
# 

# In[15]:


# Total countries

total_country = happy_df.groupby('regional_indicator')[['country_name']].count()
print(total_country)


# ### Perception of Corruption in different regions
# 
# To analyze the perceptions of corruption across different regions, the code uses the `groupby()` function on the `happy_df` DataFrame. It groups the data based on the 'regional_indicator' column.
# 
# The resulting output displays the average perceptions of corruption for each regional indicator:
# 
# 
# The code groups the DataFrame `happy_df` by the 'regional_indicator' column and then selects the 'perceptions_of_corruption' column within each group. The `mean()` function is then applied to calculate the average perceptions of corruption for each region.
# 
# This analysis provides insights into the varying levels of perceived corruption across different regional indicators in the World Happiness Report 2021 dataset. The average perceptions of corruption values range from 0 to 1, where higher values indicate a higher perception of corruption.
# 
# By examining these average values, we can compare and understand the relative levels of perceived corruption in different regions, shedding light on the trustworthiness and transparency of institutions and governance practices.
# 

# In[16]:


# Corruption in different regions

corruption = happy_df.groupby('regional_indicator')[['perceptions_of_corruption']].mean()
print(corruption)


# ### Bar Chart showing the perception of corruption among various regions
# 
# To visualize the perceptions of corruption in various regions, the code below utilizes several plotting functions from the `matplotlib.pyplot` module.
# 

# In[17]:


plt.rcParams['figure.figsize'] = (12, 8)
plt.title('Perception of Corruption in various regions')
plt.xlabel('Regions', fontsize = 15)
plt.ylabel('Corruption Index', fontsize = 15)
plt.xticks(rotation = 30, ha = 'right') 
plt.bar(corruption.index, corruption['perceptions_of_corruption'], color=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'grey', 'black'])



# ### Top 10 and Bottom 10 Countries Happiness score
# 

# #### Creating 2 new dataframes
# 
# - top_10: Represents the top (or highest) 10 rows based on the index order of the `happy_df` DataFrame.
# 
# - bottom_10: Represents the bottom (or loweest) 10 rows based on the index order of the `happy_df1 Dataframe.

# In[18]:


top_10 = happy_df.head(10)
bottom_10 = happy_df.tail(10)


# ### Bar Chart of the life expectancy of the top 10 happiest and bottom 10  least happiest countries 
# 
# To compare the life expectancy of the top 10 happiest countries and the bottom 10 least happy countries, the code uses `plt.subplots()` to create a figure with two subplots arranged side by side. The figsize parameter specifies the size of the figure as (16, 6).
# 
# The `plt.tight_layout(pad=2)` statement adds spacing between the subplots for better visual presentation.
# 
# For the first subplot:
# - The x-axis labels are set as the country names of the top 10 happiest countries (`top_10["country_name"]`).
# - The title is set as "Top 10 happiest countries Life Expectancy" using `axes[0].set_title()`.
# - The x-tick labels are rotated by 45 degrees and aligned to the right for readability.
# - A bar plot is created using `sns.barplot()` with the country names on the x-axis and the corresponding life expectancies (`top_10["healthy_life_expectancy"]`) on the y-axis.
# - The x-axis label is set as "Country name" using `axes[0].set_xlabel()`, and the y-axis label is set as "Life expectancy" using `axes[0].set_ylabel()`.
# 
# For the second subplot:
# - The same steps are followed as for the first subplot, but with the country names and life expectancies of the bottom 10 least happy countries (`bottom_10["country_name"]` and `bottom_10["healthy_life_expectancy"]`).
# 
# By creating these subplots with bar plots, we can visually compare the life expectancies of the top 10 happiest countries and the bottom 10 least happy countries.
# 
# 

# In[19]:


fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.tight_layout(pad=2)


xlabels = bottom_10["country_name"]
axes[1].set_title('Bottom 10 least happy countries Life Expectancy')
axes[1].set_xticks(range(len(xlabels)))
axes[1].xaxis.set_major_locator(FixedLocator(range(len(xlabels))))
axes[1].set_xticklabels(xlabels, rotation=45, ha='right')
sns.barplot(x=bottom_10["country_name"], y=bottom_10["healthy_life_expectancy"], ax=axes[1])
axes[1].set_xlabel('Country name')
axes[1].set_ylabel('Life expectancy')

xlabels = top_10["country_name"]
axes[0].set_title('Top 10 happiest countries Life Expectancy')
axes[0].set_xticks(range(len(xlabels)))
axes[0].xaxis.set_major_locator(FixedLocator(range(len(xlabels))))
axes[0].set_xticklabels(xlabels, rotation=45, ha='right')
sns.barplot(x=top_10["country_name"], y=top_10["healthy_life_expectancy"], ax=axes[0])
axes[0].set_xlabel('Country name')
axes[0].set_ylabel('Life expectancy')


# ### Scatterplot between happiness and freedom to make life choices
# 
# The code below generates a scatter plot to explore the relationship between the freedom to make life choices and the happiness score of countries in the `happy_df` DataFrame. Here's a breakdown of the code:
# 
# 

# In[20]:


plt.rcParams['figure.figsize'] = (15, 7)
sns.scatterplot(x=happy_df['freedom_to_make_life_choices'], y=happy_df['happiness_score'], hue=happy_df['regional_indicator'], s=200)
plt.legend(loc = 'upper left', fontsize = '12')
plt.xlabel('Freedom to  make life choices')
plt.ylabel('Happiness Score')


# ### Bar chart of countries with the highest and lowest perception of corruption
# 
# The code below creates two bar plots to display the countries with the least and most perception of corruption based on the 'perceptions_of_corruption' column in the `happy_df` DataFrame. Here's a breakdown of the code:
# 

# #### Countries with lowest perception of corruption:

# In[21]:


country = happy_df.sort_values(by = 'perceptions_of_corruption').head(10)
plt.rcParams['figure.figsize'] = (12, 6)
plt.title('countries with the least perception of Corruption')
plt.xlabel('Country', fontsize = 13)
plt.ylabel('Corruption Index', fontsize = 13)
plt.xticks(rotation = 30, ha = 'right')
plt.bar(country["country_name"], country["perceptions_of_corruption"], color=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'grey', 'black'])


# #### countries witht the highest perception of corruption:
# 

# In[22]:


country = happy_df.sort_values(by = 'perceptions_of_corruption').tail(10)
plt.rcParams['figure.figsize'] = (12, 6)
plt.title('countries with the most perception of Corruption')
plt.xlabel('Country', fontsize = 13)
plt.ylabel('Corruption Index', fontsize = 13)
plt.xticks(rotation = 30, ha = 'right')
plt.bar(country["country_name"], country["perceptions_of_corruption"], color=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet', 'brown', 'grey', 'black'])


# ### Scatterplot between perception of corruption and happiness score
# 
# The code below creates a scatter plot to explore the relationship between happiness score and perceptions of corruption in the `happy_df` DataFrame. Here's a breakdown of the code:
# 
# 

# In[23]:


# corruption vs happiness

plt.rcParams['figure.figsize'] = (15, 7)
sns.scatterplot(x = happy_df["happiness_score"], y = happy_df["perceptions_of_corruption"], hue = happy_df["regional_indicator"], s = 200)
plt.legend(loc = 'lower left', fontsize = '14')
plt.xlabel('Happiness Score')
plt.ylabel('Corruption')


# ### 3D scatterplot of 3 variables and its effect on happiness
# 
# - x, y, and z are assigned the corresponding columns from the happy_df DataFrame: 'logged_GDP_per_capita', 'happiness_score', and 'healthy_life_expectancy'.
# 
# 
# - `go.Scatter3d()` from the `plotly.graph_objects` module is used to create a 3D scatter plot. The x, y, and z values represent the variables for the three axes. The mode parameter is set to 'markers' to create a scatter plot.
# 
# - `marker=dict(size=5, color=z, colorscale='Viridis', opacity=0.8)` defines the markers' size, color, and opacity. The color of the markers is determined by the 'healthy_life_expectancy' values, with the 'Viridis' color scale used. The opacity is set to 0.8 for transparency.
# 
# - `fig.update_layout()` is used to set the axes labels and the plot title. The `scene` parameter is used to define the labels for the x, y, and z axes.
# 
# 
# By visualizing these three variables in a 3D scatter plot, we can explore the relationships and patterns between GDP per capita, happiness score, and healthy life expectancy. The plot provides a comprehensive view of the data in three dimensions and enables a deeper understanding of how these variables interact.

# In[30]:


# Select three variables for the 3D plot
x = happy_df['logged_GDP_per_capita']
y = happy_df['happiness_score']
z = happy_df['healthy_life_expectancy']

# Create a 3D scatter plot
fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=z,  # Color points based on 'Healthy life expectancy'
        colorscale='Viridis',
        opacity=0.8
    )
)])

# Set axes labels and plot title
fig.update_layout(
    scene=dict(
        xaxis_title='GDP per capita',
        yaxis_title='Happiness score',
        zaxis_title='Healthy_life_expectancy'
    ),
    title='World Happiness Report 2021 - 3D Plot'
)

# Show the 3D plot
fig.show()


# ### Mean Happiness Score
# 
# The code below groups the data by regional indicators and calculates the mean happiness score for each region. It then creates a bar plot to visualize the average happiness score by regional indicators.
# 
# - `region_scores` is created by grouping the `happy_df` DataFrame by the 'regional_indicator' column and calculating the mean of the 'happiness_score' column using `.groupby('regional_indicator')['happiness_score'].mean()`. The `reset_index()` function is then used to convert the resulting Series back to a DataFrame.
# 
# - `px.bar()` from the `plotly.express` module is used to create a bar plot. The `region_scores` DataFrame is specified as the data source. The 'regional_indicator' column is assigned to the x-axis (`x='regional_indicator'`), and the 'happiness_score' column is assigned to the y-axis (`y='happiness_score'`). The `title` parameter sets the title of the bar plot, and the `labels` parameter is used to customize the axis labels.
# 
# By visualizing the average happiness scores by regional indicators in a bar plot, we can compare the happiness levels across different regions. The plot provides a clear overview of the average happiness scores and facilitates insights into regional variations in happiness.

# In[25]:


# Group the data by regional indicators and calculate the mean happiness score
region_scores = happy_df.groupby('regional_indicator')['happiness_score'].mean().reset_index()

# Create the bar plot
fig = px.bar(region_scores, x='regional_indicator', y='happiness_score', 
             title='Average Happiness Score by Regional Indicators',
             labels={'Regional indicator': 'Regional Indicator', 'Happiness score': 'Average Happiness Score'})

# Show the bar plot
fig.show()


# ### Line Plot  showing each countries happiness score in ascending order
# 
# The code below generates a line graph to visualize the happiness scores for each country in the `happy_df` DataFrame. Here's a breakdown of the code:
# 
# - `year = 2021` sets the year for the dataset.
# 
# - `happy_df = happy_df.sort_values('happiness_score')` sorts the `happy_df` DataFrame in ascending order based on the 'happiness_score' column. This ensures that the line graph displays the countries in increasing order of happiness score.
# 
# - `fig = px.line()` from the `plotly.express` module creates a line graph. The `happy_df` DataFrame is specified as the data source. The 'country_name' column is assigned to the x-axis (`x='country_name'`), and the 'happiness_score' column is assigned to the y-axis (`y='happiness_score'`). The `title` parameter sets the title of the line graph, and the `labels` parameter is used to customize the axis labels.
# 
# - `fig.update_layout(xaxis_tickangle=-45)` adjusts the angle of the x-axis tick labels to -45 degrees for better readability.
# 

# In[26]:


# Set the year for the dataset
year = 2021

# Sort the data by ascending order of happiness score
happy_df = happy_df.sort_values('happiness_score')

# Create the line graph
fig = px.line(happy_df, x='country_name', y='happiness_score', 
              title=f'Happiness Score for the Year {year}', 
              labels={'country_name': 'Country', 'happiness_score': 'Happiness Score'})

# Customize the layout
fig.update_layout(xaxis_tickangle=-45)

# Show the line graph
fig.show()
     






# ### Line plot showing the happiness score of each countries in each regions:
# 
# The code below generates a line graph to visualize the happiness scores for each country in the `happy_df` DataFrame, grouped by regional indicators.
# 
# - `year = 2021` sets the year for the dataset.
# 
# - `happy_df = happy_df.sort_values('happiness_score')` sorts the `happy_df` DataFrame in ascending order based on the 'happiness_score' column. This ensures that the line graph displays the countries in increasing order of happiness score.
# 
#  - `fig = px.line()` from the `plotly.express` module creates a line graph. The `happy_df` DataFrame is specified as the data source. The 'country_name' column is assigned to the x-axis (`x='country_name'`), and the 'happiness_score' column is assigned to the y-axis (`y='happiness_score'`). The `color` parameter is set to 'regional_indicator' to group the lines by regional indicators. The `title` parameter sets the title of the line graph, and the `labels` parameter is used to customize the axis labels.
#  
# - `fig.update_layout()` is used to customize the layout of the graph. The `xaxis_tickangle` parameter sets the rotation angle of the x-axis tick labels to -50 degrees for better readability. The `legend` parameter is used to customize the legend, setting the title as 'Regional Indicator', orienting it vertically (`orientation='v'`), and adjusting its position outside the plot area at the bottom left corner. The `yaxis` parameter sets the title of the y-axis as 'Happiness Score', and the `tickmode` and `dtick` parameters customize the y-axis ticks.
# 

# In[27]:


# Set the year for the dataset
year = 2021

# Sort the data by ascending order of happiness score
happy_df = happy_df.sort_values('happiness_score')

# Create the line graph
fig = px.line(happy_df, x='country_name', y='happiness_score', color='regional_indicator',
              title=f'Happiness Score for the Year {year} by Regional Indicator',
              labels={'country_name': 'Country', 'happiness_score': 'Happiness Score', 'regional_indicator': 'Regional Indicator'})

# Customize the layout
fig.update_layout(
    xaxis_tickangle=-50,
    legend=dict(
        title='Regional Indicator',
        orientation='v',  # Set orientation to vertical
        yanchor='bottom',  # Anchor to the top
        y=0,  # Adjust the position outside the plot area
        xanchor='left',
        x=1  # Adjust the position outside the plot area
    ),
    yaxis=dict(
        title='Happiness Score',
        tickmode='linear',
        dtick=1
    )
)

# Show the line graph
fig.show()




# ### Box plot showing the relationship between happiness and regions
# 
# The code below generates a line graph to visualize the happiness scores for each country in the `happy_df` DataFrame, grouped by regional indicators.
# 
# - `year = 2021` sets the year for the dataset.
# 
# - `happy_df = happy_df.sort_values('happiness_score')` sorts the `happy_df` DataFrame in ascending order based on the 'happiness_score' column. This ensures that the line graph displays the countries in increasing order of happiness score.
# 
# - `fig = px.line()` from the `plotly.express` module creates a line graph. The `happy_df` DataFrame is specified as the data source. The 'country_name' column is assigned to the x-axis (`x='country_name'`), and the 'happiness_score' column is assigned to the y-axis (`y='happiness_score'`). The `color` parameter is set to 'regional_indicator' to group the lines by regional indicators. The `title` parameter sets the title of the line graph, and the `labels` parameter is used to customize the axis labels.
# 
# - `fig.update_layout()` is used to customize the layout of the graph. The `xaxis_tickangle` parameter sets the rotation angle of the x-axis tick labels to -50 degrees for better readability. The `legend` parameter is used to customize the legend, setting the title as 'Regional Indicator', orienting it vertically (`orientation='v'`), and adjusting its position outside the plot area at the bottom left corner. The `yaxis` parameter sets the title of the y-axis as 'Happiness Score', and the `tickmode` and `dtick` parameters customize the y-axis ticks.
# 

# In[28]:


# Create the box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=happy_df, x='regional_indicator', y='happiness_score')
plt.title('Relationship between Happiness Score and Regional Indicator')
plt.xlabel('Regional Indicator')
plt.ylabel('Happiness Score')

# Rotate x-axis labels if needed
plt.xticks(rotation=45, ha='right')

# Show the plot
plt.show()


# ### Conclusion
# 
# Based on the analysis and visualizations of the World Happiness Report 2021, the following key findings and insights can be concluded:
# 
# 1. **Regional Variations**: The happiness scores varied significantly across different regions. The report categorized countries into regional indicators, such as Central and Eastern Europe, Commonwealth of Independent States, East Asia, Latin America and Caribbean, Middle East and North Africa, North America and ANZ, South Asia, Southeast Asia, Sub-Saharan Africa, and Western Europe. These regional indicators provided insights into regional differences in happiness levels.
# 
# 2. **Economic Factors**: The analysis revealed a positive correlation between GDP per capita and happiness score. Countries with higher GDP per capita generally exhibited higher happiness scores. This suggests that economic prosperity plays a significant role in determining happiness levels.
# 
# 3. **Social Support**: The availability of social support emerged as a crucial factor influencing happiness. Countries with higher social support scores tended to have higher happiness scores. Strong social networks and supportive relationships contribute to overall well-being and happiness.
# 
# 4. **Health and Well-being**: Healthy life expectancy showed a positive association with happiness scores. Countries with longer life expectancies and better overall health reported higher happiness levels. Good health and well-being are important contributors to individual and societal happiness.
# 
# 5. **Freedom and Life Choices**: Countries where individuals had more freedom to make life choices tended to have higher happiness scores. The ability to exercise personal agency and autonomy in decision-making positively impacts happiness levels.
# 
# 6. **Generosity and Perceptions of Corruption**: The analysis highlighted the role of generosity and perceptions of corruption. Countries with higher levels of generosity tended to have higher happiness scores. Conversely, perceptions of corruption showed a negative correlation with happiness. Trust and integrity in governance and institutions are important for overall happiness.
# 
# 7. **Top and Bottom Performers**: The report identified the top and bottom countries in terms of happiness scores. It is crucial to further explore the factors contributing to the high happiness scores of top-performing countries and address the challenges faced by countries with lower happiness scores.
# 
# Overall, the World Happiness Report 2021 provides valuable insights into the factors influencing happiness levels across countries and regions. The analysis highlights the importance of economic well-being, social support, health, freedom, generosity, and perceptions of corruption in determining happiness. The findings can guide policymakers and stakeholders in developing strategies and policies to enhance happiness and well-being at individual, community, and societal levels.
# 
# 

# In[ ]:




