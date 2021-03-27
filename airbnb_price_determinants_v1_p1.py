#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
from numpy import unique
%matplotlib inline

#Reading data set
airbnb = pd.read_csv("listings.csv")
#First 5 observation displays
print(airbnb.head(5))

#Detailed representation these features
airbnb.info()

#Data set has 24519 samples(rows) and 74 features(columns)
airbnb.shape

#Displaying these columns meanly features
airbnb.columns

#Basic descriptive statistics for features
airbnb.describe().T

#Slightly more detailed descriptive statistics for features
airbnb.describe(include = "all").T

#Relationships of features with each other by correlation
airbnb.corr()

#Visual representation of the properties' relationships with each other by correlation
corr=airbnb.corr()

#Display price
print(airbnb['price'])
h=sns.heatmap(corr,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(20,220,n=200),square=True)

#Plot price distribution
plt.figure(figsize=(6,6))
sns.boxplot(y=airbnb['price'])
plt.title("Distribution of Price")
plt.show()

plt.hist(airbnb['price'])
plt.show()

#Function for price distribution
def price_distribution(cols):
    Price = cols
    count=0
    lower_limit=90000
    upper_limit=100000
    
    if (Price<upper_limit and Price>lower_limit):
        count=count+1
        
    return count
    
 # Apply the function to the price column
num=airbnb['price'].apply(price_distribution)
print(num)

#If the price is in the desired range, the function result will be 1, if not result will be 0
num.value_counts()

#Number of data with price equal to 10000
(airbnb['price']==10000).value_counts()

#Number of data with price equal to 0
(airbnb['price']==0).value_counts()

airbnb=airbnb[0<airbnb.price]

airbnb.shape   
airbnb=airbnb[airbnb.price<10000]
airbnb.shape

#Exploratory Data Analysis

target_columns=["id","listing_url","scrape_id","last_scraped","name","description","neighborhood_overview","picture_url",
"host_id","host_url","host_name","host_since","host_location","host_about","host_thumbnail_url","host_picture_url",
"host_neighbourhood","host_verifications","calendar_updated","calendar_last_scraped","first_review","last_review",
"license","amenities"]
airbnb.drop(target_columns,axis=1,inplace=True)

#Handling Missing Data

#Are there any missing observations (values)
airbnb.isnull().values.any()

#In which variable how many
airbnb.isnull().sum()

#Completely empty or almost empty columns should be dropped
airbnb.drop(["neighbourhood","neighbourhood_group_cleansed","bathrooms"],axis=1,inplace=True)

#Filling missing values

#Some features have missing data:
sns.heatmap(airbnb.isnull(), yticklabels = False, cbar = False)
#White color indicates missing values

airbnb['reviews_per_month'] = airbnb['number_of_reviews'] / 12

reviews = ['review_scores_value', 'review_scores_cleanliness', 'review_scores_location', 'review_scores_accuracy', 'review_scores_communication', 'review_scores_checkin', 'review_scores_rating']
for i in reviews:
  airbnb[i].fillna(airbnb[i].mean(), inplace=True)

#Selecting categorical data
a=airbnb.copy()
categorical_a = a.select_dtypes(include = ["object"])

#Total class numbers of host_response_time categorical data
print(categorical_a["host_response_time"].value_counts().count())
categorical_a["host_response_time"].value_counts()

host_columns = ['host_response_time', 'host_response_rate','host_acceptance_rate','host_is_superhost','host_listings_count',
'host_total_listings_count','host_has_profile_pic','host_identity_verified']

for i in host_columns:
  airbnb[i].fillna(airbnb[i].value_counts().idxmax(), inplace=True)
  
airbnb.info()

cols = ['beds', 'bedrooms','bathrooms_text']
for i in cols:
  airbnb[i].fillna(airbnb[i].mean(), inplace=True)
 
#In which variable how many
airbnb.isnull().sum()

#Some features have missing data:
sns.heatmap(airbnb.isnull(), yticklabels = False, cbar = False)
#White color indicates missing values
