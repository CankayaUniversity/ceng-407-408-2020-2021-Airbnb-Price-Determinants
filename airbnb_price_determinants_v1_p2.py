#Handling Categorical Features

#Selecting categorical data
a=airbnb.copy()
categorical_a = a.select_dtypes(include = ["object"])
list(categorical_a)

#Total class numbers of host_response_time categorical data
print(categorical_a["host_response_time"].value_counts().count())
categorical_a["host_response_time"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
host_response_time_new=pd.get_dummies(airbnb['host_response_time'])
host_response_time_new.head()

#Total class numbers of host_is_superhost categorical data
print(categorical_a["host_is_superhost"].value_counts().count())
categorical_a["host_is_superhost"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
host_is_superhost_new=pd.get_dummies(airbnb['host_is_superhost'])
host_is_superhost_new.head()

#Total class numbers of host_has_profile_pic categorical data
print(categorical_a["host_has_profile_pic"].value_counts().count())
categorical_a["host_has_profile_pic"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
host_has_profile_pic_new=pd.get_dummies(airbnb['host_has_profile_pic'])
host_has_profile_pic_new.head()

#Total class numbers of host_identity_verified categorical data
print(categorical_a["host_identity_verified"].value_counts().count())
categorical_a["host_identity_verified"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
host_identity_verified_new=pd.get_dummies(airbnb['host_identity_verified'])
host_identity_verified_new.head()

#Total class numbers of neighbourhood_cleansed categorical data
print(categorical_a["neighbourhood_cleansed"].value_counts().count())
categorical_a["neighbourhood_cleansed"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
neighbourhood=pd.get_dummies(airbnb['neighbourhood_cleansed'])
neighbourhood.head()

#Total class numbers of property_type categorical data
print(categorical_a["property_type"].value_counts().count())
categorical_a["property_type"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
house_type=pd.get_dummies(airbnb['property_type'])
house_type.head()

#Total class numbers of room_type categorical data
print(categorical_a["room_type"].value_counts().count())
categorical_a["room_type"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
room_type_new=pd.get_dummies(airbnb['room_type'])
room_type_new.head()

#Total class numbers of has_availability categorical data
print(categorical_a["has_availability"].value_counts().count())
categorical_a["has_availability"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
has_availability_new=pd.get_dummies(airbnb['has_availability'])
has_availability_new.head()

#Total class numbers of instant_bookable categorical data
print(categorical_a["instant_bookable"].value_counts().count())
categorical_a["instant_bookable"].value_counts()

#Converting categorical variables into "dummy" or indicator variables 
instant_bookable_new=pd.get_dummies(airbnb['instant_bookable'])
instant_bookable_new.head()

# Add new dummy columns to data frame
airbnb= pd.concat([airbnb,host_response_time_new,host_is_superhost_new,host_has_profile_pic_new,host_identity_verified_new,neighbourhood,house_type,room_type_new,has_availability_new,instant_bookable_new],axis = 1)
airbnb.drop(['host_response_time','host_is_superhost','host_has_profile_pic','host_identity_verified','neighbourhood_cleansed','property_type','room_type','has_availability','instant_bookable'],axis = 1, inplace = True)

airbnb.shape
airbnb.columns

#Selecting categorical data
a=airbnb.copy()
categorical_a = a.select_dtypes(include = ["object"])
list(categorical_a)

#Are there any missing observations (values)
airbnb.isnull().values.any()

#Model Training
#Random Forests

#Import library
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

#df=airbnb
df = airbnb.copy()

#TEST 0
#rf_model mean_squared_error=521.4673
#rf_tuned mean_squared_error=601.7007
#feature 190

#TEST 1
#rf_model mean_squared_error=521.2745896444618
#rf_tuned mean_squared_error=596.3227774806285
#feature 190->180
df.drop(['Island','Private room in cabin','Private room in camper/rv','Private room in castle','Private room in hut',
'Private room in pension','Shared room in pension','Tent','Treehouse','Yurt'],axis=1,inplace=True)

#TEST 2
#rf_model mean_squared_error=522.2285005472032
#rf_tuned mean_squared_error=588.1095404520753
#feature 180->130
df.drop(['Bayrampasa','Esenler','Gungoren','Sultanbeyli','Campsite','Entire chalet','Entire guest suite','Entire guesthouse',
'Entire home/apt','Entire hostel','Entire place','Hut','Lighthouse','Private room','Private room in bungalow',
'Private room in casa particular','Private room in chalet','Private room in earth house',
'Private room in farm stay','Private room in guest suite','Private room in houseboat','Private room in nature lodge',
'Private room in pousada','Private room in tent','Room in casa particular','Room in nature lodge','Room in pension',
'Shared room','Shared room in aparthotel','Shared room in bed and breakfast','Shared room in boutique hotel',
'Shared room in cabin','Shared room in cave','Shared room in condominium','Shared room in earth house',
'Shared room in guest suite','Shared room in guesthouse','Shared room in hotel','Shared room in loft',
'Shared room in nature lodge','Shared room in serviced apartment','Shared room in tiny house',
'Shared room in townhouse','Shared room in villa','Shared room in yurt','Tiny house','Windmill'],axis=1,inplace=True)

#TEST 3
#rf_model mean_squared_error=528.5975955581303
#rf_tuned mean_squared_error=563.6328941374361
#feature 130->81
df.drop(['f','t','Atasehir','Avcilar','Bagcilar','Bahcelievler','Bakirkoy','Besiktas','Beylikduzu','Catalca','Cekmekoy',
'Eyup','Gaziosmanpasa','Kartal','Pendik','Sancaktepe','Sultangazi','Umraniye','Uskudar',
'Boat','Camper/RV','Casa particular','Castle','Dome house','Entire bed and breakfast',
'Entire bungalow','Entire condominium','Entire loft','Private room in guesthouse','Private room in hostel',
'Private room in loft','Private room in tiny house','Private room in treehouse','Private room in villa',
'Private room in yurt','Room in aparthotel','Room in hostel','Room in serviced apartment','Shared room in casa particular',
'Shared room in hostel','Shared room in house'],axis=1,inplace=True)

#TEST 4
#rf_model mean_squared_error=527.6839180955582
#rf_tuned mean_squared_error=562.7216809338322
#feature 81->72
df.drop(['Basaksehir','Kucukcekmece','Maltepe','Tuzla','Entire cottage','Entire townhouse','Private room in condominium',
'Private room in serviced apartment','Private room in townhouse'],axis=1,inplace=True)

#TEST 5
#rf_model mean_squared_error=531.7460241782671
#rf_tuned mean_squared_error=557.8012255014384
#feature 72->62
##0.1 ... drop
df.drop(['within a day','Adalar','Arnavutkoy','Kagithane','Sile','Sisli','Entire serviced apartment',
'Private room in house','Room in bed and breakfast','Shared room in apartment'],axis=1,inplace=True)

#TEST 6
#rf_model mean_squared_error=528.9530682771826
#rf_tuned mean_squared_error=556.3725963025881
#feature 62->59
##0.1 ... drop
df.drop(['Entire house','Private room in bed and breakfast','Room in hotel'],axis=1,inplace=True)

#TEST 7
#rf_model mean_squared_error=530.5877424356488
#rf_tuned mean_squared_error=553.9720900930081
#feature 59->55
##0.2 ... drop
df.drop(['Esenyurt','number_of_reviews_l30d','reviews_per_month','Kadikoy'],axis=1,inplace=True)

#TEST 8
#rf_model mean_squared_error=536.4966840800744
#rf_tuned mean_squared_error=549.8637767997963
#feature 55->47
##0.3 ... drop
df.drop(['within a few hours','Zeytinburnu','review_scores_cleanliness','calculated_host_listings_count_shared_rooms',
'Beyoglu','Buyukcekmece','Fatih','Farm stay'],axis=1,inplace=True)

#TEST 9
#rf_model mean_squared_error=538.613943765785
#rf_tuned mean_squared_error=547.851240768342
#feature 47->44
##0.4 ... drop
df.drop(['Silivri','Room in boutique hotel','Hotel room'],axis=1,inplace=True)

#TEST 10
#rf_model mean_squared_error=536.8384281302258
#rf_tuned mean_squared_error=546.5737855490202
#feature 44->41
##0.4 ... drop
df.drop(['latitude','review_scores_value','Shared room in chalet'],axis=1,inplace=True)

#Checking airbnb shape
df.shape

#Data Train-Test Split
y = df["price"]
X= df.drop(['price'], axis=1).astype('int64')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,random_state=42)

#Prediction
rf_model = RandomForestRegressor(random_state = 42)
rf_model.fit(X_train, y_train)
rf_model.predict(X_test)[0:5]
y_pred = rf_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))
#Model Tuning
rf_tuned = RandomForestRegressor(max_depth  = 8,max_features = 3, n_estimators =200)
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
#MSE error checking
np.sqrt(mean_squared_error(y_test, y_pred))
#Feature importance checking
Importance = pd.DataFrame({"Importance": rf_tuned.feature_importances_*100},
                         index = X_train.columns)
pd.set_option('display.max_rows', None)
print(Importance)

from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
