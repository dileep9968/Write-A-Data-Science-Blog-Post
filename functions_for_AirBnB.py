# The collection of functions for the Boston AirBnB dataset

# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar #To check holidays in the U.S
import time
import copy

def load_bnb_files():
    '''Load AirBnB files'''
    df_listing = pd.read_csv('./data/listings.csv')
    df_calendar = pd.read_csv('./data/calendar.csv')

    return df_listing, df_calendar



# Modify df_calendar for future work
# Special event : marathon, new academic season

def modify_calendar(df_calendar):
    '''
    This function creates 'year', 'month', 'day', 'weekday',  and 'week_number' columns from 'date' coulmn of df_calendar 
    and remove '$' string from 'price' coulmn.
    
    Input : a Pandas dataframe having a date data column
    Output : a Pandas dataframe having year, month, day, weekday, us_holiday columns
    '''
    # Split date column into year, month,day, weekday columns
    # The day of the week with Monday=0, Sunday=6
    # Set the range of weekends from Friday to Sunday
    df_calendar['year'] = pd.DatetimeIndex(df_calendar['date']).year
    df_calendar['month'] = pd.DatetimeIndex(df_calendar['date']).month
    df_calendar['day'] = pd.DatetimeIndex(df_calendar['date']).day
    df_calendar['weekday'] = pd.DatetimeIndex(df_calendar['date']).weekday
    df_calendar['week_number'] = pd.DatetimeIndex(df_calendar['date']).week
    df_calendar['price']= df_calendar['price'].str.replace('$','')
    df_calendar['price']=df_calendar['price'].str.replace(',','')
    df_calendar['price'] = df_calendar['price'].astype(float)
    
    # Add us_holiday column
    cal = calendar()
    holidays = cal.holidays(start=df_calendar.date.min(), end=df_calendar.date.max())
    df_calendar['us_holiday'] = df_calendar.date.astype('datetime64').isin(holidays)
    
    # Add weekend column #Friday, Saturday
    weekend = [4,5]
    df_calendar['weekend'] = df_calendar.weekday.isin(weekend)
    
    # Replace values in weekday column 
    df_calendar['weekday'].replace({0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday',4:'Friday', 5:'Saturday', 6:'Sunday'}, inplace=True)
    
    return df_calendar



def add_availabledays_price(df_listing, df_cal_modified):
    '''
    This function creates the columns of 'unavail_days', 'avail_days_weekends', 
    'avail_days_weekdays', 'price_weekend', and 'price_weekday' where calculated from df_cal_modified on df_listing.
    
    Input : 
    - A Pandas dataframe made from 'listings.csv' : df_listing
    - A pandas dataframe modified by modify_calendar() : df_cal_modified
    
    Output :
    - The modified df_listing dataframe with new 'unavail_days', 'avail_days_weekends',
    'avail_days_weekdays', 'price_weekend', and 'price_weekday' columns 
    '''
    id_list = df_listing.id[:]
    unavailable_days_array = np.array([])
    avail_days_weekends_array =  np.array([])
    avail_days_weekdays_array = np.array([])
    price_weekend_array = np.array([])
    price_weekday_array = np.array([])

    for i in np.nditer(id_list):
        tmp = df_cal_modified[(df_cal_modified.listing_id == i)] # Make a dataframe coming from df_listing with a certain id
        available_dict = tmp.available.value_counts().to_dict()
        if 'f' in available_dict:
            unavailable_days = tmp[tmp.available == 'f'].shape[0]
        else:
            unavailable_days = 0

        if 't' in available_dict:
            available_weekends = tmp[(tmp.available == 't') & (tmp.weekend == True)].shape[0]
            available_weekdays = tmp[(tmp.available == 't') & (tmp.weekend == False)].shape[0]
            price_weekend = tmp[(tmp.weekend == True) & (tmp.available == 't')].price.astype(float).describe()['mean']
            price_weekday = tmp[(tmp.weekend == False) & (tmp.available == 't')].price.astype(float).describe()['mean']

        else:
            available_weekends = 0
            available_weekdays = 0
            price_weekend = np.nan
            price_weekday = np.nan


        unavailable_days_array = np.append(unavailable_days_array, unavailable_days)
        avail_days_weekends_array = np.append(avail_days_weekends_array, available_weekends)
        avail_days_weekdays_array = np.append(avail_days_weekdays_array, available_weekdays)
        price_weekend_array = np.append(price_weekend_array, price_weekend)
        price_weekday_array = np.append(price_weekday_array, price_weekday)

    df_listing['unavail_days'] = pd.Series(unavailable_days_array)
    df_listing['avail_days_weekends'] = pd.Series(avail_days_weekends_array)
    df_listing['avail_days_weekdays'] = pd.Series(avail_days_weekdays_array)
    df_listing['price_weekend'] = pd.Series(price_weekend_array)
    df_listing['price_weekday'] = pd.Series(price_weekday_array)
    
    return df_listing



def clean_listing_df(df_listing):
    '''
    This function aims to make the df_listing dataframe for data analysis by
    - removing irrelevant columns
    - changing object type columns to numeric columns or manipulating them using one hot encoding
    - filling NaN values
    - creating an integrated_score_log column by the natural log of the result from 'review_scores_rating' times 'number_of_reviews' +1 

    Input : 
    - A Pandas dataframe made from 'listings.csv' : df_listing

    Output :
    - Cleaned df_listing
    '''
    # Drop columns having 50% of nan value. There were reasons that I decided 50% the threshold for dropping columns.
    # 1. Easy to see the dataframe and to check the meaning of the columns.
    # 2. Decide which ones have to be dropped.
    # The candidates columns to be dropped are 'notes', 'neighbourhood_group_cleansed', 'square_feet', 'weekly_price', 'monthly_price', 'security_deposit', 'has_availability', 'license', 'jurisdiction_names'. Most of them are duplicated to other columns or irrelavant except  'security_deposit' column. I didn't do imputing by the mean or mode of the column because it can distort real shape. I didn't do one-hot-encoding to make the dataframe straightforward. 'security_deposit' has 55 unique values.
    df_missing = df_listing.isna().mean()
    df_listing_modi1 = df_listing.drop(df_missing[df_missing>0.5].index.to_list(), axis=1)
    # Drop columns related with urls and other irrelevant columns. 
    # url and othe columns are all unique or useless.
    remove_list1 = ['listing_url', 'scrape_id', 'last_scraped', 'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url', 'host_url', 
                'host_thumbnail_url', 'host_picture_url', 'country_code', 'country']
    df_listing_modi1.drop(remove_list1, axis=1, inplace=True)
    # Drop the columns because of data overlap [city, smart_location], Only one value [state], 
    # Drop the wrong data [market, calendar_last_scraped]
    remove_list2 = ['smart_location', 'state', 'name', 'summary', 'space', 'description','neighborhood_overview',
                'transit','access','market','calendar_last_scraped']
    df_listing_modi1.drop(remove_list2, axis=1, inplace=True)
    
    # Modify 'house_rules' column to 'house_rules_exist_tf' having True value if there is a rule.
    # False value, if there is no rule.
    # Houes_rules are different for every host. So it is not practical to use one-hot-encoding. Instead of that,
    #  It is changed to binary type, which is there is rule in a house, True, otherwise, False.
    # This can save some information, which is better than just dropping.

    df_listing_modi1['house_rules_exist_tf']= pd.notna(df_listing_modi1.house_rules)
    df_listing_modi1.drop(['house_rules'], axis=1, inplace=True)
    # Remove columns having 1000 unique string valuses and irrelevant data
    remove_list3 = ['interaction', 'host_name', 'host_since', 'host_about', 'street','first_review','experiences_offered','requires_license',
                    'last_review','host_location','neighbourhood_cleansed','experiences_offered','requires_license']
    df_listing_modi2 = df_listing_modi1.drop(remove_list3, axis=1)

    # Change the columns 'host_response_rate', 'host_acceptance_rate' to float type
    columns_change_type = ['host_response_rate','host_acceptance_rate', 'price', 'cleaning_fee']
    for i in columns_change_type:
        df_listing_modi2[i] = df_listing_modi2[i].str.replace('%','')
        df_listing_modi2[i] = df_listing_modi2[i].str.replace('$','')
        df_listing_modi2[i] = df_listing_modi2[i].str.replace(',','')
        df_listing_modi2[i] = df_listing_modi2[i].astype(float)
    
    # Modify and Split values in 'amenities' column
    # Amenities can be one of reason that potential candidate might consider.
    df_listing_modi2.amenities = df_listing_modi2.amenities.str.replace("[{}]", "")
    df_amenities = df_listing_modi2.amenities.str.get_dummies(sep = ",")
    df_amenities = df_amenities.add_prefix('amenities_')
    df_listing_modi2 = pd.concat([df_listing_modi2, df_amenities], axis=1)
    df_listing_modi2 = df_listing_modi2.drop('amenities', axis=1)
    
    # Use get_dummies for columns having unique values less then 10
    # It is reasonable to use one-hot-encoding if the nunber of unique values are less then 10.
    # It doesn't lose information, and keep the dataframe simple.
    columns_of_object_less10 =[]
    for i,j in zip(df_listing_modi2.columns.to_list(), df_listing_modi2.dtypes.to_list()):
        if j == object and len(df_listing_modi2[i].value_counts()) < 10 :
            columns_of_object_less10.append(i)
    df_listing_modi2 = pd.get_dummies(df_listing_modi2, columns=columns_of_object_less10, prefix=columns_of_object_less10, 
                                      dummy_na=True)
    
    #  Modify 'extra_people' coulmn to get boolean type of 'extra_people_fee_tf'
    # Instead of dropping, I decided to change 'extra_people' coulmn to binary type to save some information
    df_listing_modi2['extra_people'] = df_listing_modi2['extra_people'].astype(str)
    df_listing_modi2['extra_people']= df_listing_modi2['extra_people'].str.replace('$','')
    df_listing_modi2['extra_people']=df_listing_modi2['extra_people'].str.replace(',','')
    df_listing_modi2['extra_people'] = df_listing_modi2['extra_people'].astype(float)
    df_listing_modi2['extra_people'] = df_listing_modi2['extra_people'].replace(to_replace=0, value=np.nan)
    df_listing_modi2['extra_people_fee_tf']= pd.notna(df_listing_modi2.extra_people)
    df_listing_modi2 = df_listing_modi2.drop('extra_people', axis=1)
    
    # Modify and Split values in 'host_verifications' column
    df_listing_modi2.host_verifications = df_listing_modi2.host_verifications.str.replace("[", "")
    df_listing_modi2.host_verifications = df_listing_modi2.host_verifications.str.replace("]", "")
    df_host_verifications = df_listing_modi2.host_verifications.str.get_dummies(sep = ",")
    df_host_verifications = df_host_verifications.add_prefix('host_verification_')
    df_listing_modi2 = pd.concat([df_listing_modi2, df_host_verifications], axis=1)
    df_listing_modi2 =  df_listing_modi2.drop(['host_verifications'], axis=1)
    df_listing_modi2 =  df_listing_modi2.drop(['host_neighbourhood'], axis=1)
    
    # Modify 'calendar_updated' column
    # Instead of dropping, I decided to change 'calendar_updated' coulmn to binary type (updated within a week or not)
    # to save some information
    df_listing_modi2["calendar_updated_1weekago"] = np.where(df_listing_modi2['calendar_updated'].str.contains(
        "days|yesterday|today|a week ago")==True, 'yes', 'more_than_1week')
    df_listing_modi2 =  df_listing_modi2.drop(['calendar_updated'], axis=1)
    
    # Use get_dummies for the columns 'neighbourhood', 'city', 'zipcode', 'property_type'
    tmp = df_listing_modi2.columns.to_list()
    tmp1 = df_listing_modi2.dtypes.to_list()
    columns_of_object_over10 =[]
    for i,j in zip(tmp,tmp1):
            if j == object and len(df_listing_modi2[i].value_counts()) > 10 :
                columns_of_object_over10.append(i)
                
    df_listing_modi2 = pd.get_dummies(df_listing_modi2, columns=columns_of_object_over10, 
                                      prefix=columns_of_object_over10, dummy_na=True)
    
    df_listing_modi2 = pd.get_dummies(df_listing_modi2, columns=['calendar_updated_1weekago','house_rules_exist_tf','extra_people_fee_tf'], 
                                  prefix=['calendar_updated_1weekago','house_rules_exist_tf','extra_people_fee_tf'], dummy_na=True)
    
    df_listing_modi2["host_response_rate_100"] = np.where(df_listing_modi2['host_response_rate'] ==100, True, False)
    df_listing_modi2["host_acceptance_rate_100"] = np.where(df_listing_modi2['host_acceptance_rate'] ==100, True, False)
    df_listing_modi2 =  df_listing_modi2.drop(['host_response_rate','host_acceptance_rate','reviews_per_month'], axis=1)
    
    # bathrooms, bedrooms, beds, cleaning_fee, review_scores_rating, review_... : : fillna with mean value
    # The empty cell are filled with mean values of corresponding columns. Because these are numerical type,
    # I thought imputing with mean values is better than dropping or one-hot-encoding
    columns1 = ['bathrooms','bedrooms','beds','cleaning_fee','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
            'review_scores_communication','review_scores_location','review_scores_value']
    df_listing_modi2[columns1] = df_listing_modi2[columns1].fillna(df_listing_modi2.mean())
    df_listing_modi2.price_weekend.fillna(df_listing_modi2.price, inplace=True)
    df_listing_modi2.price_weekday.fillna(df_listing_modi2.price, inplace=True)
    df_listing_modi2['integrated_score_log'] = np.log(df_listing_modi2['review_scores_rating']*df_listing_modi2['number_of_reviews']+1)
    
    df_listing_modi2 = pd.get_dummies(df_listing_modi2, columns=['host_response_rate_100','host_acceptance_rate_100'], 
                                  prefix=['host_response_rate_100','host_acceptance_rate_100'])
    df_listing_modi2 = df_listing_modi2.drop(['id', 'host_id', 'latitude', 'longitude','price','host_listings_count','host_total_listings_count','maximum_nights'], axis=1)
    
    
    
    return df_listing_modi2


def conditioning_listing_df(df_listing_modi2):
    '''
    This function is for conditioning a dataframe returned by the funtion 'clean_listing_df(df_listing)''
    
    Input : 
    - A Pandas dataframe came from the function 'clean_listing_df(df_listing)''

    Output :
    - Cleaned df_listing_modi2 : df_listing_modi3
    '''
    
    threshold_80 = df_listing_modi2.integrated_score_log.quantile(0.8)
    condition = [df_listing_modi2['integrated_score_log'] == 0, df_listing_modi2['integrated_score_log'] >= threshold_80]
    label_list = ['poor','high']
    df_listing_modi2['y_label'] = np.select(condition, label_list, default='normal')
    
    # Drop columns related to 'y_label' column
    # Without dropping, the remained columns affect model's prediction
    df_listing_modi3 =  df_listing_modi2.drop(['integrated_score_log','number_of_reviews','review_scores_rating', 'review_scores_value',
                                          'review_scores_communication','review_scores_accuracy','review_scores_checkin','review_scores_cleanliness',
                                          'review_scores_location', 'availability_30','availability_60', 'availability_90','availability_365','calculated_host_listings_count'], axis=1)
    
    return df_listing_modi3

def investigate(df_listing_scaled, pca, i):
    '''
    This function checks pca components that which original features are storngly related to a pca component

    Input : 
    - Dataframe : df_listing_scaled a dataframe scaled by StandardScaler()
    - pca instance
    - i : The number of pca component

    Output :
    - pos_list : Original features having positive relationship with a 
    corresponding pca component,which are sorted in order of importance 
    - neg_list : Original features having positive relationship with a 
    corresponding pca component,which are sorted in order of importance 
    '''
    pos_list =[]
    neg_list =[]
    feature_names = list(df_listing_scaled.columns)
    weights_pca = copy.deepcopy(pca.components_[i])
    combined = list(zip(feature_names, weights_pca))
    combined_sorted= sorted(combined, key=lambda tup: tup[1], reverse=True)
    tmp_list = [list(x) for x in combined_sorted]
    tmp_list = [(x[0],"{0:.3f}".format(x[1])) for x in tmp_list]
    print("positive to pca{}:".format(i), tmp_list[0:10])
    print()
    print("negative to pca{}:".format(i), tmp_list[-1:-11:-1])
    print()
    for j in range(0,10):
        pos_list.append(tmp_list[j][0])
    for k in range(1,11):
        neg_list.append(tmp_list[-k][0])
    
    return pos_list, neg_list
    

def check_difference(pos_list, neg_list, df_listing_poor, df_listing_high):
    '''
    Print original features that are stongly related with a corresponding pca component.
    '''
    data_pos = [[df_listing_high[x].mean(), df_listing_poor[x].mean()] for x in pos_list]
    data_neg = [[df_listing_high[x].mean(), df_listing_poor[x].mean()] for x in neg_list]
    tmp_pos = pd.DataFrame(data=data_pos , index=pos_list, columns=['high', 'poor'])
    tmp_neg = pd.DataFrame(data=data_neg , index=neg_list, columns=['high', 'poor'])
    tmp_both = pd.concat([tmp_pos, tmp_neg])
    tmp_both["difference"] = tmp_both.high - tmp_both.poor
    tmp_both["difference"] = tmp_both["difference"].abs()
    result = tmp_both.sort_values(by=['difference'], ascending=False)
    return result
