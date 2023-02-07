

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.io as pio
import plotly.express as px
import io

title = " Analysis of Pakistan Covid-19 Data & Model Training"

st.markdown("<h1 style='font-size:30px;'>{}</h1>".format(title), unsafe_allow_html=True)

# # setting layout in streamlit
# st.set_page_config(layout="wide")
head1="Loading Data And Reading Data Set"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head1), unsafe_allow_html=True)


sheet_id="1vD2amsm9K39e7Hs0c6EbmHsVxP4gh4It"

df=pd.read_csv(f"https://drive.google.com/uc?export=download&id={sheet_id}")
st.write(df.head())
head2="Changing Column Names to Lower Case"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head2), unsafe_allow_html=True)
df = df.rename(columns={'Date':'date','Cases':'cases','Deaths':'deaths','Recovered':'recovered','Travel_history':'travel_history','Province':'province','City':'city'})
st.write(df.head())

head3="Checking Data Dimensions & Checking Data Types"

st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head3), unsafe_allow_html=True)
st.write(df.shape)

st.write(df.dtypes)

# checking datainfo
head4="Checking Data Info"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head4), unsafe_allow_html=True)


#Checking info



buffer = io.StringIO()
df.info(buf=buffer)
s = buffer.getvalue()

st.text(s)

# # checking for null values
head4="Checking for Missing Values"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head4), unsafe_allow_html=True)
st.write(df.isnull().sum())

head5="Replacing NULL Values"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head5), unsafe_allow_html=True)
st.write("As we can see, travel_history has NULL values, we will it with mode value and again will check again for NULL values")

df['travel_history'] = df['travel_history'].replace(np.nan, df['travel_history'].mode()[0])
st.write(df.isnull().sum())


head6="Checking Unique Values for Province"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head6), unsafe_allow_html=True)
st.write(df["province"].unique())

head7="Shortening Provinces Names and Correcting Misspelled Names"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head7), unsafe_allow_html=True)
st.write("As we can see, there are some misspelled names and some names are too long, we will correct them")

df["province"] = df["province"].replace({"Khyber Pakhtunkhwa": "KPK", "Gilgit-Baltistan": "GB",
     "Islamabad Capital Territory":"ISB","Federal Administration Tribal Area":"FATA","Azad Jummu Kashmir":"AJK", 
     "islamabad Capital Territory":"ISB","khyber Pakhtunkhwa":"KPK","Baluchistan":"Baluch"})
st.write(df["province"].unique())

head8="Checking Statiscal Parameters of Data"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head8), unsafe_allow_html=True)
st.write(df.describe())


# sorting data w.r.t Date
df = df.sort_values('date')

# checking data

head9="Checking & Exploring  Data with the Help of Graphs"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head9), unsafe_allow_html=True)
fig = px.scatter_matrix(df, dimensions=["cases", "deaths", "recovered"], color="province")
st.write(fig)



import plotly.graph_objects as go
fig = make_subplots(rows=1, cols=4,subplot_titles=("Cases","Deaths","Recovered","Travel History"))
fig.add_trace(go.Bar(x=df["province"], y=df["cases"],name="Cases"),row=1, col=1)
fig.add_trace(go.Bar(x=df["province"], y=df["deaths"],name="Deaths"),row=1, col=2)
fig.add_trace(go.Bar(x=df["province"], y=df["recovered"],name="Recovered"),row=1, col=3)
fig.add_trace(go.Bar(x=df["province"], y=df["travel_history"],name="Travel History"),row=1, col=4)
fig.update_layout(height=350, width=700, title_text="Covid-19 Cases in Province Based Distribution")
st.write(fig)




cases_index_cum = df.groupby('date')['cases'].sum().index
cases_values_cum = df.groupby('date')['cases'].sum().cumsum().values
recovered_index_cum = df.groupby('date')['recovered'].sum().index
recovered_values_cum= df.groupby('date')['recovered'].sum().cumsum().values
deaths_index_cum= df.groupby('date')['deaths'].sum().index
deaths_values_cum= df.groupby('date')['deaths'].sum().cumsum().values
fig = go.Figure()
fig.add_trace(go.Scatter(x=cases_index_cum, y=cases_values_cum, name="Cases",line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=recovered_index_cum, y=recovered_values_cum, name="Recovered",line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=deaths_index_cum, y=deaths_values_cum, name="Deaths",line=dict(color='red', width=2)))
fig.update_layout(title='Cummulative Cases, Recovered, Deaths Per Day in Pakistan',xaxis_title='Date',yaxis_title='Total Cases')
st.write(fig)


cases_index_province = df.groupby('province')['cases'].sum().index
cases_values_province = df.groupby('province')['cases'].sum().values
recovered_index_province = df.groupby('province')['recovered'].sum().index
recovered_values_province= df.groupby('province')['recovered'].sum().values
deaths_index_province= df.groupby('province')['deaths'].sum().index
deaths_values_province= df.groupby('province')['deaths'].sum().values
fig = go.Figure()
fig.add_trace(go.Scatter(x=cases_index_province, y=cases_values_province, name="Cases",line=dict(color='black', width=2)))
fig.add_trace(go.Scatter(x=recovered_index_province, y=recovered_values_province, name="Recovered",line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=deaths_index_province, y=deaths_values_province, name="Deaths",line=dict(color='red', width=2)))
fig.update_layout(title='Cummulative Cases, Recovered, Deaths in Province Based',xaxis_title='Date',yaxis_title='Total Cases')
st.write(fig)

head10="Cases vs Recovered"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head10), unsafe_allow_html=True)

import plotly.express as px
fig = px.scatter(df, x="cases", y="recovered", size="cases", color="province",
           hover_name="city", log_x=True, size_max=60)
st.write(fig)

head11="Cases vs Deaths"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head11), unsafe_allow_html=True)
import plotly.express as px
fig = px.scatter(df, x="cases", y="deaths", size="deaths", color="province",
           hover_name="city", log_x=True, size_max=60)
st.write(fig)


head12="Recovered vs Deaths"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head12), unsafe_allow_html=True)
import plotly.express as px
fig = px.scatter(df, x="recovered", y="deaths", size="deaths", color="province",
           hover_name="city", log_x=True, size_max=60)
st.write(fig)

head16="Impact of International Travel History"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head16), unsafe_allow_html=True)

import plotly.express as px
fig = px.histogram(df, x="cases", y="deaths", color="International_history", marginal="rug", hover_data=df.columns)
st.write(fig)

head17="Total Cases %age based on Prvinces"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head17), unsafe_allow_html=True)

fig = px.pie(df, values='cases', names='province')
st.write(fig)

head18="Total Recovered %age based on Prvinces"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head18), unsafe_allow_html=True)

fig = px.pie(df, values='recovered', names='province')
st.write(fig)

head19="Total Deaths %age based on Prvinces"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head19), unsafe_allow_html=True)

fig = px.pie(df, values='deaths', names='province')
st.write(fig)



head13="Total New Cases Per Day"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head13), unsafe_allow_html=True)

df['new_cases_rate'] = df['cases'].pct_change()
df['new_cases_rate'] = df['new_cases_rate'].fillna(0)
df['new_cases_rate'] = df['new_cases_rate'].replace(np.inf, 0)
df['new_cases_rate'] = df['new_cases_rate'].replace(-np.inf, 0)
df['new_cases_rate'] = df['new_cases_rate'].round(2)

plt.figure(figsize=(20,10))
plt.xticks(fontsize = 15,rotation=90)
plt.yticks(fontsize = 15)
# plt.xlabel("Date",fontsize = 15)
# plt.ylabel('Total cases',fontsize = 15)
plt.title('Total New Cases Rate Per Day',fontsize = 20)
x = df['date']
y = df['new_cases_rate']
plt.bar(x,y, color = 'Salmon', width = 0.5)
plt.show()
st.pyplot()

head14="Mortality Rate"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head14), unsafe_allow_html=True)

df['cum_cases'] = df['cases'].cumsum()
df['cum_deaths'] = df['deaths'].cumsum()
df['cum_recovered'] = df['recovered'].cumsum()
df['active_cases'] = df['cum_cases'] - df['cum_recovered'] - df['cum_deaths']
corona_data = df.groupby(['date'])['cases', 'recovered','active_cases','deaths','cum_cases'].sum().reset_index().sort_values('date',ascending=True)
corona_data['Mortality Rate'] = ((corona_data['deaths']/corona_data['cum_cases'])*100)
corona_data['Recovery Rate'] = ((corona_data['recovered']/corona_data['cum_cases'])*100) 
corona_data['Active Cases Rate'] = ((corona_data['active_cases']/corona_data['cum_cases'])*100)
corona_data['Mortality Rate'] = corona_data['Mortality Rate'].fillna(0)
corona_data['Recovery Rate'] = corona_data['Recovery Rate'].fillna(0)
corona_data['Active Cases Rate'] = corona_data['Active Cases Rate'].fillna(0)
corona_data['Mortality Rate'] = corona_data['Mortality Rate'].replace(np.inf, 0)
corona_data['Recovery Rate'] = corona_data['Recovery Rate'].replace(np.inf, 0)
corona_data['Active Cases Rate'] = corona_data['Active Cases Rate'].replace(np.inf, 0)
corona_data['Mortality Rate'] = corona_data['Mortality Rate'].round(2)
corona_data['Recovery Rate'] = corona_data['Recovery Rate'].round(2)
corona_data['Active Cases Rate'] = corona_data['Active Cases Rate'].round(2)

plt.figure(figsize=(20,10))
plt.xticks(fontsize = 15,rotation=90)
plt.yticks(fontsize = 15)
plt.title("Mortality Rate",fontsize=20)

plt.plot(corona_data['date'], corona_data['Mortality Rate'], marker = 'o',lw=2,color='Crimson' )

plt.show()
st.pyplot()



head15="Recovery Rate"
st.markdown("<h1 style='font-size:20px;'>{}</h1>".format(head15), unsafe_allow_html=True)

plt.figure(figsize=(20,10))
plt.xticks(fontsize = 15,rotation=90)
plt.yticks(fontsize = 15)
plt.title("Recovery Rate",fontsize=20)

plt.plot(corona_data['date'], corona_data['Recovery Rate'], marker = 'o', ls='dashdot',lw=2,color='Indigo' )

plt.show()
st.pyplot()


title1 = " Training Machine Model"

st.markdown("<h1 style='font-size:30px;'>{}</h1>".format(title1), unsafe_allow_html=True)


st.write("Now Machine Learning Model Training will be performed on this Data,Steps are share below")
st.write("1-Libraries are imported")
st.write("2-Data Imported")
st.write("3-Data cleaning and EDA is performed")
st.write("4-Data Train Test Split is performed in ratio 80%/20%")
st.write("5-Model is trained on data ")
st.write("6-Model is tested on test data")
st.write("7-Model is evaluated on test data")
st.write("8-Model is saved in joblib")
st.write("9-Model is deployed on Streamlit")



st.write("Different models like (Logistic Regression,SVM,Decision Tree, Random Forest,KNN) will be trained against this data, during process, one model will be selected and then its results will be evaluated by F1, Recall and Precision ")














# import streamlit as st

# algorithm_options = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'KNN']
# selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
le.fit(df["travel_history"])
df["travel_history"]=le.transform(df["travel_history"])
le.fit(df["date"])
df["date"]=le.transform(df["date"])
le.fit(df["city"])
df["city"]=le.transform(df["city"])
le.fit(df["International_history"])
df["International_history"]=le.transform(df["International_history"])


X = df[['cases', 'deaths', 'recovered', 'travel_history',"city"]]
y = df['International_history']


algorithm_options = ['Logistic Regression', 'SVM', 'Decision Tree', 'Random Forest', 'KNN']
selected_algorithm = st.sidebar.selectbox("Select Algorithm", algorithm_options)

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

if selected_algorithm == 'Logistic Regression':
    st.write("Result for Logistic Regression")
elif selected_algorithm == 'SVM':
    st.write("Result for SVM")
elif selected_algorithm == 'Decision Tree':
    st.write("Result for  Decision Tree")
elif selected_algorithm == 'Random Forest':
    st.write("Result for Random Forest")
elif selected_algorithm == 'KNN':
    st.write("Result for  KNN")

if selected_algorithm == 'Logistic Regression':
    model = LogisticRegression()
elif selected_algorithm == 'SVM':
    model = SVC()
elif selected_algorithm == 'Decision Tree':
    model = DecisionTreeClassifier()
elif selected_algorithm == 'Random Forest':
    model = RandomForestClassifier()
elif selected_algorithm == 'KNN':
    model = KNeighborsClassifier()

# fit the model on the training data
model.fit(X_train, y_train)

y_pred=model.predict(X_test)

# evaluate the model on the test data



from sklearn.metrics import precision_score,recall_score,f1_score
st.write("Precision: ",round(precision_score(y_test,y_pred),3))
st.write("Recall: ",round(recall_score(y_test,y_pred),3))
st.write("F1: ",round(f1_score(y_test,y_pred),3))





