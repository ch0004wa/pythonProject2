import pandas as pd
from datetime import datetime, timedelta
from datetime import date
import numpy as np
import streamlit as st
import newsapi
from newsapi import NewsApiClient
import plotly.express as px
from textblob import TextBlob
import copy
import markdown
# modules for generating the word cloud
from os import path, getcwd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator, STOPWORDS
import cv2
import random
import wget
import os

api_keys = ['3002a7015bd143c48157daa21b02e0ce', 'a6ce43de65624b38b77583a8fb169a2f', 'f0ca9830a4164802b7a97a0354578333',
            '4c10ff42cd0e42e79b8d00929b54149e', '96366c24528b4f32981623578f932899', '9312360e92e94909af3113c06bcbcdfc',
            '4e44f023ead44a299e0484aa36d5e8c3']
st.markdown(
    f"""
<style>
    .reportview-container .main .block-container{{
        max-width: 1000px;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.image('./docs/lidaslogo.png', width=250)
md = markdown.Markdown()

st.markdown(
    '''
    <link rel="preconnect" href="https://fonts.gstatic.com">
    <link href="https://fonts.googleapis.com/css2?family=PT+Sans:wght@700&display=swap" rel="stylesheet">
        <div style="font-family: 'PT Sans';font-size:48px"><center>Latest COVID-19 News Update</center></div>
    ''', unsafe_allow_html=True
)


@st.cache(max_entries=1)
def get_data(date):
    os.system("rm cases.csv")
    url = "https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
    filename = wget.download(url, "cases.csv")
    casedata = pd.read_csv(filename, encoding='latin-1')
    casedata['dateRep'] = pd.to_datetime(casedata['dateRep'], format="%d/%m/%Y")
    mindt = min(casedata['dateRep'])
    casedata['days_since_start'] = casedata['dateRep'].apply(lambda x: (x - mindt).days)
    # casedata['first_case_date']
    country_first_case_date = casedata[casedata['cases'] != 0].groupby("countryterritoryCode").aggregate(
        {'dateRep': 'min'}).reset_index()
    country_first_case_date.columns = ['countryterritoryCode', 'firstcasedate']
    casedata = casedata.merge(country_first_case_date, on='countryterritoryCode', how='left')
    casedata['days_since_country_first_case'] = casedata.apply(lambda x: (x['dateRep'] - x['firstcasedate']).days,
                                                               axis=1)
    casedata = casedata[casedata['days_since_country_first_case'] >= 0]
    casedata['geoId'] = casedata['geoId'].apply(lambda x: str(x).lower())
    casedata = casedata.sort_values(by='dateRep')
    casedata['cumdeaths'] = casedata.groupby(by=['countryterritoryCode'])['deaths'].apply(lambda x: x.cumsum())
    casedata['cumcases'] = casedata.groupby(by=['countryterritoryCode'])['cases'].apply(lambda x: x.cumsum())
    return casedata


curdate = date.today()
casedata = get_data(curdate)

st.markdown(f"<center> <h1> Global COVID-19 Update </h1></center>", unsafe_allow_html=True)

type_data = st.selectbox("Number of COVID-19 cases or deaths by country (You can click on the legend labels to select countries to compare)",
                         ['Cases', 'Deaths'], index=0)
if type_data == 'Deaths':
    st.plotly_chart(
        px.line(casedata, x='days_since_country_first_case', y='cumdeaths', line_group='countriesAndTerritories',
                color='countriesAndTerritories'), use_container_width=True)
elif type_data == 'Cases':
    st.plotly_chart(
        px.line(casedata, x='days_since_country_first_case', y='cumcases', line_group='countriesAndTerritories',
                color='countriesAndTerritories'), use_container_width=True)

st.sidebar.markdown(f'''<div class="card text-white bg-info mb-3" style="width: 18rem">
  <div class="card-body">
    <h5 class="card-title">Total COVID-19 Cases</h5>
    <p class="card-text">{sum(casedata['cases']):,d}</p>
  </div>
</div>''', unsafe_allow_html=True)

st.sidebar.markdown(f'''<div class="card text-white bg-danger mb-3" style="width: 18rem">
  <div class="card-body">
    <h5 class="card-title">Total Deaths caused by COVID-19</h5>
    <p class="card-text">{sum(casedata['deaths']):,d}</p>
  </div>
</div>''', unsafe_allow_html=True)

st.sidebar.markdown(f'''<div class="card text-white bg-warning mb-3" style="width: 18rem">
  <div class="card-body">
    <h5 class="card-title">Average Death Rate</h5>
    <p class="card-text">{sum(casedata['deaths']) / sum(casedata['cases']) * 100:,.2f}%</p>
  </div>
</div>''', unsafe_allow_html=True)

country_death_cases = casedata.groupby(['countriesAndTerritories'])[['cases', 'deaths']].aggregate(np.sum).reset_index()
country_death_cases['fatalityRate'] = country_death_cases['deaths'] / country_death_cases['cases'] * 100
country_death_cases = country_death_cases[country_death_cases['cases'] > 100].sort_values(by='fatalityRate',
                                                                                          ascending=False)
st.sidebar.markdown("### Fatality Rates in countries with minimum 100 cases")
# st.sidebar.table(country_death_cases[:10][['countriesAndTerritories','fatalityRate']])
st.sidebar.plotly_chart(
    px.bar(country_death_cases[:10].sort_values(by='fatalityRate'), y='countriesAndTerritories', x='fatalityRate',
           orientation='h'), use_container_width=True)

# # Top News results every 6 minutes
st.cache(ttl=360, max_entries=20)


def create_dataframe_top(queries, country):
    newsapi = NewsApiClient(api_key=random.choice(api_keys))
    fulldata = pd.DataFrame()
    for q in queries:
        json_data = newsapi.get_top_headlines(q=q,
                                              language='en',
                                              country=country)
        data = pd.DataFrame(json_data['articles'])
        if len(data) > 0:
            data['source'] = data['source'].apply(lambda x: x['name'])
            data['publishedAt'] = pd.to_datetime(data['publishedAt'])
            fulldata = pd.concat([fulldata, data])
    if len(fulldata) > 0:
        fulldata = fulldata.drop_duplicates(subset='url').sort_values(by='publishedAt', ascending=False).reset_index()
    return fulldata


def create_most_recent_markdown(df, width=700):
    if len(df) > 0:
        # img url
        img_path = df['urlToImage'].iloc[0]
        if not img_path:
            images = [x for x in df.urlToImage.values if x is not None]
            if len(images) != 0:
                img_path = random.choice(images)
            else:
                img_path = 'https://www.nfid.org/wp-content/uploads/2020/02/Coronavirus-400x267.png'
        # st.write(img_path)
        img_alt = df['title'].iloc[0]
        df = df[:5]
        markdown_str = f"<img src='{img_path}' width='{width}'/> <br> <br>"
        for index, row in df.iterrows():
            markdown_str += f"[{row['title']}]({row['url']}) by {row['author']}<br> "
        return markdown_str
    else:
        return ''


@st.cache()
def country_data_codes():
    iso_code = pd.read_csv("ISO 3166.csv")
    iso_code['Alpha-2 code'] = iso_code['Alpha-2 code'].apply(lambda x: str(x).lower())
    code_country_name_dict = iso_code.set_index('Alpha-2 code').to_dict()['Country name']
    country_name_code_dict = iso_code.set_index('Country name').to_dict()['Alpha-2 code']
    country_options = ['ae', 'ar', 'at', 'au', 'be', 'bg', 'br', 'ca', 'ch', 'cn', 'co', 'cu', 'cz', 'de', 'eg', 'fr',
                       'gb', 'gr', 'hk', 'hu', 'id', 'ie', 'il', 'in', 'it', 'jp', 'kr', 'lt', 'lv', 'ma', 'mx', 'my',
                       'ng', 'nl', 'no', 'nz', 'ph', 'pl', 'pt', 'ro', 'rs', 'ru', 'sa', 'se', 'sg', 'si', 'sk', 'th',
                       'tr', 'tw', 'ua', 'us', 've', 'za']
    country_options_st = [code_country_name_dict[x] for x in country_options]
    return country_name_code_dict, country_options_st


def textblob_sentiment(title, description):
    blob = TextBlob(str(title) + " " + str(description))
    return blob.sentiment.polarity


@st.cache()
def create_mask():
    mask = np.array(Image.open("coronavirus.png"))
    im_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(im_gray, thresh=20, maxval=255, type=cv2.THRESH_BINARY)
    mask = 255 - mask
    return mask




country_name_code_dict, country_options_st = country_data_codes()
country_name = st.selectbox("Select your Country", country_options_st, index=51)
country_code = country_name_code_dict[country_name]
# st.write(country_code)
data = create_dataframe_top(['covid', 'corona'], country=country_code)
top_results_markdown = create_most_recent_markdown(data)
st.markdown(f"<center> <h1> Most Recent COVID-19 News Headlines from {country_name}</h1></center>", unsafe_allow_html=True)
st.markdown("<center>" + md.convert(top_results_markdown) + "</center>", unsafe_allow_html=True)


# st.write(data)
# Sentiment Trends

@st.cache()
def get_sources(country):
    newsapi = NewsApiClient(api_key=random.choice(api_keys))
    sources = newsapi.get_sources(country=country)
    sources = [x['id'] for x in sources['sources']]
    return sources


# set to update every 24 hours
@st.cache(ttl=60 * 60 * 24, max_entries=20)
def create_dataframe_last_30d(queries, sources):
    newsapi = NewsApiClient(api_key=random.choice(api_keys))
    fulldata = pd.DataFrame()
    for q in queries:
        for s in sources:
            print(s)
            json_data = newsapi.get_everything(q=q,
                                               language='en',
                                               from_param=str(date.today() - timedelta(days=3)),
                                               to=str(date.today()),
                                               sources=s,
                                               page_size=100,
                                               page=1, sort_by='relevancy'
                                               )
            if len(json_data['articles']) > 0:
                data = pd.DataFrame(json_data['articles'])
                fulldata = pd.concat([fulldata, data])
    if len(fulldata) > 0:
        fulldata['source'] = fulldata['source'].apply(lambda x: x['name'])
        fulldata['publishedAt'] = pd.to_datetime(fulldata['publishedAt'])
        fulldata = fulldata.drop_duplicates(subset='url').sort_values(by='publishedAt', ascending=False).reset_index()
    return fulldata




