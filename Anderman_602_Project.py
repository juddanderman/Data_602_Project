import pandas as pd
import numpy as np
import scipy as sp
import requests
import math
import re
import geopandas as gpd
import matplotlib.pyplot as plt
import pysal as ps
import seaborn as sns
import folium
import tabula
import json
import zipfile
import io
from bs4 import BeautifulSoup
from pandas.tools.plotting import scatter_matrix

def soda_data(end_pt):
    token = 'Usj2xbIj0MccORApWzL94Y1dh'

    url = end_pt + '?$$app_token=' + token

    q_count = requests.get(url + '&$select=count(*)')
    recs = pd.read_json(q_count.text)["count"][0]

    if recs <= 50000:
        query = url + '&$limit=' + str(recs)
        r = requests.get(query)
        data = pd.read_json(r.text)
    else:
        pages = int(math.ceil(float(recs) / 50000))
        query = url + '&$limit=50000&$offset='
        df = []

        for i in range(0,pages):
            r = requests.get(query + str(i * 50000))
            df.append(pd.read_json(r.text))

        data = pd.concat(df)

    return data

def lin_func(x, m, b):
    return m * x + b

def get_unzip_shp():
    cwd = os.getcwd()
    url_zip = 'https://github.com/juddanderman/Data_602_Project/raw/master/UHF_42_DOHMH_2009.zip'
    shp_data = requests.get(url_zip, stream = True)
    
    with zipfile.ZipFile(io.BytesIO(shp_data.content)) as zf:       
        zf.extractall(cwd)

def corr_map_stacked(df, air_q, asthma):
    uhf42 = gpd.GeoDataFrame.from_file(
        'UHF_42_DOHMH_2009/UHF_42_DOHMH_2009.shp'
        )
    uhf42 = pd.merge(uhf42, df, on = 'UHFCODE', how = 'left')
    uhf42 = uhf42.iloc[1:42]
    f, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 6))

    df.plot.scatter(air_q, asthma, ax = ax1)
    ax1.set_xlabel(air_q)
    ax1.set_ylabel(asthma.replace('_', ' '))
    coef = sp.optimize.curve_fit(lin_func, df[air_q], df[asthma])[0]
    if (coef[1] >= 0):
        eq = 'admit_rate = {:.1f} * air_quality + {:.1f}'.format(
            coef[0], coef[1]
            )
    else:
        eq = 'admit_rate = {:.1f} * air_quality - {:.1f}'.format(
            coef[0], -coef[1]
            )
    ax1.plot(df[air_q], lin_func(df[air_q], *coef), 'r-')
    r_sq = np.corrcoef(df[air_q], df[asthma])[0, 1]**2

    txt = eq + ', R^2 = {:.3f}'.format(r_sq)
    ax1.set_title(txt)
    
    uhf42 = uhf42.to_crs(epsg = '4326')
    ax2.patch.set_alpha(1)
    a = uhf42.plot(column = air_q, scheme = 'QUANTILES', k = 5,
               cmap = 'YlOrRd', alpha = 0.7, ax = ax2, 
               edgecolor = 'white', legend = True)
    leg1 = ax2.get_legend()
    leg1._set_loc(6)
    leg1.set_title(air_q)
     
    b = uhf42.plot(column = asthma, scheme = 'QUANTILES', k = 5,
               cmap = 'YlGnBu', alpha = 0.7, ax = ax2,
               edgecolor = 'white', legend = True)
    leg2 = ax2.get_legend()
    leg2._set_loc(2)

    if asthma == 'PDI_14':
        lab = 'Admission Rate 2-17 yrs'
    elif asthma == 'PQI_15':
        lab = 'Admission Rate 18-39 yrs'
    elif asthma == 'PQI_05':
        lab = 'Admission Rate 40+ yrs'

    leg2.set_title(lab)
    f.gca().add_artist(leg1)
    ax2.set_title('United Health Fund (UHF) Neighborhood')
    if df.name == 'inpatient':
        title = ('NYC Air Quality & Asthma Related Hospitalizations ' +
                 '(All Inpatient Admissions)')
    elif df.name == 'medicaid':
        title = ('NYC Air Quality & Asthma Related Hospitalizations ' +
                 '(All Medicaid Admissions)')
    plt.suptitle(title)
    return f

def folium_map(df, air_q, asthma):
    uhf42 = gpd.GeoDataFrame.from_file(
        'UHF_42_DOHMH_2009/UHF_42_DOHMH_2009.shp'
        )
    uhf42 = pd.merge(uhf42, df, on = 'UHFCODE', how = 'left')
    uhf42 = uhf42.iloc[1:42]
    uhf_str = uhf42.to_crs(epsg = '4326').to_json()
    m = folium.Map(location = [40.73, -74], zoom_start = 10)
    m.choropleth(geo_str = uhf_str,
                 data = uhf42,
                 columns = ['UHFCODE', air_q],
                 key_on = 'feature.properties.UHFCODE',
                 fill_color = 'YlOrRd',
                 fill_opacity = 0.7,
                 legend_name = air_q)

    if asthma == 'PDI_14':
        lab = 'Admission Rate 2-17 yrs'
    elif asthma == 'PQI_15':
        lab = 'Admission Rate 18-39 yrs'
    elif asthma == 'PQI_05':
        lab = 'Admission Rate 40+ yrs'

    m.choropleth(geo_str = uhf_str,
                 data = uhf42,
                 columns = ['UHFCODE', asthma],
                 key_on = 'feature.properties.UHFCODE',
                 fill_color = 'YlGnBu',
                 fill_opacity = 0.7,
                 legend_name = lab)

    return m
    
if __name__ == "__main__":
####    set plotting style    
    sns.set_style('ticks')

####    API endpoints for air quality and inpatient discharges data
##    ep = ['https://data.cityofnewyork.us/resource/ah89-62h9.json',
##          'https://health.data.ny.gov/resource/v8gb-z2mi.json',
##          'https://health.data.ny.gov/resource/w9ks-guwu.json',
##          'https://health.data.ny.gov/resource/73ft-mcyi.json',
##          'https://health.data.ny.gov/resource/3n9x-27pa.json'
##          ]

####    fetch NYC air quality and NYS inpatient discharges data
##    air_qual = soda_data(ep[0])
##    pdi_medicaid_peds = soda_data(ep[1])
##    pdi_pediatric = soda_data(ep[2])
##    pqi_medicaid_adult = soda_data(ep[3])
##    pqi_adult = soda_data(ep[4])

####    write air quality data to CSV
##    air_qual.to_csv('Air_Quality.csv')
    
####    filter inpatient datasets to asthma hospitalizations measures  
##    pdi_medicaid_peds = (
##        pdi_medicaid_peds[pdi_medicaid_peds['pdi_number'] == 'PDI_14']
##        )                         
##    pdi_pediatric = (
##        pdi_pediatric[pdi_pediatric['pdi_number'] == 'PDI_14']
##        )
##    pqi_medicaid_adult = (
##        pqi_medicaid_adult[(pqi_medicaid_adult['pqi_number'].
##                            isin(['PQI_05', 'PQI_15'])) &
##                           (pqi_medicaid_adult['dual_status'].
##                            isin(['Total', 'TOTAL']))]
##        )
##    pqi_adult = (
##        pqi_adult[pqi_adult['pqi_number'].isin(['PQI_05', 'PQI_15'])]
##        )

####    write asthma hospitalizations datasets to CSV files    
##    pdi_medicaid_peds.to_csv('PDI_Medicaid_Pediatric_Asthma.csv')
##    pdi_pediatric.to_csv('PDI_Pediatric_Asthma.csv')
##    pqi_medicaid_adult.to_csv('PQI_Medicaid_Adult_Asthma.csv')
##    pqi_adult.to_csv('PQI_Adult_Asthma.csv')


#### import and clean previously fetched, processed, and saved data
    air_qual = pd.read_csv('Air_Quality.csv', index_col = 0)
    air_qual = air_qual.loc[air_qual['geo_type_name'] == 'UHF42']
    air_qual = air_qual.drop_duplicates()

    pdi_medicaid_peds = pd.read_csv('PDI_Medicaid_Pediatric_Asthma.csv',
                                    index_col = 0)
    pqi_medicaid_adult = pd.read_csv('PQI_Medicaid_Adult_Asthma.csv',
                                     index_col = 0)
    pdi_medicaid_peds = pdi_medicaid_peds[[
        'discharge_year',
        'medicaid_pdi_hospitalizations',
        'observed_rate_per_100_000_people',
        'pdi_number',
        'zip_code',
        'zip_code_medicaid_population']]
    pdi_medicaid_peds = pdi_medicaid_peds.rename(
        columns = {'discharge_year': 'year',
                   'medicaid_pdi_hospitalizations': 'hospitalizations',
                   'observed_rate_per_100_000_people': 'observed_rate',
                   'pdi_number': 'measure',
                   'zip_code_medicaid_population': 'population'}
        )
    pqi_medicaid_adult = pqi_medicaid_adult[[
        'discharge_year',
        'medicaid_pqi_hospitalizations',
        'observed_rate_per_100_000_people',
        'pqi_number',
        'zip_code',
        'zip_code_medicaid_population']]
    pqi_medicaid_adult = pqi_medicaid_adult.rename(
        columns = {'discharge_year': 'year',
                   'medicaid_pqi_hospitalizations': 'hospitalizations',
                   'observed_rate_per_100_000_people': 'observed_rate',
                   'pqi_number': 'measure',
                   'zip_code_medicaid_population': 'population'}
        )

    medicaid_asthma = pd.concat([pdi_medicaid_peds, pqi_medicaid_adult],
                                ignore_index = True
                                )
    medicaid_asthma['zip_code'].unique()
    medicaid_asthma = medicaid_asthma[
        medicaid_asthma['zip_code'] != 'Statewide'].reset_index()
    medicaid_asthma.loc[medicaid_asthma['zip_code'] == '63', 'zip_code'] = '063'
    
    pdi_pediatric = pd.read_csv('PDI_Pediatric_Asthma.csv',
                                index_col = 0)
    pqi_adult = pd.read_csv('PQI_Adult_Asthma.csv',
                            index_col = 0)
    pdi_pediatric = pdi_pediatric[[
        'discharge_year',
        'observed_rate_per_100_000_people',
        'patient_zip_code',
        'pdi_number']]
    pdi_pediatric = pdi_pediatric.rename(
        columns = {'discharge_year': 'year',
                   'observed_rate_per_100_000_people': 'observed_rate',
                   'patient_zip_code': 'zip_code',
                   'pdi_number': 'measure'}
        )
    pqi_adult = pqi_adult[[
        'year',
        'observed_rate_per_100_000_people',
        'patient_zipcode',
        'pqi_number']]
    pqi_adult = pqi_adult.rename(
        columns = {'observed_rate_per_100_000_people': 'observed_rate',
                   'patient_zipcode': 'zip_code',
                   'pqi_number': 'measure'}
        )

    inpt_asthma = pd.concat([pdi_pediatric, pqi_adult],
                            ignore_index = True
                            )
    inpt_asthma['zip_code'].unique()
    inpt_asthma = inpt_asthma[
        inpt_asthma['zip_code'] != 'STATEWIDE'].reset_index()
    inpt_asthma.loc[inpt_asthma['zip_code'] == '6390', 'zip_code'] = '06390'

####    combine asthma admissions rates across all three age groups
    medicaid_14 = medicaid_asthma[medicaid_asthma['year'] == 2014]
    inpt_14 = inpt_asthma[inpt_asthma['year'] == 2014]

    grps = ['PDI_14', 'PQI_15', 'PQI_05']

####    exploratory data analysis and visualization
    fig1, axes = plt.subplots(3, sharex = True, figsize = (9, 6))
    for i, grp in enumerate(grps):
        medicaid_asthma.loc[medicaid_asthma['measure'] == grp].boxplot(
            'observed_rate',
            'year',
            ax = axes[i]
            )
    for ax in axes:
        ax.set_xlabel('')
        ax.set_title('')
    axes[2].set_xlabel('Discharge year')
    axes[0].set_ylabel('2-17 yrs')
    axes[1].set_ylabel('18-39 yrs')
    axes[2].set_ylabel('40+ yrs')
    t = ('Observed asthma admission rate per 100,000' +
         '\n' + '(NYS Medicaid inpatients)')
    fig1.suptitle(t)

    fig2, axes = plt.subplots(3, sharex = True, figsize = (9, 6))
    for i, grp in enumerate(grps):
        p99 = medicaid_asthma.loc[(medicaid_asthma['measure'] == grp),
                                  'observed_rate'].quantile(0.99)
        medicaid_asthma.loc[(medicaid_asthma['measure'] == grp) &
                            (medicaid_asthma['observed_rate'] < p99)].boxplot(
                                'observed_rate',
                                'year',
                                ax = axes[i]
                                )
    for ax in axes:
        ax.set_xlabel('')
        ax.set_title('')
    axes[2].set_xlabel('Discharge year')
    axes[0].set_ylabel('2-17 yrs')
    axes[1].set_ylabel('18-39 yrs')
    axes[2].set_ylabel('40+ yrs')
    t = ('Observed asthma admission rate per 100,000' +
         '\n' + '(NYS Medicaid inpatients)')
    fig2.suptitle(t)

    fig3, axes = plt.subplots(3, sharex = True, figsize = (9, 6))
    for i, grp in enumerate(grps):
        inpt_asthma.loc[inpt_asthma['measure'] == grp].boxplot(
            'observed_rate',
            'year',
            ax = axes[i]
            )
    for ax in axes:
        ax.set_xlabel('')
        ax.set_title('')
    axes[2].set_xlabel('Discharge year')
    axes[0].set_ylabel('2-17 yrs')
    axes[1].set_ylabel('18-39 yrs')
    axes[2].set_ylabel('40+ yrs')
    t = ('Observed asthma admission rate per 100,000' +
         '\n' + '(All NYS inpatients)')
    fig3.suptitle(t)
 
    fig4, axes = plt.subplots(3, sharex = True, figsize = (9, 6))
    for i, grp in enumerate(grps):
        p99 = inpt_asthma.loc[(inpt_asthma['measure'] == grp),
                                  'observed_rate'].quantile(0.99)       
        inpt_asthma.loc[(inpt_asthma['measure'] == grp) &
                        (inpt_asthma['observed_rate'] < p99)].boxplot(
                            'observed_rate',
                            'year',
                            ax = axes[i]
                            )
    for ax in axes:
        ax.set_xlabel('')
        ax.set_title('')
    axes[2].set_xlabel('Discharge year')
    axes[0].set_ylabel('2-17 yrs')
    axes[1].set_ylabel('18-39 yrs')
    axes[2].set_ylabel('40+ yrs')
    t = ('Observed asthma admission rate per 100,000' +
         '\n' + '(All NYS inpatients)')
    fig4.suptitle(t)

    fig5, axes = plt.subplots(1, 3, figsize = (9, 6))
    for i, grp in enumerate(grps):
        medicaid_14.loc[medicaid_14['measure'] == grp].hist(
            column = 'observed_rate',
            ax = axes[i],
            bins = 25
            )
    for ax in axes:
        ax.set_title('')
        ax.set_xlim(left = min([0, ax.get_xlim()[0]]),
                    right = ax.get_xlim()[1])
    axes[0].set(xlabel = '2-17 yrs',
           ylabel = 'Frequency'
           )
    axes[1].set(xlabel = '18-39 yrs',
           ylabel = ''
           )
    axes[2].set(xlabel = '40+ yrs',
           ylabel = ''
           )
    t = ('2014 asthma/COPD admission rate per 100,000' +
         '\n' + '(NYS Medicaid inpatients)')
    fig5.suptitle(t)

    fig6, axes = plt.subplots(1, 3, figsize = (9, 6))
    for i, grp in enumerate(grps):
        p99 = medicaid_14.loc[(medicaid_asthma['measure'] == grp),
                              'observed_rate'].quantile(0.99)
        medicaid_14.loc[(medicaid_14['measure'] == grp) &
                        (medicaid_14['observed_rate'] < p99)].hist(
                            column = 'observed_rate',
                            ax = axes[i],
                            bins = 12
                            )
    for ax in axes:
        ax.set_title('')
        ax.set_xlim(left = min([0, ax.get_xlim()[0]]),
                    right = ax.get_xlim()[1])
    axes[0].set(xlabel = '2-17 yrs',
           ylabel = 'Frequency'
           )
    axes[1].set(xlabel = '18-39 yrs',
           ylabel = ''
           )
    axes[2].set(xlabel = '40+ yrs',
           ylabel = ''
           )
    t = ('2014 asthma/COPD admission rate per 100,000' +
         '\n' + '(NYS Medicaid inpatients)')
    fig6.suptitle(t)

    fig7, axes = plt.subplots(1, 3, figsize = (9, 6))
    for i, grp in enumerate(grps):
        inpt_14.loc[inpt_14['measure'] == grp].hist(
            column = 'observed_rate',
            ax = axes[i],
            bins = 50
            )
    for ax in axes:
        ax.set_title('')
        ax.set_xlim(left = min([0, ax.get_xlim()[0]]),
                    right = ax.get_xlim()[1])
    axes[0].set(xlabel = '2-17 yrs',
           ylabel = 'Frequency'
           )
    axes[1].set(xlabel = '18-39 yrs',
           ylabel = ''
           )
    axes[2].set(xlabel = '40+ yrs',
           ylabel = ''
           )
    t = ('2014 asthma/COPD admission rate per 100,000' +
         '\n' + '(All NYS inpatients)')
    fig7.suptitle(t)

    fig8, axes = plt.subplots(1, 3, figsize = (9, 6))
    for i, grp in enumerate(grps):
        p99 = inpt_14.loc[(inpt_asthma['measure'] == grp),
                          'observed_rate'].quantile(0.99)
        inpt_14.loc[(inpt_14['measure'] == grp) &
                    (inpt_14['observed_rate'] < p99)].hist(
                        column = 'observed_rate',
                        ax = axes[i],
                        bins = 25
                        )
    for ax in axes:
        ax.set_title('')
        ax.set_xlim(left = min([0, ax.get_xlim()[0]]),
                    right = ax.get_xlim()[1])
    axes[0].set(xlabel = '2-17 yrs',
           ylabel = 'Frequency'
           )
    axes[1].set(xlabel = '18-39 yrs',
           ylabel = ''
           )
    axes[2].set(xlabel = '40+ yrs',
           ylabel = ''
           )
    t = ('2014 asthma/COPD admission rate per 100,000' +
         '\n' + '(All NYS inpatients)')
    fig8.suptitle(t)

    medicaid_14_wide = medicaid_14.pivot(index = 'zip_code',
                                         columns = 'measure',
                                         values = 'observed_rate'
                                         )[grps]
    scatter_matrix(medicaid_14_wide,
                   diagonal = 'kde',
                   figsize = (7, 7))
    t = ('2014 asthma/COPD admission rates across age groups' +
         '\n' + '(NYS Medicaid inpatients)')
    plt.suptitle(t)

    inpt_14_wide = inpt_14.pivot(index = 'zip_code',
                                 columns = 'measure',
                                 values = 'observed_rate'
                                 )[grps]
    scatter_matrix(inpt_14_wide,
                   diagonal = 'kde',
                   figsize = (7, 7))
    t = ('2014 asthma/COPD admission rates across age groups' +
         '\n' + '(All NYS inpatients)')
    plt.suptitle(t)

####    create crosswalk between UHF42 neighborhoods and ZIP codes    
    uhf_zips = []
    url = 'https://www.health.ny.gov/statistics/cancer/registry/appendix/neighborhoods.htm'
    html = requests.get(url).content
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table')

    rows = table.find_all('tr')
    for row in rows:
        cols = row.find_all('td', attrs={'headers': ['header2', 'header3']})
        cols = [ele.text.strip() for ele in cols]
        uhf_zips.append([ele for ele in cols if ele]) 

    uhf_zips = pd.DataFrame(uhf_zips)
    uhf_zips = uhf_zips.iloc[1:]
    uhf_zips = pd.concat(
        [pd.Series(row[0], row[1].split(','))
         for _, row in uhf_zips.iterrows()]
        ).reset_index()
    uhf_zips.rename(
        columns = {"index": "zip_code", 0: "neighborhood"},
        inplace = True)
    uhf_zips['zip_code'] = uhf_zips['zip_code'].str.strip()
    uhf_zips.sort_values('zip_code', inplace = True)
    uhf_zips.reset_index(drop = True, inplace = True)

    table = 'http://www1.nyc.gov/assets/doh/downloads/pdf/ah/zipcodetable.pdf'
    uhf_codes = tabula.read_pdf_table(table)
    uhf_codes = uhf_codes.iloc[:, [0, 2]]
    uhf_codes = pd.concat(
        [pd.Series(row[0], str(row[1]).split(','))
         for _, row in uhf_codes.iterrows()]
        ).reset_index()
    uhf_codes.rename(
        columns = {"index": "zip_code", 0: "neighborhood"},
        inplace = True)
    uhf_codes = uhf_codes[uhf_codes['zip_code'] != 'nan']
    uhf_codes['zip_code'] = uhf_codes['zip_code'].str.strip()
    uhf_codes.sort_values('zip_code', inplace = True)
    uhf_codes.reset_index(drop = True, inplace = True)
    uhf_codes['uhf_code'] = uhf_codes['neighborhood'].str[0:3] + '.0'

    uhf_zips = pd.merge(uhf_zips, uhf_codes, on = 'zip_code', how = 'left')
    uhf_zips.loc[pd.isnull(uhf_zips['uhf_code']), 'uhf_code'] = '305.0'
    uhf_zips = uhf_zips.iloc[:, [0, 1, 3]]
    uhf_zips.rename(
        columns = {'neighborhood_x': 'neighborhood'},
        inplace = True)
    uhf_zips['zip_3'] = uhf_zips['zip_code'].str[0:3]

####    write UHF-ZIP code crosswalk to CSV file for later use
##    uhf_zips.to_csv('UHF_ZIP_Codes.csv')

    air_qual['var'] = air_qual['name'] + ' ' + air_qual['measure']
    air_wide = air_qual.pivot_table(
        index = ['geo_entity_name', 'geo_entity_id'],
        columns = ['var'],
        values = 'data_valuemessage')
    air_wide.reset_index(inplace = True)
    air_wide['UHFCODE'] = air_wide['geo_entity_id'].astype('float64')

####    scatter plot matrix of NYC air quality indicators
    sm = scatter_matrix(
        air_wide[list(air_wide.columns[2:13]) +
                 list(air_wide.columns[23:26])],
        diagonal = 'kde',
        figsize = (7, 7)
        )
    for subaxis in sm:
        for ax in subaxis:
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
    t = ('NYC Air Quality Measures')
    plt.suptitle(t)

    medicaid_14 = medicaid_14.pivot_table(
        index = ['year', 'zip_code'],
        columns = 'measure',
        values = 'observed_rate').reset_index()
    medicaid_14 = pd.merge(medicaid_14,
                           uhf_zips,
                           left_on = ['zip_code'],
                           right_on = ['zip_3'])
    medicaid_14 = medicaid_14.groupby(
        ['neighborhood', 'uhf_code']
        )['PDI_14', 'PQI_05', 'PQI_15'].mean().reset_index()
    
    inpt_14 = inpt_14.pivot_table(index = ['year', 'zip_code'],
                                  columns = 'measure',
                                  values = 'observed_rate').reset_index()
    inpt_14 = pd.merge(inpt_14, uhf_zips, on = ['zip_code'])
    inpt_14 = inpt_14.groupby(
        ['neighborhood', 'uhf_code']
        )['PDI_14', 'PQI_05', 'PQI_15'].mean().reset_index()

    air = air_wide[list(air_wide.columns[2:13]) +
                   list(air_wide.columns[23:27])]
    air.columns = ['Avg Benzene',
                   'Avg Formaldehyde',
                   'NOx per km^2',
                   'PM2.5 per km^2',
                   'SO2 per km^2',
                   'Avg EC',
                   'Avg PM2.5',
                   'Avg NO',
                   'Avg NO2',
                   'Avg O3',
                   'Avg SO2',
                   'VMT/100 km^2 Total',
                   'VMT/100 km^2 Cars',
                   'VMT/100 km^2 Trucks',
                   'UHFCODE'
                   ]
        
    medicaid_14['uhf_code'] = medicaid_14['uhf_code'].apply(float)
    inpt_14['uhf_code'] = inpt_14['uhf_code'].apply(float)
     
    medicaid = pd.merge(medicaid_14, air,
                        left_on = ['uhf_code'], right_on = ['UHFCODE'])
    medicaid.name = 'medicaid'
    
    inpatient = pd.merge(inpt_14, air,
                         left_on = ['uhf_code'], right_on = ['UHFCODE'])
    inpatient.name = 'inpatient'

####    correlations between admissions rates and air quality measures    
    medicaid.corr().iloc[1:18, 1:4]
    inpatient.corr().iloc[1:18, 1:4]

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(medicaid.corr().iloc[1:18, 1:18],
                square = True, ax = ax1)
    ax1.set_title('NYC Medicaid Admissions')
    ax1.set_yticklabels(
        ax1.yaxis.get_ticklabels(),
        rotation = 0, fontsize = 7
        )
    ax1.set_xticklabels(
        ax1.xaxis.get_ticklabels(),
        rotation = 90, fontsize = 7
        )

    sns.heatmap(inpatient.corr().iloc[1:17, 1:17],
                square = True, ax = ax2)
    ax2.set_title('NYC All Inpatient Admissions')
    ax2.set_yticklabels(
        ax2.yaxis.get_ticklabels(),
        rotation = 0, fontsize = 7
        )
    ax2.set_xticklabels(
        ax2.xaxis.get_ticklabels(),
        rotation = 90, fontsize = 7
        )
    plt.suptitle('Correlation Matrix Heatmaps')

####    get and unzip UHF42 shapefile archive from remote github repo
##    get_unzip_shp()

##    corr_map_stacked(medicaid, 'Avg O3', 'PDI_14')
    corr_map_stacked(inpatient, 'Avg SO2', 'PQI_05')
##
##    m = folium_map(inpatient, 'Avg SO2', 'PQI_05')
##    m.save('NYC_UHF_Asthma_Test.html')
##
    plt.show()

