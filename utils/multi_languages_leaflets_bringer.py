import pandas as pd
import csv
import json
import os
import time
import requests
import openpyxl


# csv_file = pd.read_csv('medicines_output_medicines_en.xlsx', delimiter='\t', encoding='ISO-8859-9')

dfs = pd.read_excel(r'./data/raw_data_links/medicines_output_medicines_en.xlsx',engine='openpyxl')
dfs = dfs[dfs['Category'] == 'Human']
dfs = dfs[dfs['Medicine status'] == 'Authorised']

dfs = dfs[['Name of medicine', 'Marketing authorisation developer / applicant / holder', 'Medicine URL']].copy()
dfs = dfs.rename(columns={"Name of medicine": "Name", "Marketing authorisation developer / applicant / holder": "Company", "Medicine URL": "Url"})
dfs = dfs.sort_values(by=['Name'])

# ## EN
# info = []
# standard_url = 'https://www.ema.europa.eu/en/documents/product-information/'
# suffix = '-epar-product-information_en.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('../data/raw_data_links/meds_list_EN.json', 'w'), indent=4)


# ## FR

# info = []
# standard_url = 'https://www.ema.europa.eu/fr/documents/product-information/'
# suffix = '-epar-product-information_fr.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('../data/raw_data_links/meds_list_FR.json', 'w'), indent=4)


# # ## RO

# info = []
# standard_url = 'https://www.ema.europa.eu/ro/documents/product-information/'
# suffix = '-epar-product-information_ro.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('../data/raw_data_links/meds_list_RO1.json', 'w'), indent=4)


# # # ## DE

# info = []
# standard_url = 'https://www.ema.europa.eu/de/documents/product-information/'
# suffix = '-epar-product-information_de.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('../data/raw_data_links/meds_list_DE.json', 'w'), indent=4)

# # ## IT

# info = []
# standard_url = 'https://www.ema.europa.eu/it/documents/product-information/'
# suffix = '-epar-product-information_it.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('../data/raw_data_links/meds_list_IT.json', 'w'), indent=4)

# # ## DA

# info = []
# standard_url = 'https://www.ema.europa.eu/da/documents/product-information/'
# suffix = '-epar-product-information_da.pdf'
# for row in dfs.itertuples():
#     parts = row.Url.split('/')
#     m_name = parts[-1]

#     info.append({
#             'nume': row.Name,
#             'nume_cleaned': m_name,
#             'url': standard_url + m_name + suffix,
#             'firma': row.Company
#         })
    
# json.dump(info, open('./data/raw_data_links/meds_list_DA1.json', 'w'), indent=4)

# ## NL

info = []
standard_url = 'https://www.ema.europa.eu/nl/documents/product-information/'
suffix = '-epar-product-information_nl.pdf'
for row in dfs.itertuples():
    parts = row.Url.split('/')
    m_name = parts[-1]

    info.append({
            'nume': row.Name,
            'nume_cleaned': m_name,
            'url': standard_url + m_name + suffix,
            'firma': row.Company
        })
    
json.dump(info, open('./data/raw_data_links/meds_list_NL1.json', 'w'), indent=4)