import pandas as pd
import csv
import json
import os
import time
import requests


csv_file = pd.read_csv('Export-authorised medicinal products-human use-20250111.csv', delimiter=";")
csv_file['Name'] = csv_file['Name'].str.lstrip("\t ")
csv_file = csv_file.sort_values(by=['Name'])

## FR

french_info = csv_file[['Name', 'Company', 'URL Leaflet FR']].copy()
french_info = french_info.rename(columns={"URL Leaflet FR": "Url_fr"})
french_info = french_info.dropna()

# exclude entries that are in nl or don't have a url
french_info = french_info[french_info['Url_fr'].str.contains('https', na=False)]
french_info = french_info[french_info['Url_fr'].str.contains('nl', na=True) == False]

# exclude entries that are product information
french_info = french_info[french_info['Url_fr'].str.contains('product-information', na=True) == False]

fr = []
for row in french_info.itertuples():
    fr.append({
            'nume': row.Name,
            'url': row.Url_fr,
            'firma': row.Company
        })

   
json.dump(fr, open('meds_list_FR.json', 'w'), indent=4)


## NL

nl_info = csv_file[['Name', 'Company', 'URL Leaflet NL']].copy()
nl_info = nl_info.rename(columns={"URL Leaflet NL": "Url_nl"})
nl_info = nl_info.dropna()

# exclude entries that don't have a url
nl_info = nl_info[nl_info['Url_nl'].str.contains('https', na=False)]

# exclude entries that are product information
nl_info = nl_info[nl_info['Url_nl'].str.contains('product-information', na=True) == False]

nl = []
for row in nl_info.itertuples():
    nl.append({
            'nume': row.Name,
            'url': row.Url_nl,
            'firma': row.Company
        })
    
json.dump(nl, open('meds_list_NL.json', 'w'), indent=4)


## DE

de_info = csv_file[['Name', 'Company', 'URL Leaflet DE']].copy()
de_info = de_info.rename(columns={"URL Leaflet DE": "Url_de"})
de_info = de_info.dropna()

# exclude entries that are in nl or don't have a url
de_info = de_info[de_info['Url_de'].str.contains('https', na=False)]

# exclude entries that are product information
de_info = de_info[de_info['Url_de'].str.contains('product-information', na=True) == False]

de = []
for row in de_info.itertuples():
    de.append({
            'nume': row.Name,
            'url': row.Url_de,
            'firma': row.Company
        })
    
json.dump(de, open('meds_list_DE.json', 'w'), indent=4)