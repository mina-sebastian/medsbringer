import unidecode
import os
import json
import re
from util_functions import clean_leaflet
import csv
import pandas as pd

def calculate_percents(filenames, llm_folder):
    data = {
        "Leaflet id": [],
        "Chat GPT excerpts %": []
    }

    for f in filenames:
        leaflet_no = int(f.split('_')[0])
        with open(llm_folder + '/' + f, 'r') as file:
            print(file.name)
            leaflet_llm = json.load(file)
        leaflet_text = clean_leaflet(leaflet_no)

        # concatenate excerpts
        if len(leaflet_llm):
            leaflet_llm = ' '.join(list(map(lambda x : x['excerpt'], leaflet_llm)))
        else:
            leaflet_llm = ''

        # get rid of diacritics
        leaflet_llm = unidecode.unidecode(leaflet_llm)
        leaflet_text = unidecode.unidecode(leaflet_text)

        leaflet_llm_words_no = len(leaflet_llm.split())
        leaflet_text_words_no = len(leaflet_text.split())

        llm_percent = leaflet_llm_words_no * 100.0 / leaflet_text_words_no

        data['Leaflet id'].append(leaflet_no)
        data['Chat GPT excerpts %'].append(llm_percent)

    df = pd.DataFrame(data)

    llm_mean = df.loc[:, 'Chat GPT excerpts %'].mean()

    df.loc[len(df)] = ['Average', llm_mean]

    return df


output_csv = "LLM_extractions/results/complexity.csv"

llm_folder = "LLM_extractions/results/llm_results"

filenames = [f for f in os.listdir(llm_folder) if f.endswith(".json")]
filenames.sort(key=lambda x : int(x.split('_')[0]))

df = calculate_percents(filenames=filenames, llm_folder=llm_folder)
df.to_csv(output_csv, index=False)