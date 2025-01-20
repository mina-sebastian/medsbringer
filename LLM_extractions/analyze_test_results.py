import unidecode
import os
import json
import re
from util_functions import clean_leaflet
import csv
import pandas as pd

# find common substrings: https://stackoverflow.com/questions/18715688/find-common-substring-between-two-strings
def common_substrings(str1,str2, min_com):
    len1,len2=len(str1),len(str2)

    if len1 > len2:
        str1,str2=str2,str1 
        len1,len2=len2,len1
    #short string=str1 and long string=str2
    
    cs_array=[]
    for i in range(len1,min_com-1,-1):
        for k in range(len1-i+1):
            if (str1[k:i+k] in str2):
                flag=1
                for m in range(len(cs_array)):
                    if str1[k:i+k] in cs_array[m]:
                    #print(str1[k:i+k])
                        flag=0
                        break
                if flag==1:
                    cs_array.append(str1[k:i+k])
    return cs_array


def calculate_percents(filenames, llm_folder, human_folder):
    data = {
        "Leaflet id": [],
        "Chat GPT excerpts %": [],
        "Manual validation excerpts %": [],
        "Common excerpts %": [],
        "Common excerpts % of llm excerpts": []
    }

    for f in filenames:
        leaflet_no = int(f.split('_')[0])
        with open(llm_folder + '/' + f, 'r') as file:
            leaflet_llm = json.load(file)
        with open(human_folder + '/' + f, 'r') as file:
            leaflet_human = json.load(file)
        leaflet_text = clean_leaflet(leaflet_no)

        # concatenate excerpts
        leaflet_llm = ' '.join(list(map(lambda x : x['excerpt'], leaflet_llm)))
        leaflet_human = ' '.join(list(map(lambda x : x['excerpt'], leaflet_human)))

        # human excerpts don t have hyphens
        leaflet_llm = leaflet_llm.replace('-', '')

        # get rid of diacritics
        leaflet_llm = unidecode.unidecode(leaflet_llm)
        leaflet_human = unidecode.unidecode(leaflet_human)
        leaflet_text = unidecode.unidecode(leaflet_text)

        leaflet_llm = leaflet_llm.replace('\n', ' ')
        leaflet_human = leaflet_human.replace('\n', ' ')

        common_excerpts = common_substrings(leaflet_llm, leaflet_human, 15)
        common_excerpts = ' '.join(common_excerpts)

        leaflet_llm_words_no = len(leaflet_llm.split())
        leaflet_human_words_no = len(leaflet_human.split())
        leaflet_common_words_no = len(common_excerpts.split())
        leaflet_text_words_no = len(leaflet_text.split())

        llm_percent = leaflet_llm_words_no * 100.0 / leaflet_text_words_no
        human_percent = leaflet_human_words_no * 100.0 / leaflet_text_words_no
        common_percent = leaflet_common_words_no * 100.0 / leaflet_text_words_no
        common_percent_of_llm = leaflet_common_words_no * 100.0 / leaflet_llm_words_no

        data['Leaflet id'].append(leaflet_no)
        data['Chat GPT excerpts %'].append(llm_percent)
        data['Manual validation excerpts %'].append(human_percent)
        data['Common excerpts %'].append(common_percent)
        data['Common excerpts % of llm excerpts'].append(common_percent_of_llm)

    df = pd.DataFrame(data)

    llm_mean = df.loc[:, 'Chat GPT excerpts %'].mean()
    human_mean = df.loc[:, 'Manual validation excerpts %'].mean()
    common_mean = df.loc[:, 'Common excerpts %'].mean()
    common_of_llm_mean = df.loc[:, 'Common excerpts % of llm excerpts'].mean()

    df.loc[len(df)] = ['Average', llm_mean, human_mean, common_mean, common_of_llm_mean]

    return df

output_csv = "LLM_extractions/results/test/comparison.csv"

llm_folder = "LLM_extractions/results/test/llm_results"
human_folder = "LLM_extractions/results/test/human_results"

filenames = [f for f in os.listdir(llm_folder) if f.endswith(".json")]
filenames.sort(key=lambda x : int(x.split('_')[0]))

data = calculate_percents(filenames=filenames, llm_folder=llm_folder, human_folder=human_folder)
data.to_csv(output_csv, index=False)