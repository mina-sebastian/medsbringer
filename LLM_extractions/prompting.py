from openai import OpenAI
import json
import random
import pandas as pd
import re
from pdfminer.high_level import extract_text
import tiktoken
import unidecode
import os
from util_functions import clean_leaflet

def get_excerpts(system_prompt, user_prompt, leaflet_no, is_test=True):

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # write the extractions in files with the medicine's index as filename
    if is_test:
        filename = f"LLM_extractions/results/test/llm_results/{leaflet_no}_excerpts.json"
    else:
        filename = f"LLM_extractions/results/llm_results/{leaflet_no}_excerpts.json"

    with open(filename, "w") as outfile:
        outfile.write(completion.choices[0].message.content)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model("gpt-4o")
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_excerpts_for_leaflets(no_leaflets, system_prompt):
    idx = 0
    not_checked = [x for x in range(500) if x not in checked_leaflets]
    while idx < no_leaflets:
        print(idx)
        # random_leaflet_idx = random.randint(0, 500)
        random_leaflet_idx = random.choice(not_checked)

        try:
            leaflet_text = clean_leaflet(random_leaflet_idx)
        except:
            continue

        # no_tokens = num_tokens_from_string(leaflet_text)
            
        if random_leaflet_idx not in checked_leaflets:
            get_excerpts(system_prompt, leaflet_text, random_leaflet_idx, is_test=False)
            idx += 1
            checked_leaflets.append(random_leaflet_idx)
            not_checked.remove(random_leaflet_idx)


client = OpenAI()
system_prompt = 'You will be provided with the content of a medication package insert in romanian. The leaflet should be simple, readable and comprehensible, even to people with limited health literacy skills. Your task is to extract from the leaflet\'s text the passages that don\'t meet these criteria. Pretend the words are correctly spelled, ignore the fact that the hyphens are missing. Provide output in a JSON array as follows: [{"excerpt": "..."}, ... {"excerpt": "..."}]. Don\'t add any other text to the output but the array with the excerpts. Take care not to omit any passage difficult to understand. If the text does not contain passages with high-complexity then simply write an empty array.'
# system_prompt = 'Pretend you are a person with average education level and no medical literacy skills. You will be provided with a content of a medication package insert in romanian. Your task is to extract the passages that you find complex, difficult to understand. Provide output in a JSON array as follows: [{"excerpt": "..."}, ... {"excerpt": "..."}]. Don\'t add any other text to the output but the array with the excerpts. Take care not to omit any difficult passage. If the text does not contain passages with high-complexity then simply write an empty array.'
# checked_leaflets = []

# if there are already extracted leaflets, don t duplicate them
input_folder = "LLM_extractions/results/llm_results"
checked_leaflets = [int(f.split('_')[0]) for f in os.listdir(input_folder) if f.endswith(".json")]
print(len(checked_leaflets))

# for llm testing
# random_leaflet_idx = random.randint(0, 500)
# print(random_leaflet_idx)
# leaflet_text = clean_leaflet(random_leaflet_idx)
# print(leaflet_text)
# print(num_tokens_from_string(leaflet_text))
# get_excerpts(system_prompt, leaflet_text, random_leaflet_idx)


# for llm excerpts
# get_excerpts_for_leaflets(no_leaflets=100, system_prompt=system_prompt)