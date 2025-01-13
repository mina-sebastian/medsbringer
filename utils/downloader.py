import json
import os
import time
import requests
from tqdm import tqdm

failures = []


def download_file(url, filename, path='./raw_data'):
    response = requests.get(url)
    # if file exists, add a number to the filename
    i = 1
    while os.path.exists(os.path.join(path, filename)):
        filename = filename.split('.')[0] + str(i) + '.' + filename.split('.')[1]
        i += 1
    with open(os.path.join(path, filename), 'wb') as file:
        if response.status_code != 200:
            return False
        file.write(response.content)
    return True


def read_meds_list_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)


def download_meds(meds_list, output_dir, delay=10, startFrom=156):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, med in tqdm(enumerate(meds_list)):
        if i < startFrom:
            continue
        filename = f"{i}.pdf"

        success = download_file(med['url'], filename, output_dir)
        time.sleep(delay)
        if not success:
            print(f"Unsuccessful download for {filename} at index {i}")
            failures.append(med['url'])
            continue


if __name__ == '__main__':
    meds_list = read_meds_list_json('../data/raw_data_links/meds_list_EN.json')
    download_meds(meds_list, '../data/raw_data_pdf/en_meds_downloaded')
