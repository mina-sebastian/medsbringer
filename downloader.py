import json
import os
import time
import requests

def download_file(url, filename, path='.'):
    response = requests.get(url)
    # if file exists, add a number to the filename
    i = 1
    while os.path.exists(os.path.join(path, filename)):
        filename = filename.split('.')[0] + str(i) + '.' + filename.split('.')[1]
        i += 1
    with open(os.path.join(path, filename), 'wb') as file:
        file.write(response.content)

def read_meds_list_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def download_meds(meds_list, output_dir, delay=1, startFrom=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, med in enumerate(meds_list):
        if i < startFrom:
            continue
        download_file(med['url'], str(i) + '.pdf', output_dir)
        time.sleep(delay)


if __name__ == '__main__':
    meds_list = read_meds_list_json('meds_list_RO.json')
    download_meds(meds_list, 'ro_meds_downloaded', startFrom=501)