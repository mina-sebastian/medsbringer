import unidecode
from pdfminer.high_level import extract_text
import re


def clean_leaflet(leaflet_no):
    leaflet_text = extract_text('data/raw_data_pdf/ro_meds_downloaded/'+str(leaflet_no)+'.pdf')

    without_diacritics = unidecode.unidecode(leaflet_text)
    start = without_diacritics.find('Cititi')
    compozitie_idx = without_diacritics.find('Compozitie')
    if start == -1:
        if compozitie_idx != -1:
            start = compozitie_idx
        else: 
            start = 0

    end = without_diacritics.find('Detinatorul')
    leaflet_text = leaflet_text[start:end]
    without_diacritics = without_diacritics[start:end]

    street_idx = without_diacritics.find('Str.')
    report_idx = without_diacritics.find('Raportand')
    if street_idx and report_idx:
        leaflet_text = leaflet_text[:street_idx] + leaflet_text[report_idx:]
        without_diacritics = without_diacritics[:street_idx] + without_diacritics[report_idx:]

    producator_idx = without_diacritics.find('Producator')
    if producator_idx != -1:
        leaflet_text = leaflet_text[:(producator_idx - 2)]

    leaflet_text = re.sub(r"\n[0-9]", "\n", leaflet_text, flags=re.MULTILINE)
    leaflet_text = re.sub(r"[•–-]", "", leaflet_text, flags=re.MULTILINE)
    leaflet_text = re.sub(r"\n[\. ]", "\n", leaflet_text, flags=re.MULTILINE)
    leaflet_text = re.sub(r"^\s+", "", leaflet_text, flags=re.MULTILINE)
    leaflet_text = re.sub(r"(?m)^\s*\n", "", leaflet_text)
    leaflet_text = leaflet_text.strip()

    return leaflet_text