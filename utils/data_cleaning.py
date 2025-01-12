import re

input_file = "../raw_data_txt/en_meds.txt"
output_file = "../clean_data_txt/en_meds.txt"

def clean_text(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as infile:
        # Read the file content
        text = infile.read()
    #remove leading spaces, lines or tabs from each line
    cleaned_text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)

    with open(output_file, "w", encoding="utf-8") as outfile:
        outfile.write(cleaned_text)

    print(f"Text cleaned and saved to {output_file}")

clean_text(input_file, output_file)
