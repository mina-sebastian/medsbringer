import re
import os
import csv
from PyPDF2 import PdfReader


def clean_text(text):
    """
    Cleans text by:
    1. Removing leading/trailing spaces or tabs from each line.
    2. Removing unwanted symbols like , " . / etc.
    3. Removing multiple spaces.
    4. Removing empty lines.
    """
    # Remove leading and trailing spaces or tabs from each line
    cleaned_text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)
    # Remove unwanted symbols
    cleaned_text = re.sub(r'[",./]', "", cleaned_text)
    # Replace multiple spaces with a single space
    cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)
    # Remove empty lines
    cleaned_text = re.sub(r"(?m)^\s*\n", "", cleaned_text)  # Multiline mode
    return cleaned_text.strip()


def extract_text_from_pdfs_to_csv(input_folder, output_csv):
    # Sorted list of PDF files to process in order
    pdf_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".pdf")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    with open(output_csv, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "extracted_text"])  # CSV header

        for idx, pdf_file in enumerate(pdf_files):
            pdf_path = os.path.join(input_folder, pdf_file)
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"

                cleaned_text = clean_text(text.strip())
                if not cleaned_text:
                    print(f"File {pdf_file} had no readable content. Skipping.")
                    continue

                # Write the cleaned text to the CSV
                writer.writerow([idx, cleaned_text])

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    print("Extraction to CSV completed successfully.")


# Example usage
input_folder = "../raw_data_pdf/da_meds_downloaded/"
output_csv = "../raw_data_csv/da_meds.csv"
extract_text_from_pdfs_to_csv(input_folder, output_csv)
