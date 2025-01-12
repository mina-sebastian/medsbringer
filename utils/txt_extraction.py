import os
from PyPDF2 import PdfReader

input_folder = "../raw_data_pdf/en_meds_downloaded/"
output_file = "../raw_data_txt/en_meds.txt"


def extract_text_from_pdfs(input_folder, output_file):
    #sorted list of PDF to writhem in order
    pdf_files = sorted(
        [f for f in os.listdir(input_folder) if f.endswith(".pdf")],
        key=lambda x: int(os.path.splitext(x)[0])
    )

    #set to track content hashes to avoid writing duplicate content
    processed_hashes = set()

    def compute_content_hash(content):
        """hash func to avoid writing duplicate content"""
        import hashlib
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def extract_section_from_text(text, section_header="B. PACKAGE LEAFLET"):
        """Extract text starting from the specified section header, excluding the header"""
        #find where the section starts
        start_idx = text.find(section_header)
        if start_idx != -1:
            # taking ext after the header
            return text[start_idx + len(section_header):].strip()
        return ""

    with open(output_file, "w", encoding="utf-8") as outfile:
        for pdf_file in pdf_files:
            pdf_path = os.path.join(input_folder, pdf_file)
            try:
                reader = PdfReader(pdf_path)
                text = ""
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"

                #check if the content is duplicate
                cleaned_text = text.strip()
                if not cleaned_text:
                    print(f"File {pdf_file} had no readable content. Skipping.")
                    continue

                content_hash = compute_content_hash(cleaned_text)
                if content_hash in processed_hashes:
                    #skip duplicates
                    continue

                #take the section starting from "B. PACKAGE LEAFLET"
                section_text = extract_section_from_text(cleaned_text)

                if section_text:  #msg if section not found
                    outfile.write(section_text + "\n")
                    processed_hashes.add(content_hash)

            except Exception as e:
                print(f"Error processing {pdf_file}: {e}")

    print("ready to go")


extract_text_from_pdfs(input_folder, output_file)

