import boto3
import os
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    's3',
    aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY")
)

BUCKET_NAME = "fink-hotel-invoice-scraped"
PREFIX = "" # Keep empty to scan entire bucket

def count_pdfs():
    print(f"🔍 Counting PDF objects in '{BUCKET_NAME}'...")
    paginator = s3.get_paginator('list_objects_v2')
    
    pdf_count = 0
    total_objects_scanned = 0

    # Using tqdm for a live update in the terminal
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=PREFIX):
        contents = page.get('Contents', [])
        total_objects_scanned += len(contents)
        
        # Filter for PDFs only
        pdfs_in_page = [obj for obj in contents if obj['Key'].lower().endswith('.pdf')]
        pdf_count += len(pdfs_in_page)
        
        print(f"Processed: {total_objects_scanned} | Found PDFs: {pdf_count}", end="\r")

    print(f"\n\n📊 Final Results:")
    print(f"Total Objects in Bucket: {total_objects_scanned}")
    print(f"Total PDF Invoices:      {pdf_count}")
    print(f"Non-PDF/Other files:     {total_objects_scanned - pdf_count}")

if __name__ == "__main__":
    count_pdfs()