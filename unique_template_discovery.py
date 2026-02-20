import os
import gc
import boto3
import imagehash
from pdf2image import convert_from_bytes, convert_from_path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION BLOCK ---
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "fink-hotel-invoice-scraped")
S3_PREFIX = os.getenv("S3_PREFIX", "")
TARGET_LIMIT = 1000
TEMPLATES_DIR = "./templates"
HASH_THRESHOLD = 12
# ---------------------------

AWS_ACCESS_KEY_ID = os.getenv("S3_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("S3_SECRET_ACCESS_KEY")

def get_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
    )

def main():
    if not os.path.exists(TEMPLATES_DIR):
        os.makedirs(TEMPLATES_DIR)

    try:
        s3 = get_s3_client()
    except Exception as e:
        print(f"Failed to initialize S3 client: {e}")
        return

    unique_template_hashes = []
    
    # ==========================================
    # PHASE 0: BOOTSTRAP (RESUME LOGIC)
    # ==========================================
    existing_files = [f for f in os.listdir(TEMPLATES_DIR) if f.lower().endswith('.pdf')]
    
    if existing_files:
        print(f"🔄 Found {len(existing_files)} existing templates locally. Hashing to resume...")
        for filename in tqdm(existing_files, desc="Bootstrapping local files"):
            try:
                path = os.path.join(TEMPLATES_DIR, filename)
                # Convert local file
                images = convert_from_path(path, first_page=1, last_page=1, fmt='jpeg')
                if images:
                    # Using hash_size=8 strictly to match the S3 hashes (64-bit)
                    h = imagehash.phash(images[0], hash_size=8)
                    unique_template_hashes.append(h)
            except Exception as e:
                pass # Skip silently if a local file is corrupted
                
    downloaded_count = len(unique_template_hashes)
    print(f"\n✅ Ready! Starting S3 scan with {downloaded_count} unique templates already in memory.")

    if downloaded_count >= TARGET_LIMIT:
        print(f"Target limit of {TARGET_LIMIT} already reached. Exiting.")
        return

    # ==========================================
    # PHASE 1: S3 DISCOVERY
    # ==========================================
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=BUCKET_NAME, Prefix=S3_PREFIX)

    try:
        for page in page_iterator:
            if 'Contents' not in page:
                continue

            for obj in tqdm(page['Contents'], desc="Scanning S3"):
                if downloaded_count >= TARGET_LIMIT:
                    print(f"\n🎯 Target limit of {TARGET_LIMIT} templates reached. Discovery complete.")
                    return

                key = obj['Key']
                if not key.lower().endswith('.pdf'):
                    continue

                filename = os.path.basename(key)
                local_path = os.path.join(TEMPLATES_DIR, filename)

                # FAST SKIP: If we already downloaded this exact file, skip the S3 GET request
                if os.path.exists(local_path):
                    continue

                try:
                    # Stream PDF bytes into memory
                    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
                    pdf_bytes = response['Body'].read()
                    
                    images = convert_from_bytes(pdf_bytes, first_page=1, last_page=1, fmt='jpeg')
                    
                    if not images:
                        continue
                        
                    phash = imagehash.phash(images[0], hash_size=8)

                    # Check for uniqueness against existing memory
                    is_unique = True
                    for existing_hash in unique_template_hashes:
                        if phash - existing_hash <= HASH_THRESHOLD:
                            is_unique = False
                            break
                    
                    if is_unique:
                        unique_template_hashes.append(phash)
                        downloaded_count += 1
                        
                        # Save the new template
                        with open(local_path, 'wb') as f:
                            f.write(pdf_bytes)

                except Exception:
                    pass
                finally:
                    # Explicit memory management for M1 Unified Memory
                    if 'pdf_bytes' in locals(): del pdf_bytes
                    if 'images' in locals(): del images
                    if 'response' in locals(): del response
                    gc.collect()

    except KeyboardInterrupt:
        print(f"\n⏸️ Process interrupted by user. Saved {downloaded_count} templates so far. Run again to resume.")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")

    print(f"\n🏁 Finished. Total unique templates in '{TEMPLATES_DIR}': {downloaded_count}")

if __name__ == "__main__":
    main()