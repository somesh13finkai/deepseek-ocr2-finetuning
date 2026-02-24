"""
Phase 1, Task 2: Generate Weak Supervision Labels (Knowledge Distillation)
===========================================================================
Iterates over local hotel invoice PDFs in ./templates, sends each to
Gemini 2.5 Pro via Vertex AI, and saves structured "Silver" JSON labels
to ./silver_labels.

Authentication:
    Uses the service-account key at ./credentials.json via the
    GOOGLE_APPLICATION_CREDENTIALS environment variable (set at runtime).

Usage:
    python generate_silver_labels.py
"""

import os
import sys
import json
import base64
import logging
import pathlib
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from google.cloud import aiplatform
from google.api_core.exceptions import ResourceExhausted, GoogleAPICallError
from vertexai.generative_models import (
    GenerativeModel,
    Part,
    GenerationConfig,
)
import vertexai

from tqdm import tqdm
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

LOCATION = "us-central1"
MODEL_ID = "gemini-2.5-pro"

TEMPLATES_DIR = pathlib.Path("./templates")
OUTPUT_DIR = pathlib.Path("./silver_labels")
CREDENTIALS_PATH = pathlib.Path("./credentials.json")
FAILED_LOG = pathlib.Path("./failed_extractions.log")

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger("silver_label_gen")
logger.setLevel(logging.INFO)

# Console handler (INFO)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(console_handler)

# File handler for failures (WARNING+)
file_handler = logging.FileHandler(FAILED_LOG, mode="a", encoding="utf-8")
file_handler.setLevel(logging.WARNING)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
)
logger.addHandler(file_handler)

# ============================================================================
# RESPONSE SCHEMA (Hotel Invoice)
# ============================================================================
# This dict mirrors a Pydantic-style schema and is passed directly to the
# Vertex AI response_schema parameter to enforce structural compliance.

HOTEL_INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "hotel_name": {
            "type": "string",
            "description": "Full legal name of the hotel or property.",
        },
        "hotel_address": {
            "type": "string",
            "description": "Full address of the hotel including city, state, zip, and country.",
        },
        "hotel_gstin": {
            "type": "string",
            "description": "GSTIN / Tax ID of the hotel (if present on the invoice).",
        },
        "invoice_number": {
            "type": "string",
            "description": "Unique invoice or folio number.",
        },
        "invoice_date": {
            "type": "string",
            "description": "Date the invoice was issued (YYYY-MM-DD).",
        },
        "guest_name": {
            "type": "string",
            "description": "Name of the guest or the billing entity.",
        },
        "guest_gstin": {
            "type": "string",
            "description": "GSTIN / Tax ID of the guest or billing entity (if present).",
        },
        "check_in_date": {
            "type": "string",
            "description": "Check-in date (YYYY-MM-DD).",
        },
        "check_out_date": {
            "type": "string",
            "description": "Check-out date (YYYY-MM-DD).",
        },
        "room_number": {
            "type": "string",
            "description": "Room number assigned to the guest.",
        },
        "currency": {
            "type": "string",
            "description": "ISO 4217 currency code (e.g. INR, USD, THB).",
        },
        "line_items": {
            "type": "array",
            "description": "Itemized charges on the invoice.",
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "Description of the charge (e.g. Room Rent, Laundry, F&B).",
                    },
                    "hsn_sac_code": {
                        "type": "string",
                        "description": "HSN/SAC code for the line item (if present).",
                    },
                    "quantity": {
                        "type": "string",
                        "description": "Quantity or number of nights/units.",
                    },
                    "unit_price": {
                        "type": "string",
                        "description": "Price per unit before tax.",
                    },
                    "amount": {
                        "type": "string",
                        "description": "Total amount for this line item before tax.",
                    },
                },
                "required": ["description", "amount"],
            },
        },
        "subtotal": {
            "type": "string",
            "description": "Sum of all line items before taxes.",
        },
        "cgst_amount": {
            "type": "string",
            "description": "Central GST amount (Indian invoices).",
        },
        "sgst_amount": {
            "type": "string",
            "description": "State GST amount (Indian invoices).",
        },
        "igst_amount": {
            "type": "string",
            "description": "Integrated GST amount (Indian invoices, inter-state).",
        },
        "service_tax": {
            "type": "string",
            "description": "Any other service tax or VAT amount.",
        },
        "total_tax": {
            "type": "string",
            "description": "Total tax charged on the invoice.",
        },
        "total_amount": {
            "type": "string",
            "description": "Grand total payable including taxes.",
        },
        "payment_mode": {
            "type": "string",
            "description": "Mode of payment (e.g. Cash, Credit Card, UPI, Company Credit).",
        },
    },
    "required": [
        "hotel_name",
        "invoice_number",
        "invoice_date",
        "guest_name",
        "line_items",
        "total_amount",
    ],
}

# ============================================================================
# EXTRACTION PROMPT
# ============================================================================

EXTRACTION_PROMPT = """You are an expert document processing engine specializing in hotel invoices.

Your task is to extract ALL structured data from the provided hotel invoice PDF.

RULES:
1. Extract every field with MAXIMUM PRECISION. Transcription must be an EXACT COPY of the text as it appears on the page.
2. DO NOT normalize dates or monetary values. Preserve all commas, currency symbols, and original date formats exactly as printed.
3. If a field is not present on the invoice, omit it from the response.
4. For line_items, capture EVERY individual charge row visible on the invoice.
5. Pay special attention to tax breakdowns (CGST, SGST, IGST) which are critical for GST compliance.
6. Do NOT hallucinate or infer values that are not explicitly printed on the document.

Extract the data now."""

# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def init_vertex_ai() -> None:
    """Initialize Vertex AI SDK with project credentials."""
    if not CREDENTIALS_PATH.exists():
        raise FileNotFoundError(
            f"Credentials file not found at {CREDENTIALS_PATH}. "
            "Place your service account JSON key there."
        )

    # Load project_id directly from the service account credentials
    with open(CREDENTIALS_PATH, "r", encoding="utf-8") as f:
        creds_data = json.load(f)
        project_id = creds_data.get("project_id")
        
    if not project_id:
        raise ValueError("The key 'project_id' was not found in credentials.json.")

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS_PATH.resolve())

    vertexai.init(project=project_id, location=LOCATION)
    logger.info(f"Vertex AI initialized | project={project_id} | location={LOCATION}")


def get_pdf_files() -> list[pathlib.Path]:
    """Collect all PDF files from the templates directory."""
    if not TEMPLATES_DIR.exists():
        raise FileNotFoundError(f"Templates directory not found: {TEMPLATES_DIR}")

    pdf_files = sorted(TEMPLATES_DIR.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {TEMPLATES_DIR}")
    return pdf_files


def load_pdf_as_part(pdf_path: pathlib.Path) -> Part:
    """Read a local PDF and return it as a Vertex AI Part with inline data."""
    return Part.from_data(data=pdf_path.read_bytes(), mime_type="application/pdf")


@retry(
    retry=retry_if_exception_type(ResourceExhausted),
    wait=wait_exponential(multiplier=2, min=5, max=120),
    stop=stop_after_attempt(5),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)
def call_vertex_api(model: GenerativeModel, pdf_part: Part) -> str:
    """
    Send a PDF to Gemini via Vertex AI and return the raw JSON string.

    Retries up to 5 times with exponential backoff on ResourceExhausted
    (quota) errors. All other exceptions propagate immediately.
    """
    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=HOTEL_INVOICE_SCHEMA,
        temperature=0.0,
    )

    response = model.generate_content(
        [pdf_part, EXTRACTION_PROMPT],
        generation_config=generation_config,
    )

    return response.text


def extract_single_invoice(
    model: GenerativeModel, pdf_path: pathlib.Path
) -> dict | None:
    """
    End-to-end extraction for a single PDF.

    Returns the parsed JSON dict on success, or None on failure.
    Failures are logged to the failed_extractions.log file.
    """
    try:
        pdf_part = load_pdf_as_part(pdf_path)
        raw_json = call_vertex_api(model, pdf_part)
        parsed = json.loads(raw_json)
        return parsed

    except ResourceExhausted as e:
        logger.warning(f"QUOTA_EXHAUSTED | {pdf_path.name} | {e}")
        return None

    except GoogleAPICallError as e:
        logger.warning(f"API_ERROR | {pdf_path.name} | {type(e).__name__}: {e}")
        return None

    except json.JSONDecodeError as e:
        logger.warning(f"JSON_PARSE_ERROR | {pdf_path.name} | {e}")
        return None

    except Exception as e:
        logger.warning(f"UNEXPECTED_ERROR | {pdf_path.name} | {type(e).__name__}: {e}")
        return None


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def main() -> None:
    """
    Main batch processing loop.

    1. Initialize Vertex AI
    2. Collect PDFs from ./templates
    3. For each PDF, call Gemini and save the JSON to ./silver_labels
    4. Skip files that already have a corresponding JSON output (resume support)
    5. Track progress with tqdm, log failures to file
    """
    logger.info("=" * 60)
    logger.info("SILVER LABEL GENERATION — Phase 1, Task 2")
    logger.info(f"Started at {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # --- Step 1: Init ---
    init_vertex_ai()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 2: Collect PDFs ---
    pdf_files = get_pdf_files()
    if not pdf_files:
        logger.info("No PDF files found. Exiting.")
        return

    # --- Step 3: Initialize model ---
    model = GenerativeModel(MODEL_ID)
    logger.info(f"Model loaded: {MODEL_ID}")

    # --- Step 4: Batch process ---
    success_count = 0
    skip_count = 0
    fail_count = 0

    for pdf_path in tqdm(pdf_files, desc="Extracting invoices", unit="file"):
        # Build output path
        output_path = OUTPUT_DIR / pdf_path.with_suffix(".json").name

        # Resume support: skip if output already exists
        if output_path.exists():
            skip_count += 1
            continue

        # Extract
        result = extract_single_invoice(model, pdf_path)

        if result is not None:
            # Save JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            success_count += 1
        else:
            fail_count += 1

    # --- Step 5: Summary ---
    logger.info("=" * 60)
    logger.info("BATCH COMPLETE")
    logger.info(f"  Total PDFs   : {len(pdf_files)}")
    logger.info(f"  Extracted    : {success_count}")
    logger.info(f"  Skipped      : {skip_count}")
    logger.info(f"  Failed       : {fail_count}")
    logger.info(f"  Output dir   : {OUTPUT_DIR.resolve()}")
    if fail_count > 0:
        logger.info(f"  Failure log  : {FAILED_LOG.resolve()}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
