import re
import json
import logging
import requests
import csv
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel

# Configure logging to file for debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/Users/suchitsharma/Documents/GitHub/Apollo-zapier/debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
# In a real production app, use os.getenv() for these keys
CONFIG = {
    "EPFO_AUTH": "U9J096G8ID7UTKU",
    "INTERNAL_TOKEN": "a8225adf-c1d0-4a01-9853-4456c2b9a749",
    "PORTKEY_KEY": "YoOaYwcZkhrUD4Cmom8pD4IvD+v/",
    "PORTKEY_VIRTUAL": "google-springve-6cb775",
    "APOLLO_KEY": "OClRtaraAlYhkyzxkGdDHg",
    "HATCH_KEY": "U2FsdGVkX196jXioU4DcI_nIoFDW3wF_xUS9OPtT8444P3kZCdERg77CvPB6-ZWrr5UIMTbdBdxOIxhT2_zgrQ", 
    "CAMPAIGN_ID": "68d1103b41c0f700155cd927",
    "EMAIL_SENDER_ID": "621f02521d8f2b00f95fa455",
    "JOB_TITLES": ["HR", "Human Resources", "Recruiting", "Talent Acquisition", "People Operations", "CHRO", "Chief People Officer", "Head of HR", "Head of Talent", "HR Manager"],
    "PARALLEL_API_KEY": "dZE6nnwRdoHBP0YB4CAfwoy8L-wRrxchNlekN0Q0"
}

# Setup Logging (Prints to VS Code Terminal)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# --- MCA STATUS MAPPING ---
# MCA status mapping to standardized enum values
MCA_STATUS_MAPPING = {
    "ACTIVE": "ACTIVE",
    "STRIKE OFF": "STRIKE_OFF", 
    "UNDER LIQUIDATION": "UNDER_LIQUIDATION",
    "AMALGAMATED": "AMALGAMATED",
    "CONVERTED": "CONVERTED",
    "DISSOLVED": "DISSOLVED",
    "SUSPENDED": "SUSPENDED",
    "DORMANT": "DORMANT",
    "NOT_FOUND": "NOT_FOUND",
    "UNKNOWN": "NOT_FOUND"
}

# --- DATA MODELS ---
class WebhookPayload(BaseModel):
    # Adjust these keys based on exactly what your webhook sends
    company_name: Optional[str] = None
    company: Optional[str] = None  # Alternative field name
    email: Optional[str] = None
    name: Optional[str] = None
    phone: Optional[str] = None
    # You can add extra fields here if the webhook sends more

# --- HELPER FUNCTIONS ---

def clean_company_name(name: str) -> str:
    """Step 2: Clean company name"""
    if not name or not isinstance(name, str):
        return ""
    # Remove periods, collapse spaces, trim
    cleaned = re.sub(r'\.', ' ', name)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Capitalize first letter of each word
    return cleaned.title()

def format_date_yyyy_mm_dd(date_str: str) -> str:
    if not date_str or date_str.upper() == "N/A" or not date_str.strip():
        return "N/A"
    
    clean_date = date_str.strip()
    # Check if already YYYY-MM-DD
    if re.match(r'^\d{4}-\d{2}-\d{2}$', clean_date):
        return clean_date
        
    parts = re.split(r'[-/ ]+', clean_date)
    if len(parts) == 3:
        # Try to parse YYYY/MM/DD
        if len(parts[0]) == 4:
            return f"{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}"
        # Try to parse DD/MM/YYYY
        if len(parts[2]) == 4:
            return f"{parts[2]}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
            
    return "N/A"

def extract_company_id(raw_input: Any) -> str:
    """Step 6: Extract ID from messy API response or logs"""
    if isinstance(raw_input, dict):
        # Check nested paths
        candidates = [
            raw_input.get('data', {}).get('id'),
            raw_input.get('apiResponse', {}).get('data', {}).get('id'),
            raw_input.get('id')
        ]
        for val in candidates:
            if val: return str(val)
            
    if not isinstance(raw_input, str):
        return None

    # Regex extraction logic
    # 1. Match data: { id: 123 } pattern
    match = re.search(r'data\s*[:=]\s*\{\s*[^}]*id\s*[:=]\s*[\'"]?(\d{3,})[\'"]?', raw_input, re.IGNORECASE)
    if match: return match.group(1)
    
    # 2. JSON strict pattern
    match = re.search(r'"data"\s*:\s*\{\s*"id"\s*:\s*(\d{3,})', raw_input, re.IGNORECASE)
    if match: return match.group(1)
    
    # 3. Fallback ID pattern
    match = re.search(r'id\s*[:=]\s*[\'"]?(\d{3,})[\'"]?', raw_input, re.IGNORECASE)
    if match: return match.group(1)
    
    return None

# --- API SERVICES ---

def call_external_apis(cleaned_name: str):
    """Step 3: Call MCA and EPFO APIs with Parallel API fallback for MCA only"""
    logger.info(f"Calling APIs for: {cleaned_name}")
    
    mca_data = {}
    epfo_data = {}
    
    # MCA Call
    try:
        logger.info("🔍 Making MCA API call...")
        mca_payload = {"company": cleaned_name}
        mca_headers = {"authkey": CONFIG["EPFO_AUTH"]}
        
        logger.info(f"📤 MCA Request Body: {json.dumps(mca_payload, indent=2)}")
        logger.info(f"📤 MCA Request Headers: {json.dumps({k: v if k != 'authkey' else f'{v[:10]}...' for k, v in mca_headers.items()})}")
        
        mca_res = requests.post(
            "https://cin-lookup.befisc.com", 
            json=mca_payload, 
            headers=mca_headers
        )
        logger.info(f"📊 MCA API Status Code: {mca_res.status_code}")
        logger.info(f"📥 MCA Response Headers: {dict(mca_res.headers)}")
        
        if mca_res.ok:
            mca_data = mca_res.json()
            logger.info(f"✅ MCA API Response: {json.dumps(mca_data, indent=2)}")
            
            # Check if MCA API returned error status or no records - trigger Parallel API fallback
            response_status = mca_data.get('status')
            response_msg = mca_data.get('message', '').lower()
            if (response_status == 302 or response_status == 2 or 'source down' in response_msg or 
                'no record found' in response_msg or 'no records found' in response_msg):
                logger.warning(f"⚠️ MCA API returned error/no records (status: {response_status}, msg: {mca_data.get('message')})")
                logger.info("🔄 Calling Parallel API for MCA data due to MCA API failure/no records...")
                parallel_data = call_parallel_mca_fallback(cleaned_name)
                # Use Parallel API data for MCA
                mca_data = parallel_data
                
        else:
            logger.error(f"❌ MCA API Failed: {mca_res.status_code} - {mca_res.text}")
            logger.info("🔄 Calling Parallel API for MCA data due to MCA API failure...")
            parallel_data = call_parallel_mca_fallback(cleaned_name)
            mca_data = parallel_data
            
    except Exception as e:
        logger.error(f"❌ MCA API Error: {e}")
        logger.info("🔄 Calling Parallel API for MCA data due to MCA exception...")
        parallel_data = call_parallel_mca_fallback(cleaned_name)
        mca_data = parallel_data

    # EPFO Call - Always try EPFO API, regardless of MCA status
    try:
        logger.info("🔍 Making EPFO API call...")
        epfo_payload = {
            "establishment_name": cleaned_name,
            "consent": "Y",
            "consent_text": "We confirm that we have obtained the consent of the respective customer to fetch their details by using Establishment Name and the customer is aware of the purpose for which their data is sought for being processed and have given their consent for the same and such consent is currently valid and not withdrawn."
        }
        epfo_headers = {"authkey": CONFIG["EPFO_AUTH"]}
        
        logger.info(f"📤 EPFO Request Body: {json.dumps(epfo_payload, indent=2)}")
        logger.info(f"📤 EPFO Request Headers: {json.dumps({k: v if k != 'authkey' else f'{v[:10]}...' for k, v in epfo_headers.items()})}")
        
        epfo_res = requests.post(
            "https://establishment-details.befisc.com",
            json=epfo_payload,
            headers=epfo_headers
        )
        logger.info(f"📊 EPFO API Status Code: {epfo_res.status_code}")
        logger.info(f"📥 EPFO Response Headers: {dict(epfo_res.headers)}")
        
        if epfo_res.ok:
            epfo_data = epfo_res.json()
            logger.info(f"✅ EPFO API Response: {json.dumps(epfo_data, indent=2)}")
            
            # Check for EPFO API error conditions or no records - no fallback
            response_status = epfo_data.get('status')
            response_msg = epfo_data.get('message', '').lower()
            
            if (response_status == 302 or response_status == 2 or 'source down' in response_msg or 
                'service unavailable' in response_msg or 'timeout' in response_msg or 
                'no record found' in response_msg or 'no records found' in response_msg):
                logger.warning(f"⚠️ EPFO API returned error/no records (status: {response_status}, msg: {epfo_data.get('message')})")
                logger.info("ℹ️ No fallback for EPFO - keeping original response")
                epfo_data['source'] = 'primary_api_failed'
                
        else:
            logger.error(f"❌ EPFO API Failed: {epfo_res.status_code} - {epfo_res.text}")
            logger.info("ℹ️ No fallback for EPFO - using empty response")
            epfo_data = {"status": "api_failed", "source": "primary_api_failed", "establishment_code": "", "message": f"API failed: {epfo_res.status_code}"}
            
    except Exception as e:
        logger.error(f"❌ EPFO API Error: {e}")
        logger.info("ℹ️ No fallback for EPFO - using empty response")
        epfo_data = {"status": "api_error", "source": "primary_api_failed", "establishment_code": "", "error": str(e)}
        
    return mca_data, epfo_data

def upsert_erdb(company_data: dict):
    """Step 4: Create Company in ERDB - Matches JavaScript logic exactly"""
    logger.info("Upserting to ERDB...")
    
    # 2a. Determine MCA Status using standardized mapping
    mca_status_raw = company_data.get('status', '')
    if not mca_status_raw or str(mca_status_raw).strip() == '' or str(mca_status_raw).upper() == 'UNKNOWN':
        mca_status = 'NOT_FOUND'
    else:
        # Map to standardized enum values
        raw_status_upper = str(mca_status_raw).upper()
        mca_status = MCA_STATUS_MAPPING.get(raw_status_upper, 'NOT_FOUND')
        logger.info(f"📊 MCA Status Mapping: '{mca_status_raw}' -> '{mca_status}'")
    
    # 2b. Determine EPFO Status and Payment Detail Status (matching JS logic)
    est_id = company_data.get('establishmentId', '')
    payment_details = company_data.get('paymentDetails', '')
    has_payment_details = payment_details and str(payment_details).strip() != ''
    
    if est_id and str(est_id).strip() != '':
        if has_payment_details:
            epfo_status = 'LISTED_WITH_PAYMENT_DETAILS'
            epfo_payment_status = "FOUND"
        else:
            epfo_status = 'LISTED_WITHOUT_PAYMENT_DETAILS'  
            epfo_payment_status = "NA"
    else:
        epfo_status = 'NOT_LISTED'
        epfo_payment_status = "NA"

    # 3. Construct the Full Request Body (matching JS structure)
    request_body = {
        "legalName": company_data.get('legalName'),
        "mca": {
            "status": mca_status,
            "dateOfIncorporation": format_date_yyyy_mm_dd(company_data.get('dateOfIncorporation')),
            "cin": company_data.get('cin') or None
        },
        "epfo": {
            "status": epfo_status,
            "paymentDetails": epfo_payment_status,
            "establishmentId": est_id or None
        },
        "verification": {"type": "UNKNOWN"},
        "discloseClientName": True,
        "requirements": [],
        "brandName": "",
        "name": "",
        "city": "",
        "state": "",
        "country": "",
        "website": "",
        "linkedInUrl": "",
        "gst": "",
        "additionalCharges": 0,
        "specialRemarks": "",
        "suspiciousRemarks": ""
    }
    
    logger.info(f"📤 Sending to ERDB: {json.dumps(request_body, indent=2)}")

    try:
        res = requests.put(
            "https://api-sa.in.springverify.com/internal/research/company",
            json=request_body,
            headers={"Content-Type": "application/json", "token": CONFIG["INTERNAL_TOKEN"]}
        )
        res.raise_for_status()
        return res.json()
    except Exception as e:
        logger.error(f"ERDB Upsert Failed: {e}")
        return None

def get_domain_llm(legal_name: str, email: str):
    """Step 7: Get Domain via Email Regex or LLM Search"""
    generic_domains = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
    
    # 1. Check Email
    if email:
        match = re.search(r"@([^\s@]+)", email)
        if match:
            domain = match.group(1).lower()
            if domain not in generic_domains:
                logger.info(f"Domain found from email: {domain}")
                return domain

    # 2. LLM Search via Portkey
    if not legal_name: return None
    
    logger.info(f"Asking LLM to find domain for: {legal_name}")
    prompt = f"""
    Find the website domain for company: {legal_name}.
    Return ONLY JSON: {{"domain_found": "example.com"}} or null.
    """
    
    payload = {
        "model": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [{"type": "function", "function": {"name": "google_search"}}]
    }
    
    headers = {
        "x-portkey-api-key": CONFIG["PORTKEY_KEY"],
        "x-portkey-virtual-key": CONFIG["PORTKEY_VIRTUAL"],
        "Content-Type": "application/json"
    }
    
    try:
        res = requests.post("https://api.portkey.ai/v1/chat/completions", json=payload, headers=headers)
        data = res.json()
        content = data["choices"][0]["message"]["content"]
        # Simple extraction logic (can be improved with json parsing)
        match = re.search(r'"domain_found"\s*:\s*"([^"]+)"', content)
        if match: return match.group(1)
    except Exception as e:
        logger.error(f"LLM Call Failed: {e}")
    
    return None

def search_apollo_global(target: str):
    """Search Apollo using mixed_people/search API (global search)"""
    if not target:
        return None
    
    logger.info(f"🌍 Apollo Global Search for: {target}")
    
    payload = {
        "api_key": CONFIG["APOLLO_KEY"],
        "page": 1,
        "person_titles": CONFIG["JOB_TITLES"],
        "person_locations": ["India"]
    }
    
    # Determine if target is domain or company name
    if '.' in target and ' ' not in target:
        payload["q_organization_domains_list"] = [target]
        logger.info(f"   🔍 Searching by domain: {target}")
    else:
        payload["q_organization_name"] = target
        logger.info(f"   🔍 Searching by company name: {target}")
    
    try:
        res = requests.post("https://api.apollo.io/v1/mixed_people/search", json=payload)
        if res.ok:
            data = res.json()
            logger.info(f"   ✅ Global search returned {len(data.get('contacts', []) + data.get('people', []))} contacts")
            return data
        else:
            logger.info(f"   ❌ Global search failed: {res.status_code}")
            return None
    except Exception as e:
        logger.error(f"   ❌ Global search error: {e}")
        return None

def search_apollo_internal(target: str):
    """Search Apollo using contacts/search API (internal contacts)"""
    if not target:
        return None
        
    logger.info(f"🏠 Apollo Internal Search for: {target}")
    
    payload = {
        "api_key": CONFIG["APOLLO_KEY"],
        "page": 1,
        "person_titles": CONFIG["JOB_TITLES"],
        "q_keywords": target
    }
    
    try:
        res = requests.post("https://api.apollo.io/v1/contacts/search", json=payload)
        if res.ok:
            data = res.json()
            logger.info(f"   ✅ Internal search returned {len(data.get('contacts', []))} contacts")
            return data
        else:
            logger.info(f"   ❌ Internal search failed: {res.status_code}")
            return None
    except Exception as e:
        logger.error(f"   ❌ Internal search error: {e}")
        return None

def process_contacts(domain: str, company_name: str, db_company_id: str, manual_email: str = None, manual_name: str = None):
    """Step 8: Apollo Search, Enrich, Hatch, Sequence - Updated to match JS logic"""
    if not domain and not manual_email: 
        logger.info("No domain or manual email for contact search. Skipping.")
        return []

    logger.info(f"🔍 Starting contact search for domain: {domain}")
    final_contacts = []
    locked_candidates = []
    MAX_CONTACTS = 4
    
    # Handle manual email from webhook (priority addition)
    if manual_email:
        logger.info(f"📧 Processing manual email from webhook: {manual_email}")
        
        # Parse name or use company name + HR as fallback
        if manual_name and manual_name.strip():
            name_parts = manual_name.strip().split(' ')
            first_name = name_parts[0] if name_parts else "HR"
            last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else "Manager"
        else:
            # Use company name as first name, HR as last name
            first_name = company_name or "HR"
            last_name = "HR"
        
        manual_contact = {
            "id": None,  # Manual contacts don't have Apollo IDs initially
            "first_name": first_name,
            "last_name": last_name,
            "title": "HR",
            "full_name": f"{first_name} {last_name}",
            "email": manual_email,
            "source": "manual_input"
        }
        
        final_contacts.append(manual_contact)
        logger.info(f"✅ Added manual contact: {manual_contact['full_name']} - {manual_email}")
    
    if not domain:
        logger.info("No domain provided, only processing manual email.")
        # Skip search but still execute sequence actions for manual contact
        if final_contacts:
            execute_sequence_actions(final_contacts, company_name, db_company_id)
        return final_contacts
    
    # 1. Try Global Search first, then Internal as fallback
    search_result = search_apollo_global(domain)
    if not search_result:
        logger.info("🔄 Global search failed, trying internal search...")
        search_result = search_apollo_internal(domain)
    
    if search_result and (search_result.get('contacts') or search_result.get('people')):
        all_people = (search_result.get('contacts', []) + search_result.get('people', []))
        logger.info(f"📊 Total contacts found: {len(all_people)}")
        
        # 2. Process initial contacts (unlocked vs locked)
        for i, person in enumerate(all_people):
            if len(final_contacts) + len(locked_candidates) >= MAX_CONTACTS:
                break
                
            # Extract name properly - handle both 'name' field and separate first/last
            if person.get('name'):
                name_parts = person['name'].strip().split(' ')
                first_name = name_parts[0] if name_parts else ""
                last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ""
                full_name = person['name']
            else:
                first_name = person.get('first_name', '')
                last_name = person.get('last_name', '')
                full_name = f"{first_name} {last_name}".strip()
            
            contact = {
                "id": person.get('id'),
                "first_name": first_name,
                "last_name": last_name,
                "title": person.get('title', ''),
                "full_name": full_name,
                "email": person.get('email'),
                "source": "pending"
            }
            
            # Check if email is already unlocked
            if contact['email'] and contact['email'] != "email_not_unlocked@domain.com" and "email_not_unlocked" not in contact['email']:
                # Check for duplicates (including manual email)
                if not any(f.get('email', '').lower() == contact['email'].lower() for f in final_contacts):
                    contact['source'] = "apollo_ready"
                    final_contacts.append(contact)
                    logger.info(f"✅ Contact {i+1}: {full_name} ({contact['title']}) - Email READY: {contact['email']}")
                else:
                    logger.info(f"⚠️ Contact {i+1}: {full_name} - Email already exists (duplicate): {contact['email']}")
            else:
                locked_candidates.append(contact)
                logger.info(f"🔒 Contact {i+1}: {full_name} ({contact['title']}) - Email LOCKED, will try to rescue")
        
        logger.info(f"📈 Initial Processing: {len(final_contacts)} ready, {len(locked_candidates)} locked")
    
    # 3. Rescue Locked Contacts
    if locked_candidates:
        logger.info(f"🔓 Starting rescue process for {len(locked_candidates)} locked contacts...")
        
        # 3a. Try Apollo Enrichment
        ids_to_enrich = [c['id'] for c in locked_candidates if c.get('id')]
        enriched_contacts = []
        
        if ids_to_enrich:
            logger.info(f"� Attempting Apollo enrichment for {len(ids_to_enrich)} contacts...")
            try:
                enrich_res = requests.post("https://api.apollo.io/api/v1/people/bulk_enrich", 
                    json={"api_key": CONFIG["APOLLO_KEY"], "details": [{"id": id} for id in ids_to_enrich]})
                if enrich_res.ok:
                    matches = enrich_res.json().get('matches', [])
                    enriched_contacts = matches
                    logger.info(f"📊 Enrichment returned {len(matches)} matches")
                else:
                    logger.error(f"❌ Enrichment failed: {enrich_res.status_code}")
            except Exception as e:
                logger.error(f"❌ Enrichment error: {e}")
        
        # 3b. Process each locked candidate
        rescue_count = 0
        hatch_count = 0
        
        for j, candidate in enumerate(locked_candidates):
            rescued = False
            
            # Try to find enriched match
            if candidate.get('id'):
                match = next((e for e in enriched_contacts if e.get('id') == candidate['id']), None)
                if match and match.get('email') and "email_not_unlocked" not in match.get('email'):
                    # Check for duplicates before adding
                    if not any(f.get('email', '').lower() == match['email'].lower() for f in final_contacts):
                        rescued_contact = candidate.copy()
                        rescued_contact['email'] = match['email']
                        rescued_contact['source'] = 'apollo_enriched'
                        final_contacts.append(rescued_contact)
                        rescue_count += 1
                        rescued = True
                        logger.info(f"✅ Enriched {j+1}: {candidate['full_name']} - Unlocked: {match['email']}")
            
            # If enrichment failed, try Hatch
            if not rescued and domain:
                logger.info(f"🎣 Trying Hatch for: {candidate['full_name']}")
                hatch_email = call_hatch_with_full_name(candidate['full_name'], domain)
                if hatch_email:
                    # Check for duplicates before adding
                    if not any(f.get('email', '').lower() == hatch_email.lower() for f in final_contacts):
                        rescued_contact = candidate.copy()
                        rescued_contact['email'] = hatch_email
                        rescued_contact['source'] = 'hatch'
                        rescued_contact['id'] = None  # Hatch contacts don't have Apollo IDs
                        final_contacts.append(rescued_contact)
                        hatch_count += 1
                        logger.info(f"✅ Hatch Success: {candidate['full_name']} - Found: {hatch_email}")
                    else:
                        logger.info(f"⚠️ Hatch found duplicate email for: {candidate['full_name']}")
                else:
                    logger.info(f"❌ Hatch Failed: {candidate['full_name']} - No email found")
        
        logger.info(f"🔓 Rescue Results: {rescue_count} enriched, {hatch_count} via Hatch")
    
    # Final Summary
    logger.info(f"📊 FINAL CONTACT SEARCH SUMMARY:")
    logger.info(f"   🔍 Total found by search: {len(all_people) if 'all_people' in locals() else 0}")
    logger.info(f"   ✅ Ready contacts: {len([c for c in final_contacts if c['source'] == 'apollo_ready'])}")
    logger.info(f"   🔒 Initially locked: {len(locked_candidates)}")
    logger.info(f"   🔓 Enriched contacts: {len([c for c in final_contacts if c['source'] == 'apollo_enriched'])}")
    logger.info(f"   🎣 Hatch contacts: {len([c for c in final_contacts if c['source'] == 'hatch'])}")
    logger.info(f"   🎯 FINAL TOTAL: {len(final_contacts)} contacts ready for sequence")

    # 5. Add to Sequence & ERDB
    if final_contacts:
        execute_sequence_actions(final_contacts, company_name, db_company_id)
    else:
        logger.info("⚠️ No contacts found - skipping sequence actions")
    
    # Return contacts for CSV logging
    return final_contacts

def call_hatch(fname, lname, domain):
    """Legacy function - kept for backward compatibility"""
    try:
        res = requests.post("https://api.hatchhq.ai/v1/findEmail", 
            headers={"x-api-key": CONFIG["HATCH_KEY"]}, 
            json={"firstName": fname, "lastName": lname, "domain": domain})
        return res.json().get('email')
    except: return None



def call_hatch_with_full_name(full_name, domain):
    """Updated Hatch function that matches JS logic - uses full name"""
    if not full_name or not full_name.strip():
        return None
    
    try:
        name_parts = full_name.strip().split(' ')
        first_name = name_parts[0] if name_parts else ""
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ""
        
        # Prepare request payload
        payload = {
            "firstName": first_name,
            "lastName": last_name, 
            "domain": domain
        }
        
        headers = {
            "x-api-key": CONFIG["HATCH_KEY"],
            "Content-Type": "application/json"
        }
        
        logger.info(f"   🎣 Hatch API Request:")
        logger.info(f"      URL: https://api.hatchhq.ai/v1/findEmail")
        logger.info(f"      Headers: {json.dumps({k: v if k != 'x-api-key' else f'{v[:10]}...' for k, v in headers.items()})}")
        logger.info(f"      Payload: {json.dumps(payload)}")
        
        res = requests.post("https://api.hatchhq.ai/v1/findEmail", 
            headers=headers, 
            json=payload)
        
        logger.info(f"   📊 Hatch Response Status: {res.status_code}")
        logger.info(f"   📋 Hatch Response Headers: {dict(res.headers)}")
        
        try:
            response_data = res.json()
            logger.info(f"   📄 Hatch Response Body: {json.dumps(response_data, indent=2)}")
        except:
            logger.info(f"   📄 Hatch Response Text: {res.text}")
        
        if res.ok:
            data = res.json()
            email = data.get('email')
            if email and isinstance(email, str) and '@' in email:
                logger.info(f"   ✅ Hatch found valid email: {email}")
                return email
            else:
                logger.info(f"   ❌ Hatch returned invalid/no email: {email}")
                return None
        else:
            logger.error(f"   ❌ Hatch API failed with status {res.status_code}: {res.text}")
            return None
            
    except Exception as e:
        logger.error(f"   ❌ Hatch error: {e}")
        return None

def execute_sequence_actions(contacts, company_name, db_company_id):
    logger.info(f"🚀 STARTING SEQUENCE ACTIONS for {len(contacts)} contacts...")
    contact_ids = []
    
    # Create contacts in Apollo
    logger.info("📝 Step 1: Creating contacts in Apollo...")
    for i, c in enumerate(contacts):
        try:
            contact_payload = {
                "api_key": CONFIG["APOLLO_KEY"],
                "first_name": c['first_name'], 
                "last_name": c['last_name'], 
                "email": c['email'], 
                "organization_name": company_name
            }
            
            res = requests.post("https://api.apollo.io/v1/contacts", json=contact_payload)
            cid = res.json().get('contact', {}).get('id')
            
            if cid: 
                contact_ids.append(cid)
                logger.info(f"✅ Created contact {i+1}: {c['first_name']} {c['last_name']} - ID: {cid}")
            else:
                logger.info(f"❌ Failed to create contact {i+1}: {c['first_name']} {c['last_name']} - No ID returned")
                
        except Exception as e:
            logger.error(f"❌ Error creating contact {i+1}: {c['first_name']} {c['last_name']} - {e}")

    logger.info(f"📊 Apollo Contact Creation: {len(contact_ids)} out of {len(contacts)} successfully created")

    # Add to Campaign
    if contact_ids:
        logger.info(f"🎯 Step 2: Adding {len(contact_ids)} contacts to campaign {CONFIG['CAMPAIGN_ID']}...")
        try:
            campaign_payload = {
                "api_key": CONFIG["APOLLO_KEY"], 
                "contact_ids": contact_ids, 
                "emailer_campaign_id": CONFIG['CAMPAIGN_ID'], 
                "send_email_from_email_account_id": CONFIG['EMAIL_SENDER_ID']
            }
            
            campaign_res = requests.post(f"https://api.apollo.io/api/v1/emailer_campaigns/{CONFIG['CAMPAIGN_ID']}/add_contact_ids", json=campaign_payload)
            
            if campaign_res.ok:
                logger.info(f"✅ Successfully added {len(contact_ids)} contacts to Apollo campaign")
            else:
                logger.error(f"❌ Failed to add contacts to campaign: {campaign_res.status_code} - {campaign_res.text}")
                
        except Exception as e:
            logger.error(f"❌ Error adding contacts to campaign: {e}")
    else:
        logger.info("⚠️ No contacts to add to campaign")

    # Update Internal Database
    if db_company_id:
        logger.info(f"💾 Step 3: Adding {len(contacts)} contacts to internal database (Company ID: {db_company_id})...")
        try:
            db_contacts = []
            for c in contacts:
                db_contact = {
                    "name": f"{c['first_name']} {c['last_name']}", 
                    "designation": c['title'], 
                    "email": {"type": "INDIVIDUAL", "address": c['email']}, 
                    "active": False
                }
                db_contacts.append(db_contact)
            
            db_payload = {"companyId": db_company_id, "contacts": db_contacts}
            
            db_res = requests.put("https://api-sa.in.springverify.com/internal/research/contacts", 
                headers={"token": CONFIG["INTERNAL_TOKEN"]}, json=db_payload)
            
            if db_res.ok:
                logger.info(f"✅ Successfully added {len(contacts)} contacts to internal database")
            else:
                logger.error(f"❌ Failed to add contacts to internal DB: {db_res.status_code} - {db_res.text}")
                
        except Exception as e:
            logger.error(f"❌ Error adding contacts to internal database: {e}")
    else:
        logger.info("⚠️ No company ID provided - skipping internal database update")
    
    logger.info("🎉 SEQUENCE ACTIONS COMPLETED!")


# --- CSV LOGGING FUNCTIONS ---

def ensure_csv_exists():
    """Ensure CSV file exists with proper headers"""
    csv_file_path = '/Users/suchitsharma/Documents/GitHub/Apollo-zapier/automation_data.csv'
    
    # Check if file exists
    if not os.path.exists(csv_file_path):
        # Create CSV with headers
        headers = [
            'Timestamp',
            'Company_Name',
            'CIN_Number',
            'Date_Of_Incorporation', 
            'Company_Status',
            'Establishment_Code',
            'Input_Name',
            'Input_Email',
            'Email_1',
            'Person_1',
            'Source_1',
            'Email_2', 
            'Person_2',
            'Source_2',
            'Email_3',
            'Person_3', 
            'Source_3',
            'Email_4',
            'Person_4',
            'Source_4'
        ]
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        logger.info(f"📊 Created new CSV file: {csv_file_path}")
    
    return csv_file_path

def save_to_csv(company_data: dict, contacts: list, input_name: str, input_email: str):
    """Save all automation data to CSV file"""
    try:
        csv_file_path = ensure_csv_exists()
        
        # Prepare the row data
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Company data
        company_name = company_data.get('legalName', '')
        cin_number = company_data.get('cin', '')
        date_of_incorporation = company_data.get('dateOfIncorporation', '')
        company_status = company_data.get('status', '')
        establishment_code = company_data.get('establishmentId', '')
        
        # Contact data - prepare up to 4 contacts (excluding manual input contact)
        # Filter out manual_input contacts to avoid duplication with Input_Name/Input_Email
        discovered_contacts = [c for c in contacts if c.get('source') != 'manual_input']
        
        contact_data = ['', '', '', '', '', '', '', '', '', '', '', '']  # Email_1, Person_1, Source_1, Email_2, Person_2, Source_2, Email_3, Person_3, Source_3, Email_4, Person_4, Source_4
        
        for i, contact in enumerate(discovered_contacts[:4]):  # Limit to 4 discovered contacts
            base_index = i * 3  # Each contact takes 3 columns (email, person, source)
            if base_index < len(contact_data) - 2:  # Ensure we don't exceed array bounds
                contact_data[base_index] = contact.get('email', '')  # Email
                contact_data[base_index + 1] = contact.get('full_name', '')  # Person name
                contact_data[base_index + 2] = contact.get('source', '')  # Source
        
        # Construct the complete row
        row_data = [
            timestamp,
            company_name,
            cin_number,
            date_of_incorporation,
            company_status, 
            establishment_code,
            input_name or '',
            input_email or '',
        ] + contact_data
        
        # Append to CSV
        with open(csv_file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row_data)
        
        logger.info(f"📊 ✅ Data saved to CSV: {len(discovered_contacts)} discovered contacts + input contact for {company_name}")
        logger.info(f"📁 CSV Location: {csv_file_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to save to CSV: {e}")

# --- PARALLEL API FALLBACK FUNCTIONS ---

def call_parallel_mca_fallback(company_name: str):
    """Parallel API fallback for MCA data - Gets CIN, Date of Incorporation, Company Status"""
    logger.info(f"🔄 Calling Parallel API fallback for MCA: {company_name}")
    
    try:
        # Import Parallel SDK
        try:
            from parallel import Parallel
            logger.info("✅ Parallel SDK imported successfully")
        except ImportError as import_error:
            logger.error(f"❌ Parallel SDK not installed: {import_error}")
            logger.error("   Install with: pip install parallel")
            return {
                "cin": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "sdk_not_available",
                "source": "parallel_fallback",
                "error": "Parallel SDK not installed"
            }
        
        # Parallel API configuration
        api_key = CONFIG.get("PARALLEL_API_KEY") or "dZE6nnwRdoHBP0YB4CAfwoy8L-wRrxchNlekN0Q0"
        
        logger.info(f"📤 Parallel API Request Configuration:")
        logger.info(f"   Company Name: {company_name}")
        logger.info(f"   API Key: {api_key[:20]}... (length: {len(api_key)})")
        
        # Initialize Parallel client
        logger.info("🔧 Initializing Parallel client...")
        try:
            client = Parallel(api_key=api_key)
            logger.info("✅ Parallel client initialized successfully")
            logger.info(f"   Client Type: {type(client)}")
            logger.info(f"   Client Attributes: {[attr for attr in dir(client) if not attr.startswith('_')]}")
        except Exception as client_error:
            logger.error(f"❌ Failed to initialize Parallel client: {client_error}")
            logger.error(f"   Error Type: {type(client_error).__name__}")
            logger.error(f"   Error Details: {repr(client_error)}")
            return {
                "cin": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "client_init_error",
                "source": "parallel_fallback",
                "error": f"Client initialization failed: {str(client_error)}"
            }
        
        # Use exact same parameters that worked in test
        objective = f'Find company information for {company_name}'
        search_queries = [company_name]  # Simple company name only, no "CIN" suffix
        source_domains = ["zaubacorp.com"]  # Only zaubacorp, cleartax seems to cause issues
        
        logger.info(f"📋 Parallel API Search Parameters:")
        logger.info(f"   Mode: one-shot")
        logger.info(f"   Objective: {objective}")
        logger.info(f"   Search Queries: {search_queries}")
        logger.info(f"   Max Results: 3")
        logger.info(f"   Source Domains: {source_domains}")
        
        # Make the API call using the correct SDK syntax
        logger.info("🚀 Making Parallel API call...")
        try:
            # Check if client has beta attribute
            if hasattr(client, 'beta'):
                logger.info("✅ Client has 'beta' attribute")
                if hasattr(client.beta, 'search'):
                    logger.info("✅ Client.beta has 'search' method")
                else:
                    logger.error("❌ Client.beta does not have 'search' method")
                    logger.info(f"   Available methods: {[attr for attr in dir(client.beta) if not attr.startswith('_')]}")
            else:
                logger.error("❌ Client does not have 'beta' attribute")
                logger.info(f"   Available attributes: {[attr for attr in dir(client) if not attr.startswith('_')]}")
            
            response = client.beta.search(
                mode="one-shot",
                search_queries=search_queries,
                max_results=3,  # Use same as test
                objective=objective,
                source_policy={"include_domains": source_domains}
                # Remove max_chars_per_result as it's deprecated per the warning
            )
            logger.info("✅ Parallel API call completed successfully")
        except Exception as api_error:
            logger.error(f"❌ Parallel API call failed: {api_error}")
            logger.error(f"   Error Type: {type(api_error).__name__}")
            logger.error(f"   Error Details: {repr(api_error)}")
            return {
                "cin": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "api_call_error",
                "source": "parallel_fallback",
                "error": f"API call failed: {str(api_error)}"
            }
        
        logger.info(f"✅ Parallel API Response received successfully")
        logger.info(f"� Response Type: {type(response)}")
        logger.info(f"📄 Response Structure: {type(response)}")
        logger.info(f"📋 Full Raw Response: {str(response)}")
        
        # Extract data from Parallel API response
        logger.info("🔍 Extracting company data from Parallel API response...")
        try:
            extracted_data = extract_company_data_from_parallel_response(response, company_name)
            extracted_data["source"] = "parallel_fallback"
            logger.info("✅ Data extraction completed successfully")
        except Exception as extraction_error:
            logger.error(f"❌ Data extraction failed: {extraction_error}")
            logger.error(f"   Error Type: {type(extraction_error).__name__}")
            logger.error(f"   Error Details: {repr(extraction_error)}")
            return {
                "cin": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "extraction_error",
                "source": "parallel_fallback",
                "error": f"Data extraction failed: {str(extraction_error)}"
            }
        
        logger.info(f"✅ Parallel API Data Extraction Complete")
        logger.info(f"📊 Extracted Data Summary:")
        logger.info(f"   CIN: {'✅ Found' if extracted_data.get('cin') else '❌ Missing'} - '{extracted_data.get('cin', 'N/A')}'")
        logger.info(f"   Date: {'✅ Found' if extracted_data.get('date_of_incorporation') else '❌ Missing'} - '{extracted_data.get('date_of_incorporation', 'N/A')}'")
        logger.info(f"   Status: {'✅ Found' if extracted_data.get('company_status') else '❌ Missing'} - '{extracted_data.get('company_status', 'N/A')}'")
        logger.info(f"   Overall Status: {extracted_data.get('status', 'N/A')}")
        logger.info(f"📋 Final Parallel API Result: {json.dumps(extracted_data, indent=2)}")
        
        return extracted_data
        
    except Exception as e:
        logger.error(f"❌ Parallel API Fallback Critical Error: {str(e)}")
        logger.error(f"   Error Type: {type(e).__name__}")
        logger.error(f"   Error Details: {repr(e)}")
        
        # Return error response instead of falling back to mock
        return {
            "cin": "",
            "date_of_incorporation": "",
            "company_status": "",
            "status": "api_error",
            "source": "parallel_fallback",
            "error": str(e)
        }



def extract_company_data_from_parallel_response(response, company_name: str) -> dict:
    """Extract CIN, Date of Incorporation, and Company Status from Parallel API SearchResult response"""
    logger.info(f"🔍 Starting data extraction for company: {company_name}")
    
    extracted = {
        "cin": "",
        "date_of_incorporation": "",
        "company_status": "",
        "status": "not_found"
    }
    
    logger.info(f"📊 Response structure analysis:")
    logger.info(f"   Response type: {type(response)}")
    logger.info(f"   Response attributes: {[attr for attr in dir(response) if not attr.startswith('_')][:10]}")  # Show first 10 attributes
    
    # Handle Parallel API SearchResult object
    results = []
    if hasattr(response, 'results'):
        results = response.results
        logger.info(f"✅ Found 'results' attribute with {len(results)} items")
    else:
        logger.error(f"❌ No 'results' attribute found in response")
        return extracted
    
    if results:
        logger.info(f"📋 Processing {len(results)} search results...")
        
        for i, result in enumerate(results):
            logger.info(f"🔍 Processing result {i+1}:")
            logger.info(f"   Result type: {type(result)}")
            logger.info(f"   Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")
            
            # Extract excerpts from WebSearchResult object
            excerpts = []
            if hasattr(result, 'excerpts'):
                excerpts = result.excerpts
                logger.info(f"✅ Found {len(excerpts)} excerpts")
            
            # Also log URL and title for context
            url_text = ""
            title_text = ""
            if hasattr(result, 'url'):
                url_text = result.url
                logger.info(f"   URL: {url_text}")
            if hasattr(result, 'title'):
                title_text = result.title
                logger.info(f"   Title: {title_text}")
            
            # FIRST: Try to extract CIN from URL (most reliable source)
            if url_text and not extracted["cin"]:
                logger.info(f"   🔗 Extracting CIN from URL: {url_text}")
                # Enhanced CIN pattern for URLs - more comprehensive
                cin_url_patterns = [
                    r'-([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})(?:/|$)',  # CIN with dash prefix (most common)
                    r'/([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})(?:/|$)',  # Standard CIN in URL path
                    r'CIN-([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',  # CIN- prefix
                    r'cin/([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',  # cin/ prefix
                    r'([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})'  # CIN anywhere in URL (fallback)
                ]
                for pattern in cin_url_patterns:
                    cin_match = re.search(pattern, url_text.upper())
                    if cin_match:
                        extracted["cin"] = cin_match.group(1)
                        logger.info(f"   🎯 CIN Extracted from URL: {extracted['cin']}")
                        break
            
            # Process each excerpt for remaining data
            for j, excerpt in enumerate(excerpts):
                logger.info(f"   📄 Processing excerpt {j+1}:")
                logger.info(f"   Excerpt length: {len(excerpt)}")
                logger.info(f"   Excerpt preview: {excerpt[:300]}{'...' if len(excerpt) > 300 else ''}")
                
                if isinstance(excerpt, str):
                    excerpt_upper = excerpt.upper()
                    
                    # Extract CIN from excerpt if not found in URL
                    if not extracted["cin"]:
                        cin_patterns = [
                            r'CIN[:\s]*([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',  # CIN: format
                            r'CIN\s+NUMBER[:\s]*([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})',  # CIN Number: format
                            r'([A-Z]\d{5}[A-Z]{2}\d{4}[A-Z]{3}\d{6})'  # Just the CIN pattern
                        ]
                        logger.info(f"   🔎 Searching for CIN in excerpt...")
                        for pattern in cin_patterns:
                            cin_match = re.search(pattern, excerpt_upper)
                            if cin_match:
                                extracted["cin"] = cin_match.group(1)
                                logger.info(f"   🎯 CIN Found in Excerpt: {extracted['cin']}")
                                break
                    
                    # Extract incorporation date - enhanced patterns
                    if not extracted["date_of_incorporation"]:
                        date_patterns = [
                            r'INCORPORATED ON\s+(\d{1,2}[-\s/]\w{3}[-\s/]\d{4})',  # "incorporated on 19-Dec-2015"
                            r'DATE OF INCORPORATION[:\s]+(\d{1,2}[-\s/]\w{3}[-\s/]\d{4})',  # "Date of Incorporation: 19-Dec-2015"
                            r'INCORPORATION DATE[:\s]+(\d{1,2}[-\s/]\w{3}[-\s/]\d{4})',  # "Incorporation Date: 19-Dec-2015"
                            r'(\d{1,2}[-\s/]\w{3}[-\s/]\d{4})',  # General DD-Mon-YYYY format
                            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD format
                            r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})'  # DD/MM/YYYY format
                        ]
                        logger.info(f"   🔎 Searching for incorporation date...")
                        for k, pattern in enumerate(date_patterns):
                            date_match = re.search(pattern, excerpt_upper)
                            if date_match:
                                extracted["date_of_incorporation"] = date_match.group(1)
                                logger.info(f"   🎯 Date Found (pattern {k+1}): {extracted['date_of_incorporation']}")
                                break
                    
                    # Extract company status - enhanced patterns
                    if not extracted["company_status"]:
                        # Enhanced status patterns to capture the actual status value
                        status_patterns = [
                            r'CURRENT STATUS OF [^-]+ IS - ([A-Z\s]+)\.',  # "Current status of ... is - Active."
                            r'COMPANY STATUS[:\s-]+([A-Z\s]+)\.',  # "Company Status: Active."
                            r'STATUS[:\s-]+([A-Z\s]+)\.',  # "Status: Active."
                            r'CURRENT STATUS[:\s-]+([A-Z\s]+)\.',  # "Current Status: Active."
                            r'STATUS OF [^-]+ IS - ([A-Z\s]+)\.'  # "Status of ... is - Active."
                        ]
                        
                        status_found = False
                        for i, pattern in enumerate(status_patterns):
                            status_match = re.search(pattern, excerpt_upper)
                            if status_match:
                                status_text = status_match.group(1).strip()
                                # Map found status to standardized values
                                if 'ACTIVE' in status_text:
                                    extracted["company_status"] = 'ACTIVE'
                                elif 'STRIKE OFF' in status_text:
                                    extracted["company_status"] = 'STRIKE_OFF'
                                elif 'LIQUIDATION' in status_text:
                                    extracted["company_status"] = 'UNDER_LIQUIDATION'
                                elif 'AMALGAMATED' in status_text:
                                    extracted["company_status"] = 'AMALGAMATED'
                                elif 'CONVERTED' in status_text:
                                    extracted["company_status"] = 'CONVERTED'
                                else:
                                    extracted["company_status"] = status_text
                                logger.info(f"   🎯 Status Found (pattern {i+1}): {extracted['company_status']} (from: {status_text})")
                                status_found = True
                                break
                        
                        # Fallback: look for status keywords in context
                        if not status_found:
                            status_keywords = ['ACTIVE', 'STRIKE OFF', 'LIQUIDATION', 'AMALGAMATED', 'CONVERTED']
                            logger.info(f"   🔎 Searching for status keywords...")
                            for status in status_keywords:
                                if status in excerpt_upper:
                                    # Map to standardized status
                                    if status == 'STRIKE OFF':
                                        extracted["company_status"] = 'STRIKE_OFF'
                                    elif status == 'LIQUIDATION':
                                        extracted["company_status"] = 'UNDER_LIQUIDATION'
                                    elif status in ['NON-GOVERNMENT COMPANY', 'PRIVATE COMPANY']:
                                        extracted["company_status"] = 'ACTIVE'  # These are typically active companies
                                    else:
                                        extracted["company_status"] = status
                                    logger.info(f"   🎯 Status Found (keyword): {extracted['company_status']}")
                                    break
                    
                    # If we found any data in this excerpt, mark as success
                    if extracted["cin"] or extracted["date_of_incorporation"] or extracted["company_status"]:
                        extracted["status"] = "success"
                        logger.info(f"   ✅ Found useful data in excerpt {j+1} - marking as success")
                        # Continue processing to potentially find more data
                    else:
                        logger.info(f"   ⚠️ No useful data found in excerpt {j+1}")
                else:
                    logger.info(f"   ❌ Excerpt is not a string: {type(excerpt)}")
            
            # If we found complete data, we can break out of results loop
            if extracted["cin"] and extracted["date_of_incorporation"] and extracted["company_status"]:
                logger.info(f"   ✅ Complete data found - breaking out of results loop")
                break
    else:
        logger.info(f"❌ No results found in response")
    
    logger.info(f"📊 Extraction complete - Final status: {extracted['status']}")
    logger.info(f"📋 Final extracted data:")
    logger.info(f"   CIN: {'✅' if extracted['cin'] else '❌'} {extracted['cin']}")
    logger.info(f"   Date: {'✅' if extracted['date_of_incorporation'] else '❌'} {extracted['date_of_incorporation']}")
    logger.info(f"   Status: {'✅' if extracted['company_status'] else '❌'} {extracted['company_status']}")
    
    return extracted

# --- GEMINI FALLBACK FUNCTIONS (LEGACY) ---

def call_gemini_combined_fallback(company_name: str):
    """Single Gemini fallback for both MCA and EPFO APIs - Uses Portkey Prompt"""
    logger.info(f"🤖 Calling Gemini combined fallback for: {company_name}")
    
    # Use your proven Portkey prompt approach
    PROMPT_ID = "pp-cin-epfo-3813ba"
    PORTKEY_URL = f"https://api.portkey.ai/v1/prompts/{PROMPT_ID}/completions"
    
    try:
        payload = {
            "promptID": PROMPT_ID,
            "variables": {
                "company_name": company_name or ""
            }
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-portkey-api-key": CONFIG["PORTKEY_KEY"],
            "x-portkey-virtual-key": CONFIG["PORTKEY_VIRTUAL"]
        }
        
        logger.info(f"📤 Gemini Combined Portkey Request:")
        logger.info(f"   URL: {PORTKEY_URL}")
        logger.info(f"   Payload: {json.dumps(payload, indent=2)}")
        logger.info(f"   Headers: {json.dumps({k: v if 'key' not in k.lower() else f'{v[:10]}...' for k, v in headers.items()})}")
        
        response = requests.post(PORTKEY_URL, headers=headers, json=payload, timeout=60)
        
        logger.info(f"📊 Gemini Combined Response Status: {response.status_code}")
        logger.info(f"📥 Gemini Combined Response Headers: {dict(response.headers)}")
        
        # Check for HTTP errors
        if response.status_code != 200:
            logger.error(f"❌ Portkey Combined error: Non-200 response ({response.status_code}): {response.text}")
            return {
                "cin": "",
                "establishment_code": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "api_error",
                "source": "gemini_fallback",
                "error": f"HTTP {response.status_code}: {response.text}"
            }
        
        # Try parsing JSON response
        try:
            result = response.json()
            logger.info(f"📄 Gemini Combined Portkey Response: {json.dumps(result, indent=2)}")
            
            # Extract content from Portkey response structure
            if result and isinstance(result, dict) and result.get('choices'):
                content = result['choices'][0]['message']['content']
                logger.info(f"📄 Gemini Content: {content}")
                
                # Extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*\n?(.*?)\n?```', content, re.DOTALL)
                if not json_match:
                    # Try without markdown wrapper
                    json_match = re.search(r'\{.*\}', content, re.DOTALL)
                
                if json_match:
                    try:
                        gemini_data = json.loads(json_match.group(1) if json_match.groups() else json_match.group())
                        logger.info(f"📄 Parsed Gemini JSON: {json.dumps(gemini_data, indent=2)}")
                        
                        # Add source tracking
                        gemini_data["source"] = "gemini_fallback"
                        
                        # Determine status based on whether we have useful data
                        has_cin = gemini_data.get('cin', '') and gemini_data.get('cin', '').strip()
                        has_establishment = gemini_data.get('establishment_code', '') and gemini_data.get('establishment_code', '').upper() != 'N/A'
                        gemini_data["status"] = "success" if (has_cin or has_establishment) else "not_found"
                        
                        logger.info(f"✅ Gemini Combined Final Result: {json.dumps(gemini_data, indent=2)}")
                        return gemini_data
                        
                    except json.JSONDecodeError as je2:
                        logger.error(f"❌ Failed to parse extracted JSON: {str(je2)}")
                        return {"cin": "", "establishment_code": "", "date_of_incorporation": "", "company_status": "", "status": "json_parse_error", "source": "gemini_fallback"}
                else:
                    logger.error("❌ No JSON found in Gemini content")
                    return {"cin": "", "establishment_code": "", "date_of_incorporation": "", "company_status": "", "status": "no_json_found", "source": "gemini_fallback"}
            else:
                logger.error("❌ Unexpected Portkey Combined response format")
                return {"cin": "", "establishment_code": "", "date_of_incorporation": "", "company_status": "", "status": "format_error", "source": "gemini_fallback"}
                
        except json.JSONDecodeError as je:
            logger.error(f"❌ Portkey Combined returned non-JSON response: {response.text[:200]}")
            return {
                "cin": "",
                "establishment_code": "",
                "date_of_incorporation": "",
                "company_status": "",
                "status": "parse_error", 
                "source": "gemini_fallback",
                "error": f"JSON decode error: {str(je)}"
            }
        
    except Exception as e:
        logger.error(f"❌ Gemini Combined Fallback Error: {str(e)}")
        return {
            "cin": "",
            "establishment_code": "",
            "date_of_incorporation": "",
            "company_status": "",
            "status": "exception",
            "source": "gemini_fallback",
            "error": str(e)
        }


# Legacy functions kept for backward compatibility
def call_gemini_mca_fallback(company_name: str):
    """Legacy MCA fallback - now calls combined function"""
    return call_gemini_combined_fallback(company_name)

def call_gemini_epfo_fallback(company_name: str):
    """Legacy EPFO fallback - now calls combined function"""
    return call_gemini_combined_fallback(company_name)


# --- MAIN LOGIC ORCHESTRATOR ---

def run_full_automation(data: WebhookPayload):
    """This runs in the background after webhook receipt"""
    logger.info("--- STARTING AUTOMATION ---")
    
    # Step 1: Parse Inputs
    raw_name = data.company_name or data.company or ""
    email = data.email or ""
    
    # Step 2: Clean Name
    cleaned_name = clean_company_name(raw_name)
    logger.info(f"Cleaned Name: {cleaned_name}")
    
    # Step 3: Call MCA/EPFO
    # Note: In your zapier flow you pulled specific fields from MCA/EPFO results 
    # Here is a simplified merge. You might need to adjust field names based on exact API response
    mca, epfo = call_external_apis(cleaned_name)
    
    # Prepare Data for ERDB - FIXED DATA EXTRACTION
    # Extract data from nested API responses properly
    establishment_code = ""
    payment_details_data = ""
    cin_number = ""
    date_of_incorporation = ""
    mca_status = ""
    
    # Extract MCA data - prioritize primary API, use Parallel API as fallback
    mca_source = mca.get('source', 'primary_api')
    
    # Try to extract from primary MCA API first
    if mca_source == 'primary_api' and mca.get('result') and isinstance(mca['result'], dict):
        mca_result = mca['result']
        cin_number = mca_result.get('cin', '')
        date_of_incorporation = mca_result.get('date_of_incorporation', '')
        
        # Handle different response formats
        if not date_of_incorporation:
            date_of_incorporation = mca_result.get('dateOfIncorporation', '')
        
        # For MCA status, use success status if we have result data
        mca_status = "ACTIVE" if cin_number else mca.get('status', '')
        
        logger.info(f"📋 Extracted MCA (primary_api) - CIN: {cin_number}, DOI: {date_of_incorporation}, Status: {mca_status}")
        
    # If primary MCA failed or has no data, try to use Parallel API data
    elif mca_source == 'parallel_fallback':
        # Parallel API data is directly in the response, not nested under 'result'
        cin_number = mca.get('cin', '')
        date_of_incorporation = mca.get('date_of_incorporation', '')
        
        # Handle company_status field from Parallel API
        parallel_status = mca.get('company_status', '')
        mca_status = parallel_status if parallel_status else ('ACTIVE' if cin_number else 'NOT_FOUND')
        logger.info(f"📋 Extracted MCA (parallel_fallback) - CIN: {cin_number}, DOI: {date_of_incorporation}, Status: {mca_status}")
        
    # Legacy support for Gemini fallback (if still used elsewhere)
    elif mca_source == 'gemini_fallback':
        # Gemini data is directly in the response, not nested under 'result'
        cin_number = mca.get('cin', '')
        date_of_incorporation = mca.get('date_of_incorporation', '')
        if not date_of_incorporation:
            date_of_incorporation = mca.get('dateOfIncorporation', '')
        
        # Handle company_status field from Gemini
        gemini_status = mca.get('company_status', '')
        mca_status = gemini_status if gemini_status else ('ACTIVE' if cin_number else 'NOT_FOUND')
        logger.info(f"📋 Extracted MCA (gemini_fallback) - CIN: {cin_number}, DOI: {date_of_incorporation}, Status: {mca_status}")
        
    else:
        # No valid MCA data from either source
        mca_status = mca.get('status', '')
        logger.info(f"📋 MCA ({mca_source}) has no result data, status: {mca_status}")
    
    # Extract EPFO data - prioritize primary API, use Gemini as fallback
    epfo_source = epfo.get('source', 'primary_api')
    
    # Try to extract from primary EPFO API first
    if epfo_source == 'primary_api' and epfo.get('result') and isinstance(epfo['result'], dict):
        epfo_result = epfo['result']
        
        # Extract establishment_code from establishment_details or additional_information
        if epfo_result.get('establishment_details'):
            establishment_code = epfo_result['establishment_details'].get('establishment_code', '')
        elif epfo_result.get('additional_information'):
            establishment_code = epfo_result['additional_information'].get('establishment_id', '')
        else:
            establishment_code = epfo_result.get('establishment_code', '')
            
        # Extract CIN from additional_information if available (and MCA didn't provide it)
        if not cin_number and epfo_result.get('additional_information'):
            epfo_cin = epfo_result['additional_information'].get('cin_code', '')
            if epfo_cin:
                cin_number = epfo_cin
                logger.info(f"🏢 Extracted CIN from EPFO: {cin_number}")
                
        # Extract date of setup from establishment_details if available (and MCA didn't provide it)
        if not date_of_incorporation and epfo_result.get('establishment_details'):
            epfo_date = epfo_result['establishment_details'].get('date_of_setup', '')
            if epfo_date:
                date_of_incorporation = epfo_date
                logger.info(f"📅 Extracted Date of Setup from EPFO: {date_of_incorporation}")
            
        # Extract payment details if available
        if epfo_result.get('payment_details'):
            payment_details_data = json.dumps(epfo_result['payment_details'])
            
        logger.info(f"🏢 Extracted EPFO (primary_api) - Code: {establishment_code}")
        logger.info(f"💰 Payment details available: {'Yes' if payment_details_data else 'No'}")
        
    else:
        # EPFO API failed or returned no data - no fallback available
        logger.info(f"� EPFO ({epfo_source}) has no result data, status: {epfo.get('status', '')}")
        logger.info("ℹ️ No EPFO fallback - establishment code will be empty")
    
    # Final status determination - if we have CIN from any source, company is found
    final_mca_status = mca_status
    if cin_number and (mca_status == 'NOT_FOUND' or not mca_status):
        final_mca_status = 'ACTIVE'  # Default to ACTIVE if we have CIN but no specific status
        logger.info(f"📊 Updated MCA status to ACTIVE (CIN found from EPFO)")
    
    final_data = {
        "legalName": cleaned_name,
        "cin": cin_number,
        "dateOfIncorporation": date_of_incorporation,
        "status": final_mca_status,
        "establishmentId": establishment_code,
        "paymentDetails": payment_details_data
    }
    
    logger.info(f"📋 Prepared ERDB Data: {json.dumps(final_data, indent=2)}")
    
    # Step 4: Upsert ERDB
    erdb_response = upsert_erdb(final_data)
    
    # Step 6: Extract ID
    db_company_id = None
    if erdb_response:
        db_company_id = extract_company_id(erdb_response)
        logger.info(f"✅ Extracted company ID from ERDB: {db_company_id}")
    else:
        logger.error("❌ Failed to get ERDB response - no company ID available")
        db_company_id = None

    # Step 7: Find Domain
    domain = get_domain_llm(cleaned_name, email)
    
    # Step 8: Contacts & Sequence
    # Pass manual email and name from webhook data
    final_contacts = process_contacts(domain, cleaned_name, db_company_id, email, data.name)
    
    # Step 9: CSV Logging
    # Save all data to CSV file for tracking
    try:
        save_to_csv(final_data, final_contacts or [], data.name, data.email)
    except Exception as e:
        logger.error(f"❌ CSV Logging Failed: {e}")
    
    logger.info("--- AUTOMATION COMPLETE ---")


# --- API ENDPOINT ---

@app.post("/webhook")
async def catch_webhook(request: Request, background_tasks: BackgroundTasks):
    """
    FastAPI endpoint that catches the webhook.
    It returns 200 OK immediately, then runs the logic.
    """
    # Get raw request body for debugging
    try:
        raw_body = await request.body()
        logger.info(f"📥 Raw webhook body: {raw_body.decode('utf-8')}")
        
        # Parse as JSON
        if raw_body:
            webhook_data = json.loads(raw_body)
            logger.info(f"📋 Parsed webhook data: {json.dumps(webhook_data, indent=2)}")
            
            # Fix malformed keys that have extra quotes
            cleaned_data = {}
            for key, value in webhook_data.items():
                # Remove extra quotes from keys
                clean_key = key.strip('"')
                cleaned_data[clean_key] = value
            
            if cleaned_data != webhook_data:
                logger.info(f"🔧 Fixed malformed keys: {json.dumps(cleaned_data, indent=2)}")
                webhook_data = cleaned_data
                
        else:
            logger.warning("⚠️ Empty webhook body received")
            return {"status": "error", "message": "Empty request body"}
            
        # Create payload object
        payload = WebhookPayload(**webhook_data)
        logger.info(f"✅ Webhook received for: {payload.company_name}")
        logger.info(f"📧 Email: {payload.email}")
        logger.info(f"👤 Name: {payload.name}")
        
        # Pass the data to the background function
        background_tasks.add_task(run_full_automation, payload)
        
        return {"status": "received", "message": "Automation started in background"}
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Failed to parse webhook JSON: {e}")
        return {"status": "error", "message": "Invalid JSON format"}
    except Exception as e:
        logger.error(f"❌ Webhook processing error: {e}")
        return {"status": "error", "message": f"Processing error: {str(e)}"}



# To run: uvicorn main:app --reload