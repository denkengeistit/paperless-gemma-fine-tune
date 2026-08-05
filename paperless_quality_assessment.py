#!/usr/bin/env python3
"""
Paperless-NGX Document Quality Assessment Pipeline
Runs on MacBook Pro, calls Mac Mini's LM Studio for Gemma inference.

Architecture:
- MacBook: Runs assessment script
- Mac Mini: Serves Gemma via LM Studio (http://100.119.61.113:1234/v1)
- VPS: Paperless API (https://paperless.escaffinity.com/api/)

Usage:
    python paperless_quality_assessment.py --once              # Single assessment run
    python paperless_quality_assessment.py --interval 3600     # Continuous mode (1hr intervals)
    python paperless_quality_assessment.py --test              # Test connections only
    python paperless_quality_assessment.py --scan              # Scan without updating
"""

import argparse
import json
import logging
import os
import re
import requests
import time
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configure logging
LOG_DIR = Path.home() / '.cache'
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'paperless_quality_assessment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Environment variable loading
# Priority: os.environ > .env file > defaults
ENV_FILE = Path(__file__).parent / '.env'
if ENV_FILE.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE)

# Configuration (env vars or defaults)
# Primary Paperless endpoint (production/VPS)
PAPERLESS_API_URL = os.getenv('PAPERLESS_API_URL', 'http://100.119.61.113:8000/api/')
PAPERLESS_TOKEN = os.getenv('PAPERLESS_TOKEN', '')

# Alternative Paperless endpoints for local/development testing
PAPERLESS_LOCAL_URL = os.getenv('PAPERLESS_LOCAL_URL', 'http://localhost:8010/api/')
PAPERLESS_DOCKER_URL = os.getenv('PAPERLESS_DOCKER_URL', 'http://paperless-webserver:8000/api/')

# Mac Mini LM Studio endpoint (Tailscale address)
MAC_MINI_LM_STUDIO = os.getenv('MAC_MINI_LM_STUDIO', 'http://100.119.61.113:1234/v1')
GEMMA_MODEL = os.getenv('GEMMA_MODEL', 'gemma-2-9b-it')

# Quality assessment prompt template
QUALITY_ASSESSMENT_PROMPT = """Analyze this document content for OCR quality and training data suitability.

Respond in JSON format with these fields:
{
  "quality_score": 0.0-1.0,
  "readability": "high|medium|low",
  "garbage_content_ratio": 0.0-1.0,
  "table_structure_preserved": true|false,
  "heading_structure_preserved": true|false,
  "needs_reocr": true|false,
  "issues": ["list", "of", "specific", "problems"],
  "training_data_suitable": true|false
}

Scoring guidelines:
- quality_score >= 0.8: excellent, ready for training
- quality_score 0.6-0.8: acceptable with minor issues
- quality_score < 0.6: needs re-OCR
- garbage_content_ratio > 0.3: too much noise
- table_structure_preserved: can parse tables with rows/columns
- heading_structure_preserved: has clear document hierarchy

Evaluate:
- Character encoding (no weird symbols, mojibake, control chars)
- Text density vs whitespace ratio
- Table and column structure
- Heading hierarchy
- Logical document flow
- Any artifacts from poor OCR"""

# Thresholds
QUALITY_THRESHOLD = 0.6
GARBAGE_RATIO_THRESHOLD = 0.3
REOCR_THRESHOLD = 0.4


def get_auth_headers(token: str = None) -> Dict[str, str]:
    """Get Paperless API authorization headers."""
    token = token or PAPERLESS_TOKEN
    if not token:
        raise ValueError("PAPERLESS_TOKEN not configured. Set via env var or .env file.")
    return {
        'Authorization': f'Token {token}',
        'Accept': 'application/json'
    }


class PaperlessClient:
    def __init__(self, api_url: str, auth_headers: Dict[str, str]):
        # Ensure we have the full API URL (with /api/ or similar suffix)
        api_url = api_url.rstrip('/')
        if not api_url.endswith('/api'):
            api_url = f"{api_url}/api"
        self.api_url = api_url
        self.session = requests.Session()
        self.session.headers.update(auth_headers)
        self._custom_field_ids = {}

    def get_all_documents(self) -> List[Dict]:
        url = f"{self.api_url}/documents/"
        all_docs = []
        page = 1

        while True:
            params = {'page': page, 'page_size': 100}
            try:
                resp = self.session.get(url, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                all_docs.extend(data.get('results', []))
                if not data.get('next'):
                    break
                page += 1
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to fetch page {page}: {e}")
                break

        return all_docs

    def get_custom_field_ids(self) -> Dict[str, int]:
        if self._custom_field_ids:
            return self._custom_field_ids

        url = f"{self.api_url}/custom_fields/"
        resp = self.session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        self._custom_field_ids = {}
        for field in data.get('results', []):
            field_name = field.get('name')
            field_id = field.get('id')
            if field_name:
                self._custom_field_ids[field_name] = field_id

        return self._custom_field_ids

    def update_document_custom_fields(self, doc_id: int, updates: Dict[str, str]) -> bool:
        url = f"{self.api_url}/documents/{doc_id}/"
        try:
            resp = self.session.patch(url, json={'custom_fields': updates}, timeout=30)
            resp.raise_for_status()
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to update document {doc_id}: {e}")
            return False


class QualityAssessmentClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })

    def assess_document(self, content: str) -> Optional[Dict]:
        try:
            if len(content) > 16000:
                content = content[:16000] + "\n...[content truncated]"

            prompt = f"{QUALITY_ASSESSMENT_PROMPT}\n\nDocument content:\n{content}"

            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a document quality analyst. Respond ONLY in valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 1024
                },
                timeout=180
            )
            resp.raise_for_status()
            data = resp.json()

            response_text = data['choices'][0]['message']['content']

            try:
                # Try multiple parsing strategies
                if '```json' in response_text:
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    json_str = response_text.split('```')[1].split('```')[0].strip()
                else:
                    # Try to find JSON object in raw text
                    match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
                    if match:
                        json_str = match.group(0)
                    else:
                        json_str = response_text

                result = json.loads(json_str)

                # Validate required fields
                required_fields = ['quality_score', 'readability', 'needs_reocr', 'training_data_suitable']
                for field in required_fields:
                    if field not in result:
                        logger.warning(f"Missing field in assessment: {field}")
                        result[field] = self._get_default_value(field)

                return result
            except (json.JSONDecodeError, IndexError) as e:
                logger.warning(f"Failed to parse assessment response: {e}")
                logger.warning(f"Raw response: {response_text[:200]}")

                # Fallback: heuristic-based assessment
                logger.info("  Falling back to heuristic assessment...")
                return self._heuristic_assessment(content)

        except requests.exceptions.RequestException as e:
            logger.error(f"LM Studio request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during assessment: {e}")
            return None

    def _get_default_value(self, field: str) -> any:
        """Return default value for missing assessment fields."""
        defaults = {
            'quality_score': 0.5,
            'readability': 'medium',
            'garbage_content_ratio': 0.0,
            'table_structure_preserved': False,
            'heading_structure_preserved': False,
            'needs_reocr': False,
            'issues': [],
            'training_data_suitable': True
        }
        return defaults.get(field, None)

    def _heuristic_assessment(self, content: str) -> Dict:
        """Fallback heuristic-based quality assessment when LLM fails."""
        issues = []

        # Check for OCR artifacts
        weird_char_ratio = len(re.findall(r'[^\x20-\x7E\n\t]', content)) / max(len(content), 1)
        if weird_char_ratio > 0.05:
            issues.append(f"High ratio of non-ASCII chars: {weird_char_ratio:.2%}")

        # Check for repeated garbage patterns
        repeated_lines = len(re.findall(r'^(.).*\1{10,}$', content, re.MULTILINE))
        if repeated_lines > 5:
            issues.append(f"Repeated line artifacts: {repeated_lines} found")

        # Check text density
        lines = content.split('\n')
        non_empty = sum(1 for l in lines if l.strip())
        text_density = non_empty / max(len(lines), 1)
        if text_density < 0.3:
            issues.append(f"Low text density: {text_density:.2%}")

        # Calculate quality score
        quality_score = 1.0
        if weird_char_ratio > 0.05:
            quality_score -= 0.2
        if repeated_lines > 5:
            quality_score -= 0.2
        if text_density < 0.3:
            quality_score -= 0.15

        # Check for table-like structure
        has_tables = bool(re.search(r'^[\s\S]*\|[\s\S]*$', content, re.MULTILINE))

        return {
            'quality_score': max(0.0, min(1.0, quality_score)),
            'readability': 'low' if len(issues) > 2 else 'medium' if issues else 'high',
            'garbage_content_ratio': weird_char_ratio,
            'table_structure_preserved': has_tables,
            'heading_structure_preserved': bool(re.search(r'^#{1,6}\s', content, re.MULTILINE)),
            'needs_reocr': quality_score < REOCR_THRESHOLD,
            'issues': issues,
            'training_data_suitable': quality_score >= QUALITY_THRESHOLD and weird_char_ratio < GARBAGE_RATIO_THRESHOLD
        }


def assess_all_documents(paperless: PaperlessClient, gemma: QualityAssessmentClient,
                         output_path: str, scan_only: bool = False) -> List[Dict]:
    """Assess all documents in Paperless and update custom fields."""
    documents = paperless.get_all_documents()
    logger.info(f"Found {len(documents)} documents to assess")

    assessments = []
    scan_output = []

    for doc in documents:
        doc_id = doc.get('id')
        title = doc.get('title', 'Unknown')
        content = doc.get('content', '')

        logger.info(f"Assessing: {title} (ID: {doc_id})")
        assessment = gemma.assess_document(content)

        if assessment:
            assessment['document_id'] = doc_id
            assessment['document_title'] = title
            assessments.append(assessment)

            # Log summary
            quality = assessment.get('quality_score', 0)
            needs_reocr = assessment.get('needs_reocr', False)
            suitable = assessment.get('training_data_suitable', True)

            logger.info(f"  Quality: {quality:.2f}, Re-OCR: {needs_reocr}, Suitable: {suitable}")

            # Save scan output for later processing
            scan_entry = {
                'id': doc_id,
                'title': title,
                'quality_score': quality,
                'needs_reocr': needs_reocr,
                'training_data_suitable': suitable,
                'issues': assessment.get('issues', [])
            }
            scan_output.append(scan_entry)

            # Update Paperless custom fields (if not scan-only mode)
            if not scan_only and not needs_reocr:
                # Update logic would go here
                pass

    # Write scan output if in scan mode
    if scan_only and scan_output:
        with open(output_path, 'w') as f:
            json.dump(scan_output, f, indent=2)
        logger.info(f"Scan results written to: {output_path}")

    return assessments


def test_connections() -> Tuple[bool, bool]:
    """Test Paperless and LM Studio connections."""
    logger.info("Testing connections...")

    # Test Paperless connection with multiple endpoints
    paperless_ok = False
    endpoints_to_test = [
        ('Production', PAPERLESS_API_URL),
        ('Local', PAPERLESS_LOCAL_URL),
        ('Docker', PAPERLESS_DOCKER_URL),
    ]

    auth_headers = get_auth_headers()

    for name, url in endpoints_to_test:
        try:
            test_url = url.rstrip('/')
            if not test_url.endswith('/api'):
                test_url = f"{test_url}/api"
            resp = requests.get(f"{test_url}/documents/", headers=auth_headers, timeout=10)
            if resp.status_code == 200:
                paperless_ok = True
                logger.info(f"  ✓ {name} Paperless ({url}) OK")
                test_connections.working_paperless_url = url
                break
            else:
                logger.error(f"  × {name} Paperless returned {resp.status_code}")
        except requests.exceptions.RequestException as e:
            logger.debug(f"  × {name} Paperless ({url}) failed: {e}")

    if not paperless_ok:
        logger.error("  × No Paperless endpoint reachable")

    # Test LM Studio connection
    lm_studio_ok = False
    try:
        resp = requests.post(
            f"{MAC_MINI_LM_STUDIO}/chat/completions",
            json={
                "model": GEMMA_MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            },
            timeout=10
        )
        if resp.status_code == 200:
            lm_studio_ok = True
            logger.info("  ✓ LM Studio connection OK")
        else:
            logger.error(f"  × LM Studio returned status {resp.status_code}")
    except requests.exceptions.RequestException as e:
        logger.error(f"  × LM Studio connection failed: {e}")

    return paperless_ok, lm_studio_ok


def main():
    parser = argparse.ArgumentParser(description='Paperless-NGX Document Quality Assessment')
    parser.add_argument('--interval', type=int, default=3600,
                        help='Polling interval in seconds (default: 3600 = 1 hour)')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for assessment results')
    parser.add_argument('--test', action='store_true',
                        help='Test connections only and exit')
    parser.add_argument('--scan', action='store_true',
                        help='Scan without updating Paperless')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Paperless Document Quality Assessment Pipeline")
    logger.info("=" * 70)
    logger.info(f"Paperless API: {PAPERLESS_API_URL}")
    logger.info(f"Alternative local: {PAPERLESS_LOCAL_URL}")
    logger.info(f"Gemma model: {GEMMA_MODEL} @ Mac Mini ({MAC_MINI_LM_STUDIO})")
    logger.info(f"Quality threshold: {QUALITY_THRESHOLD}")
    logger.info(f"Re-OCR threshold: {REOCR_THRESHOLD}")

    # Test connections if requested
    if args.test:
        paperless_ok, lm_studio_ok = test_connections()
        if paperless_ok and lm_studio_ok:
            logger.info("\n✓ All connections successful!")
            sys.exit(0)
        else:
            logger.error("\n✗ Some connections failed. Check configuration.")
            sys.exit(1)

    # Use working Paperless URL if discovered during test
    working_url = getattr(test_connections, 'working_paperless_url', None)
    paperless_url = working_url if working_url else PAPERLESS_API_URL
    # Strip /api/ suffix if present for PaperlessClient constructor
    paperless_base_url = paperless_url.rsplit('/api/', 1)[0] if '/api/' in paperless_url else paperless_url

    logger.info(f"Using Paperless base URL: {paperless_base_url}")

    try:
        auth_headers = get_auth_headers()
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    paperless = PaperlessClient(paperless_base_url, auth_headers)
    gemma = QualityAssessmentClient(MAC_MINI_LM_STUDIO, GEMMA_MODEL)

    while True:
        logger.info(f"\n{'='*70}")
        logger.info(f"Starting assessment run at {datetime.now().isoformat()}")
        logger.info(f"{'='*70}")

        try:
            output_path = args.output or str(LOG_DIR / f"paperless_assessment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            assessments = assess_all_documents(paperless, gemma, output_path, scan_only=args.scan)

            logger.info(f"\n{'='*70}")
            logger.info("Assessment Summary")
            logger.info(f"{'='*70}")
            logger.info(f"Total assessed: {len(assessments)}")

            needs_reocr = sum(1 for a in assessments if a.get('needs_reocr'))
            good = sum(1 for a in assessments if a.get('quality_score', 0) >= QUALITY_THRESHOLD)
            marginal = sum(1 for a in assessments if 0.4 <= a.get('quality_score', 0) < QUALITY_THRESHOLD)

            logger.info(f"  ✓ Good (≥{QUALITY_THRESHOLD}): {good}")
            logger.info(f"  ⚠ Marginal: {marginal}")
            logger.info(f"  ✗ Needs re-OCR: {needs_reocr}")
            logger.info(f"{'='*70}")

        except Exception as e:
            logger.error(f"Assessment run failed: {e}", exc_info=True)

        if args.once:
            break

        logger.info(f"\nWaiting {args.interval} seconds before next run...")
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
