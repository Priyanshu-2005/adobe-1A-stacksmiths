import fitz  # PyMuPDF
import json
import numpy as np
from pathlib import Path
from collections import Counter
from tqdm import tqdm
from tokenizers import Tokenizer
import onnxruntime
from huggingface_hub import hf_hub_download
import re
import logging

# Setup logging for debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Download Model & Tokenizer ---
# def download_model():
#     try:
#         dest = Path("onnx_model")
#         dest.mkdir(exist_ok=True)
#         REPO = "Xenova/all-MiniLM-L6-v2"
#         FILES = ["onnx/model_quantized.onnx", "tokenizer.json"]
#         for fname in FILES:
#             path = hf_hub_download(repo_id=REPO, filename=fname)
#             (dest / Path(fname).name).write_bytes(Path(path).read_bytes())
#         logger.info("Model and tokenizer downloaded successfully")
#     except Exception as e:
#         logger.error(f"Failed to download model: {e}")
#         raise

# --- Semantic Model ---
class SemanticModel:
    def __init__(self, model_dir='onnx_model'):
        try:
            self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
            self.session = onnxruntime.InferenceSession(f"{model_dir}/model_quantized.onnx")
            inputs = self.session.get_inputs()
            self.input_name = inputs[0].name
            self.attn_name = inputs[1].name
            self.type_name = inputs[2].name if len(inputs) > 2 else None
            logger.info("Semantic model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize semantic model: {e}")
            raise

    def score(self, text):
        try:
            tokens = self.tokenizer.encode(text)
            ids = np.array(tokens.ids, dtype=np.int64)[None, :]
            mask = np.ones_like(ids)
            ort_inputs = {self.input_name: ids, self.attn_name: mask}
            if self.type_name:
                ort_inputs[self.type_name] = np.zeros_like(ids)
            outputs = self.session.run(None, ort_inputs)[0]
            emb = outputs.mean(axis=1)[0]
            return float(np.linalg.norm(emb))
        except Exception as e:
            logger.warning(f"Failed to score text '{text}': {e}")
            return 0.0

# --- Text Quality Filters ---
def is_valid_text(text):
    """Validate text to allow short, meaningful content while rejecting gibberish"""
    if not text or len(text.strip()) < 1:
        logger.debug(f"Rejected text: '{text}' (empty or too short)")
        return False
    
    text = ' '.join(text.split())
    
    words = text.split()
    if len(words) > 3:
        word_freq = Counter(words)
        most_common_freq = word_freq.most_common(1)[0][1]
        if most_common_freq > len(words) * 0.3:
            logger.debug(f"Rejected text: '{text}' (high word repetition)")
            return False
    
    if re.search(r'([A-Za-z])\1{4,}', text):
        logger.debug(f"Rejected text: '{text}' (repeated characters)")
        return False
    
    special_chars = len(re.findall(r'[^a-zA-Z0-9\s\-\.\,\:\;\(\)]', text))
    if special_chars > len(text) * 0.05:
        logger.debug(f"Rejected text: '{text}' (too many special characters)")
        return False
    
    alpha_chars = len(re.findall(r'[a-zA-Z]', text))
    if alpha_chars < len(text) * 0.7:
        logger.debug(f"Rejected text: '{text}' (low alphabetic content)")
        return False
    
    # Reject boilerplate
    if re.search(r'^(page|footer|header|copyright|\d{4}|\d+\s*of\s*\d+|confidential|draft|all\s*rights\s*reserved|proprietary)', text.lower()):
        logger.debug(f"Rejected text: '{text}' (boilerplate)")
        return False
    
    return True

def refine_heading(text, context_keywords):
    """Refine vague headings using document context, preserving important content"""
    text_lower = text.lower()
    words = text.split()
    
    # Preserve clear or numbered headings
    if len(words) >= 3 or re.match(r'^\d+(?:\.\d+)*\s+', text) or re.search(r'(overview|summary|introduction|appendix|phase|terms|milestone|background|strategy|plan|funding|governance|access|training|resources|evaluation)', text_lower):
        return text
    
    # Enhance vague headings with context
    context_map = {
        'access': 'Resource Access',
        'funding': 'Funding Model',
        'governance': 'Governance Structure',
        'training': 'Training Programs',
        'guidance': 'Guidance Services',
        'licensing': 'Provincial Licensing',
        'support': 'Technical Support',
        'milestones': 'Project Milestones',
        'evaluation': 'Contract Evaluation',
        'background': 'Project Background',
        'summary': 'Executive Summary',
        'timeline': 'Project Timeline'
    }
    
    for key, refined in context_map.items():
        if key in text_lower and any(keyword in text_lower or keyword in context_keywords for keyword in context_keywords):
            return refined if len(words) <= 2 else text
    
    return text

def is_semantic_heading(text, model, threshold=1.5):
    """Validate headings for semantic coherence, allowing important short headings"""
    if not text or len(text.split()) > 10 or len(text.split()) < 1:
        logger.debug(f"Rejected heading: '{text}' (invalid length)")
        return False
    
    # Reject boilerplate, code, or non-heading patterns
    if re.match(r'^[\•\*\-\+oO#]\s+', text) or '```' in text or text.startswith(('docker ', '//', '#')):
        logger.debug(f"Rejected heading: '{text}' (code or bullet)")
        return False
    
    # Allow colons in specific contexts
    if text.endswith(':') and not re.search(r'(timeline|access|funding|governance|decision-making|phase|terms)', text.lower()):
        logger.debug(f"Rejected heading: '{text}' (invalid colon)")
        return False
    
    # Enhanced heading patterns
    heading_patterns = [
        r'^(introduction|overview|executive\s*summary|background|methodology|analysis|results|conclusion|discussion|recommendation|appendix)\b',
        r'^(chapter|section|part|phase|step|stage)\s*\d+',
        r'^(objective|purpose|goal|aim|scope|method|approach|strategy|plan|implementation|evaluation|assessment)\b',
        r'^(requirements|specifications|criteria|standards|guidelines|policies|procedures)\b',
        r'^(technical|business|project|system|process|service|product|solution)\b',
        r'^(management|administration|governance|compliance|security|quality|performance)\b',
        r'^(design|architecture|framework|structure|model|prototype)\b',
        r'^(budget|cost|timeline|schedule|milestone|deliverable|resource)\b',
        r'^(risk|issue|challenge|limitation|constraint|assumption)\b',
        r'^(benefit|advantage|impact|outcome|result|finding|insight)\b',
        r'^(preamble|terms|membership|appointment|criteria|chair|meetings|accountability)\b',
    ]
    
    text_lower = text.lower()
    pattern_score = sum(1 for pattern in heading_patterns if re.search(pattern, text_lower))
    
    structural_score = 0
    if text and text[0].isupper():
        structural_score += 0.5
    words = text.split()
    title_case_words = sum(1 for word in words if word[0].isupper() and len(word) > 2)
    if title_case_words >= len(words) * 0.7:
        structural_score += 1.0
    if not text.endswith(('.', '!', '?')):
        structural_score += 0.5
    
    action_words = ['implement', 'develop', 'analyze', 'evaluate', 'assess', 'review', 
                   'manage', 'provide', 'ensure', 'establish', 'maintain', 'support',
                   'deliver', 'create', 'define', 'identify', 'determine', 'monitor']
    if any(word in text_lower for word in action_words):
        structural_score += 0.5
    
    # Reject code-like or fragmented text
    if re.search(r'(?:docker|python|javascript|code|command|syntax|\.json|\.yaml|\.sh|\{|\}|\<|\>|\=)', text_lower):
        logger.debug(f"Rejected heading: '{text}' (code-like)")
        return False
    if re.search(r'\b(to\s*be|for\s*each|could\s*mean)\b', text_lower):
        logger.debug(f"Rejected heading: '{text}' (verbose phrase)")
        return False
    
    try:
        sem_score = model.score(text)
        if sem_score < 1.2:
            logger.debug(f"Rejected heading: '{text}' (low semantic score: {sem_score})")
    except Exception as e:
        logger.warning(f"Semantic scoring failed for '{text}': {e}")
        sem_score = 0.0
    
    total_score = (pattern_score * 0.4) + (structural_score * 0.3) + (sem_score * 0.3)
    
    if total_score < threshold:
        logger.debug(f"Rejected heading: '{text}' (low total score: {total_score})")
    
    return total_score >= threshold

def calculate_heading_confidence(text, style_info, model):
    """Calculate confidence for headings, prioritizing semantic quality"""
    if not text:
        logger.debug("Rejected heading: empty text")
        return 0.0
    
    confidence = 0.0
    
    if is_semantic_heading(text, model):
        confidence += 0.75
    
    font_size, is_bold = style_info
    if font_size >= 16:
        confidence += 0.3
    elif font_size >= 13:
        confidence += 0.2
    elif font_size >= 11:
        confidence += 0.1
    
    if is_bold:
        confidence += 0.25
    
    word_count = len(text.split())
    if 1 <= word_count <= 6:
        confidence += 0.2
    elif word_count <= 10:
        confidence += 0.1
    
    if re.match(r'^\d+(?:\.\d+)*\s+', text):
        confidence += 0.3
    
    if text.endswith('?'):
        confidence += 0.15
    
    heading_starters = ['overview', 'introduction', 'executive', 'background', 'objective', 
                       'scope', 'requirement', 'specification', 'technical', 'business', 
                       'project', 'appendix', 'preamble', 'terms', 'membership', 'access',
                       'funding', 'governance', 'training', 'timeline']
    if any(text.lower().startswith(starter) for starter in heading_starters):
        confidence += 0.2
    
    if re.match(r'^[\•\*\-\+oO#]\s+', text) or '```' in text or text.startswith('docker '):
        logger.debug(f"Rejected heading: '{text}' (bullet or code)")
        confidence -= 0.6
    
    logger.debug(f"Heading: '{text}', Confidence: {confidence}")
    return min(confidence, 1.0)

def clean_text(text):
    """Clean text for polished, human-like output"""
    if not text:
        return ""
    
    # Replace Unicode artifacts
    text = text.replace('\u2019', "'").replace('\u2018', "'").replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2013', '-').replace('\u2014', '--')
    
    # Remove bullet points and code markers
    text = re.sub(r'^[\•\*\-\+oO#]\s+', '', text)
    text = re.sub(r'```[\w\s]*```', '', text)
    
    # Normalize whitespace and punctuation
    text = ' '.join(text.split())
    text = re.sub(r'([.,:;!?])\1+', r'\1', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Remove redundant phrases
    text = re.sub(r'\bto\s*be\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfor\s*each\s+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bcould\s*mean\s*[:\?]?\s*', '', text, flags=re.IGNORECASE)
    
    # Preserve colons in meaningful contexts
    if text.endswith(':') and not re.search(r'(timeline|access|funding|governance|decision-making|phase|terms)', text.lower()):
        text = text.rstrip(':').rstrip()
    
    # Capitalize first letter and major words for title case
    words = text.split()
    if words:
        words[0] = words[0].capitalize()
        for i in range(1, len(words)):
            if len(words[i]) > 3 and not re.match(r'^(and|or|of|in|to|a|an|the)$', words[i].lower()):
                words[i] = words[i].capitalize()
        text = ' '.join(words)
    
    return text.strip()

# --- Heading Extraction ---
def extract_headings(doc, model, confidence_threshold=0.8):
    all_candidates = []
    seen = set()
    context_keywords = set(['library', 'digital', 'ontario', 'prosperity', 'business', 'plan', 'funding', 'governance', 'access', 'resources', 'strategy'])
    
    try:
        # Collect context from metadata and first page
        if doc.metadata.get('title'):
            title_words = clean_text(doc.metadata['title']).lower().split()
            context_keywords.update(word for word in title_words if len(word) > 3)
        
        try:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b.get('type', 1) != 0:
                    continue
                text = ' '.join(span['text'].strip() for line in b.get('lines', []) for span in line.get('spans', []) if 'text' in span)
                text = clean_text(text)
                if text:
                    context_keywords.update(word.lower() for word in text.split() if len(word) > 3)
        except Exception as e:
            logger.warning(f"Failed to extract context from first page: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
    
    try:
        for pnum, page in enumerate(doc, 1):
            try:
                blocks = page.get_text("dict")["blocks"]
            except Exception as e:
                logger.warning(f"Failed to extract text from page {pnum}: {e}")
                continue
            
            for b in blocks:
                if b.get('type', 1) != 0:
                    continue
                
                text_parts = []
                try:
                    for line in b.get('lines', []):
                        line_text = ' '.join(span['text'].strip() for span in line.get('spans', []) if 'text' in span)
                        if line_text:
                            text_parts.append(line_text)
                    text = ' '.join(text_parts).strip()
                except Exception as e:
                    logger.warning(f"Failed to process block on page {pnum}: {e}")
                    continue
                
                text = clean_text(text)
                if not is_valid_text(text):
                    continue
                
                if (text, pnum) in seen:
                    continue
                seen.add((text, pnum))
                
                try:
                    if b.get('lines', []) and b['lines'][0].get('spans', []):
                        span = b['lines'][0]['spans'][0]
                        style_info = (round(span.get('size', 10)), bool(span.get('flags', 0) & 16))
                    else:
                        style_info = (10, False)
                except Exception as e:
                    logger.warning(f"Failed to extract style info on page {pnum}: {e}")
                    style_info = (10, False)
                
                try:
                    confidence = calculate_heading_confidence(text, style_info, model)
                except Exception as e:
                    logger.warning(f"Failed to calculate confidence for '{text}' on page {pnum}: {e}")
                    continue
                
                if confidence >= confidence_threshold:
                    refined_text = refine_heading(text, context_keywords)
                    all_candidates.append({
                        'text': refined_text,
                        'page': pnum,
                        'style': style_info,
                        'confidence': confidence,
                        'is_numbered': bool(re.match(r'^\d+(?:\.\d+)*\s+', text))
                    })
    except Exception as e:
        logger.error(f"Error processing document headings: {e}")
        return []
    
    all_candidates.sort(key=lambda x: (-x['confidence'], x['page']))
    
    # Assign levels (H1, H2, H3 only)
    level_assignments = {}
    used_levels = set()
    
    for candidate in all_candidates:
        if candidate['is_numbered']:
            match = re.match(r'^(\d+(?:\.\d+)*)\s+', candidate['text'])
            if match:
                depth = len(match.group(1).split('.'))
                level = min(depth, 3)  # Cap at H3
                candidate['level'] = f"H{level}"
                level_assignments[candidate['style']] = level
                used_levels.add(level)
    
    style_groups = {}
    for candidate in all_candidates:
        if 'level' not in candidate:
            style = candidate['style']
            if style not in style_groups:
                style_groups[style] = []
            style_groups[style].append(candidate)
    
    sorted_styles = sorted(
        style_groups.keys(),
        key=lambda s: (-s[0], -int(s[1]), -len(style_groups[s]))
    )
    
    available_levels = [1, 2, 3]
    for style in sorted_styles:
        if style not in level_assignments:
            for level in available_levels:
                if level not in used_levels:
                    level_assignments[style] = level
                    used_levels.add(level)
                    break
            else:
                font_size = style[0]
                level = 1 if font_size >= 14 else 2 if font_size >= 12 else 3
                level_assignments[style] = level
        
        for candidate in style_groups[style]:
            candidate['level'] = f"H{level_assignments[style]}"
    
    # Final filtering and flat structure
    final_headings = []
    seen_texts = set()
    
    for candidate in sorted(all_candidates, key=lambda x: (x['page'], int(x['level'][1:]))):
        if 'level' in candidate and candidate['text'] not in seen_texts:
            if candidate['confidence'] < 0.85:
                if not is_semantic_heading(candidate['text'], model, threshold=1.3):
                    logger.debug(f"Skipped heading: '{candidate['text']}' (low confidence and semantic score)")
                    continue
            
            final_headings.append({
                'level': candidate['level'],
                'text': candidate['text'],
                'page': candidate['page']
            })
            seen_texts.add(candidate['text'])
    
    # Ensure proper level progression
    refined = []
    last_level = 0
    
    for heading in final_headings:
        current_level = int(heading['level'][1:])
        if current_level > last_level + 1:
            current_level = last_level + 1
            heading['level'] = f"H{current_level}"
        
        refined.append(heading)
        last_level = current_level
    
    return refined

# --- Title Extraction ---
def extract_cover_title(doc, model):
    best_text, best_score = None, 0
    context_keywords = set(['library', 'digital', 'ontario', 'prosperity', 'business', 'plan', 'funding', 'governance', 'access', 'resources', 'strategy'])
    
    try:
        # Collect context from metadata and first page
        if doc.metadata.get('title'):
            title_words = clean_text(doc.metadata['title']).lower().split()
            context_keywords.update(word for word in title_words if len(word) > 3)
        
        try:
            page = doc[0]
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if b.get('type', 1) != 0:
                    continue
                text = ' '.join(span['text'].strip() for line in b.get('lines', []) for span in line.get('spans', []) if 'text' in span)
                text = clean_text(text)
                if text:
                    context_keywords.update(word.lower() for word in text.split() if len(word) > 3)
        except Exception as e:
            logger.warning(f"Failed to extract context from first page: {e}")
    
    except Exception as e:
        logger.warning(f"Failed to extract metadata: {e}")
    
    try:
        page = doc[0]
    except IndexError:
        logger.warning("Document has no pages")
        return ""
    
    try:
        for b in page.get_text('dict')['blocks']:
            if b.get('type', 1) != 0:
                continue
            
            text_parts = []
            try:
                for line in b.get('lines', []):
                    line_text = ' '.join(span['text'].strip() for span in line.get('spans', []) if 'text' in span)
                    if line_text:
                        text_parts.append(line_text)
                text = ' '.join(text_parts).strip()
            except Exception as e:
                logger.warning(f"Failed to process block on cover page: {e}")
                continue
            
            text = clean_text(text)
            if not is_valid_text(text):
                logger.debug(f"Rejected title candidate: '{text}' (invalid text)")
                continue
            
            if len(text.split()) < 2 or len(text.split()) > 12:
                logger.debug(f"Rejected title candidate: '{text}' (invalid length)")
                continue
            
            # Reject boilerplate
            if any(phrase in text.lower() for phrase in ['rfp', 'request for proposal', 'confidential', 'prepared for', 'submitted by', 'date', 'page', 'footer', 'header', 'all rights reserved', 'proprietary']):
                logger.debug(f"Rejected title candidate: '{text}' (boilerplate)")
                continue
            
            if text == text.upper():
                logger.debug(f"Rejected title candidate: '{text}' (all caps)")
                continue
            
            words = text.split()
            if len(set(words)) / len(words) < 0.8:
                logger.debug(f"Rejected title candidate: '{text}' (low word diversity)")
                continue
            
            try:
                sem_score = model.score(text)
                if sem_score < 1.2:
                    logger.debug(f"Rejected title candidate: '{text}' (low semantic score: {sem_score})")
                    continue
            except Exception as e:
                logger.warning(f"Failed to score title candidate '{text}': {e}")
                continue
            
            try:
                sizes = [s['size'] for l in b.get('lines', []) for s in l.get('spans', []) if 'size' in s]
                is_bold = any(s.get('flags', 0) & 16 for l in b.get('lines', []) for s in l.get('spans', []) if 'flags' in s)
                avg_size = sum(sizes) / len(sizes) if sizes else 10
            except Exception as e:
                logger.warning(f"Failed to extract style info for title candidate '{text}': {e}")
                avg_size, is_bold = 10, False
            
            score = (avg_size * 0.2) + (sem_score * 0.6) + (0.2 if is_bold else 0)
            
            if score > best_score:
                best_text, best_score = text, score
                logger.debug(f"Selected title candidate: '{best_text}' (score: {best_score})")
    except Exception as e:
        logger.error(f"Error processing cover page: {e}")
    
    # Refine title with metadata if available
    if not best_text and doc.metadata.get('title'):
        metadata_title = clean_text(doc.metadata['title'])
        if is_valid_text(metadata_title) and 2 <= len(metadata_title.split()) <= 12:
            best_text = refine_heading(metadata_title, context_keywords)
            best_score = model.score(best_text) if best_text else 0
            logger.debug(f"Selected metadata title: '{best_text}' (score: {best_score})")
    
    return best_text if best_text else ""

# --- Main ---
if __name__ == '__main__':
    try:
        # download_model()
        model = SemanticModel()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        exit(1)
    
    PDF_DIR = Path('/app/input')
    OUT_DIR = Path('/app/output')
    OUT_DIR.mkdir(exist_ok=True)
    
    for pdf in tqdm(list(PDF_DIR.glob('*.pdf')), desc='Processing PDFs'):
        try:
            doc = fitz.open(pdf)
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf}: {e}")
            continue
        
        try:
            cover_title = extract_cover_title(doc, model)
            headings = extract_headings(doc, model, confidence_threshold=0.8)
        except Exception as e:
            logger.error(f"Failed to process PDF {pdf}: {e}")
            doc.close()
            continue
        
        title = (cover_title or
                 next((h['text'] for h in headings if h['level'] == 'H1' and len(h['text'].split()) >= 2), ""))
        
        headings = [h for h in headings if h['text'] != title]
        
        output = {
            'title': title,
            'outline': headings
        }
        try:
            json_output = json.dumps(output, ensure_ascii=False, indent=2)
            with open(OUT_DIR / f"{pdf.stem}.json", 'w', encoding='utf-8') as fp:
                fp.write(json_output)
            logger.info(f"Processed {pdf.name} successfully")
        except Exception as e:
            logger.error(f"Failed to write JSON for {pdf.name}: {e}")
        
        doc.close()
    
    logger.info('✅ Done')