# --- START OF FILE explanation_module.py ---

# import necessary libraries
import re
import PyPDF2
import google.generativeai as genai
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate # HumanMessagePromptTemplate not used
import os
import io

_model = None # Internal global model instance
_gemini_configured = False

def configure_gemini_explicitly(api_key: str):
    global _model, _gemini_configured
    effective_api_key = api_key
    if not effective_api_key:
        print("INFO: No API key provided for explicit configuration. Trying environment variables (API_KEY or GEMINI_API_KEY).")
        effective_api_key = os.getenv("API_KEY") or os.getenv("GEMINI_API_KEY")
    
    if not effective_api_key:
        _gemini_configured = False
        raise ValueError("Gemini API Key must be provided either explicitly or via API_KEY/GEMINI_API_KEY environment variable.")

    try:
        genai.configure(api_key=effective_api_key)
        _model = genai.GenerativeModel('gemini-1.5-flash-latest')
        _gemini_configured = True
        print("INFO: Gemini model configured and initialized successfully for explanation module.")
    except Exception as e:
        _gemini_configured = False
        print(f"ERROR: Failed to configure Gemini for explanation module: {e}")
        raise

def get_model():
    global _model, _gemini_configured
    if not _gemini_configured:
        print("INFO: Gemini (explanation_module) not configured. Attempting to auto-configure from environment variables.")
        try:
            configure_gemini_explicitly(None) # Tries env vars
        except ValueError as e:
            raise RuntimeError(f"Gemini model (explanation_module) not configured and failed to auto-configure: {e}. "
                               "Please call configure_gemini_explicitly first or set API_KEY/GEMINI_API_KEY environment variable.")
    if _model is None:
        raise RuntimeError("Gemini model (explanation_module) is None even after configuration attempt.")
    return _model

def generate_text(prompt, max_length=100, temperature=0.5):
    """Generate text using Gemini model"""
    model_instance = get_model()
    try:
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_length * 4, 
            temperature=temperature,
        )
        response = model_instance.generate_content(
            prompt,
            generation_config=generation_config
        )
        return [response.text]
    except Exception as e:
        print(f"Error generating text in explanation_module: {e}")
        return [f"Error: {str(e)}"]

SECTION_PATTERNS = {
    "abstract": r'\babstract\b',
    "introduction": r'\bintroduction\b',
    "methodology": r'\b(methodology|methods|materials and methods)\b',
    "results": r'\b(results|findings)\b',
    "conclusion": r'\b(conclusion|conclusions|discussion and conclusion)\b'
}

SECTION_REGEX = {k: re.compile(v, re.IGNORECASE) for k, v in SECTION_PATTERNS.items()}

def extract_text_from_pdf_bytes(pdf_content_bytes): # Renamed to clarify input
    """Extracts text from PDF content bytes."""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(pdf_content_bytes))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF bytes: {e}")
        return ""

def extract_sections(text):
    text_cleaned = re.sub(r'\s+', ' ', text).strip()
    # Ensure keys in SECTION_PATTERNS are valid group names (alphanumeric, underscore)
    valid_group_names = {k.replace('-', '_'): v for k, v in SECTION_PATTERNS.items()}
    combined_pattern = r'|'.join(f'(?P<{k}>{v})' for k, v in valid_group_names.items())
    
    section_header_regex = re.compile(combined_pattern, re.IGNORECASE)

    sections = {}
    matches = list(section_header_regex.finditer(text_cleaned))
    
    if not matches: # If no standard sections found, return the whole text as 'full_text'
        if text_cleaned:
             return {"full_text": text_cleaned} # Provide the full text if no sections are found
        return {}


    for i, match in enumerate(matches):
        section_name = match.lastgroup # This will be the key from valid_group_names
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text_cleaned)
        
        # Extract section header and its content
        # The content starts after the matched header
        header_match = re.search(valid_group_names[section_name], text_cleaned[start:end], re.IGNORECASE)
        if header_match:
            content_start = start + header_match.end()
            content = text_cleaned[content_start:end].strip()
            sections[section_name] = content
        else: # Should not happen if match.lastgroup is correct
            sections[section_name] = text_cleaned[start:end].strip()


    return sections

def extract_json_block(text):
    pattern = r'```json\s*(.*?)\s*```'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        # Return the content of the last JSON block, assuming it's the intended structured output
        return matches[-1] 

    # If no markdown json found, try to find JSON directly (simplified)
    # This simplified regex looks for a basic JSON object structure.
    json_pattern = r'^\s*\{.*\}\s*$' 
    if re.search(json_pattern, text.strip(), re.DOTALL):
        return text.strip()
    
    # Fallback: if parsing fails later, this might mean it wasn't a valid JSON string to begin with.
    # The parser itself will handle this. This function's role is just to isolate the potential JSON string.
    print(f"Warning: Could not clearly extract a JSON block from: '{text[:200]}...'")
    return text # Return the text as is, parser will try its best

def Simplify_research_part(content):
    explanation_schema = ResponseSchema(
        name="explanation",
        description="A simplified and clear explanation of the provided research text."
    )
    output_parser = StructuredOutputParser.from_response_schemas([explanation_schema])
    format_instructions = output_parser.get_format_instructions()

    prompt_template_str = """\
    You are a helpful assistant that explains research papers in a simple way.

    TASK:
    - Read the provided research content carefully.
    - Return a clear and simplified explanation in one paragraph using plain, accurate language.
    - Avoid adding any information that is not in the original content.
    - The goal is to help anyone understand this part, even without a research background, while staying true to the original meaning.
    - Format your response STRICTLY as a JSON object with a single key "explanation". Example: {{"explanation": "Simplified text here."}}

    Content:
    {content}

    {format_instructions}
    """
    
    prompt = PromptTemplate(
        template=prompt_template_str,
        input_variables=["content"],
        partial_variables={"format_instructions": format_instructions}
    )

    formatted_prompt = prompt.format(content=content)
    response_texts = generate_text(formatted_prompt, max_length=500, temperature=0.5) # Increased temp slightly
    raw_response_text = response_texts[0]

    if raw_response_text.startswith("Error:"):
        print(f"Error from generate_text: {raw_response_text}")
        return {"explanation": "Unable to simplify this section due to a generation error."}

    # Attempt to extract and parse JSON
    json_candidate_str = extract_json_block(raw_response_text)
    try:
        parsed_output = output_parser.parse(json_candidate_str)
        return parsed_output
    except Exception as e:
        print(f"Parsing error for Simplify_research_part: {e}. Raw response: '{raw_response_text[:300]}...'")
        # Fallback: Use the raw response as explanation if it's not an error message
        if "explanation" not in raw_response_text.lower() and len(raw_response_text) < 300 : # Heuristic for non-JSON-like text
             return {"explanation": raw_response_text.strip()}
        else: # If it looks like it tried to be JSON or is too long/complex for a direct explanation
             return {"explanation": "Unable to reliably parse the simplified explanation for this section. The model may not have followed JSON format."}


def process_pdf(pdf_content_bytes):
    try:
        full_text = extract_text_from_pdf_bytes(pdf_content_bytes)
        if not full_text:
            return {"error": "Could not extract text from PDF."}
        
        extracted_sections = extract_sections(full_text)
        results = {}

        # Define which sections to process, can be all keys from extracted_sections or a predefined list
        # If 'full_text' is the only key, process that.
        sections_to_process = list(extracted_sections.keys())
        if not sections_to_process : # or list(SECTION_PATTERNS.keys())
             return {"error": "No sections identified for processing after text extraction."}


        for key in sections_to_process:
            print(f"\n\n===== Simplifying Section: {key.capitalize()} =====\n")
            section_text = extracted_sections.get(key, "")
            
            if section_text:
                # Limit section text length to avoid token limits and costs
                # Gemini 1.5 Flash has a large context window, but summarization should be on concise parts.
                # Max input tokens for this model is very high, but let's be reasonable for single section summary.
                max_chars_for_simplification = 8000 # Approx 2000 tokens
                if len(section_text) > max_chars_for_simplification:
                    print(f"Warning: Section '{key}' is too long ({len(section_text)} chars). Truncating to {max_chars_for_simplification} chars for simplification.")
                    section_text = section_text[:max_chars_for_simplification] + "..."
                
                try:
                    simplified = Simplify_research_part(section_text)
                    results[key] = simplified.get('explanation', "No explanation found after simplification.")
                    print(f"✅ Successfully simplified {key}")
                except Exception as e:
                    print(f"❌ Could not simplify {key}: {str(e)}")
                    results[key] = f"Error simplifying {key}: {str(e)}"
            else:
                print(f"Section {key} not found or is empty.")
                results[key] = "Section not found or empty in the document."
        
        if not results: # If after processing all sections, results is still empty
            return {"error": "No content could be processed or simplified from the PDF."}
            
        return results
    
    except Exception as e:
        print(f"Overall error processing PDF in explanation_module: {str(e)}")
        return {"error": f"Overall error processing PDF: {str(e)}"}
    
def test_gemini():
    """Test if Gemini is properly configured for the explanation module"""
    try:
        model_instance = get_model() 
        response = model_instance.generate_content("Please respond with 'Gemini (explanation_module) is working correctly!'")
        text_response = response.text.strip()
        if "Gemini (explanation_module) is working correctly!" in text_response:
            print(f"✅ Gemini test (explanation_module) successful: {text_response}")
            return True
        else:
            print(f"❌ Gemini test (explanation_module) failed: Unexpected response '{text_response}'")
            return False
    except Exception as e:
        print(f"❌ Gemini test (explanation_module) failed: {e}")
        return False
# --- END OF FILE explanation_module.py ---
