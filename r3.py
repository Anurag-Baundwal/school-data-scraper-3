import asyncio
import re
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
from config import GEMINI_API_KEYS, GOOGLE_API_KEY, SEARCH_ENGINE_ID
import random
import logging
from datetime import datetime
import os
from urllib.parse import urlparse
from fuzzywuzzy import fuzz
import json5
import ssl

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler for DEBUG level
file_handler = logging.FileHandler('debug.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

def log_raw_gemini_output(school_name, raw_output, sheet_name):
    # Log raw output
    logger.debug(f"Raw Gemini output for {school_name}:\n{raw_output}")

    # Save raw output to file
    output_dir = "raw_gemini_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_raw_gemini_outputs.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Raw Gemini output for {school_name}:\n")
        f.write(raw_output)
        f.write("\n\n")  # Add two newlines to separate outputs

    logger.info(f"Raw Gemini output for {school_name} saved to {file_path}")

# useless
def save_body_text(school_name, body_text, sheet_name):
    # Save body text to file
    output_dir = "body_texts"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_body_texts.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Body text for {school_name}:\n")
        f.write(body_text)
        f.write("\n\n")  # Add two newlines to separate outputs

    logger.info(f"Body text for {school_name} saved to {file_path}")

# useless
def save_html_content(school_name, html_content, sheet_name):
    output_dir = "html_contents"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_html_contents.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"HTML content for {school_name}:\n")
        f.write(html_content)
        f.write("\n\n" + "="*50 + "\n\n")  # Add a separator between schools

    logger.info(f"HTML content for {school_name} saved to {file_path}")


def save_body_html_content(school_name, body_html, sheet_name):
    output_dir = "body_html_contents"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{sheet_name}_body_html_contents.txt")
    
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"Body HTML content for {school_name}:\n")
        f.write(body_html)
        f.write("\n\n" + "="*50 + "\n\n")  # Add a separator between schools

    logger.info(f"Body HTML content for {school_name} saved to {file_path}")

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0

    def get_next_key(self):
        key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key

api_key_manager = APIKeyManager(GEMINI_API_KEYS)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8').decode('unicode_escape')
        return super().default(obj)

def normalize_url(url):
    # Add https:// if no protocol is specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Reconstruct the URL without trailing slash
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    
    return normalized.lower()

async def load_excel_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        return xls
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return None

# def validate_player_data(players, body_text):
#     valid_players = []
#     for player in players:
#         # Check if player name exists and is not None before calling lower()
#         if player.get('name') and body_text:
#             # Check if at least 70% of the player's name is found in the body text
#             if fuzz.partial_ratio(player['name'].lower(), body_text.lower()) >= 85:
#                 # Check if at least two other fields are found near the name
#                 name_index = body_text.lower().find(player['name'].lower())
#                 if name_index != -1:
#                     surrounding_text = body_text[max(0, name_index - 10000):min(len(body_text), name_index + 10000)]
#                     fields_found = sum(1 for field in ['position', 'year', 'hometown', 'highSchool'] 
#                                        if player.get(field) and player[field].lower() in surrounding_text.lower())
#                     if fields_found >= 2:
#                         valid_players.append(player)
#         else: 
#             print(f"Player has no name. Data: {player}")
#     return valid_players

# def validate_player_data(players, body_text): #v2
#     valid_players = []
#     for player in players:
#         if player.get('name') and body_text:
#             # Log the fuzzy match ratio for debugging
#             match_ratio = fuzz.partial_ratio(player['name'].lower(), body_text.lower())
#             logger.debug(f"Fuzzy match ratio for {player['name']}: {match_ratio}")

#             if match_ratio >= 85:
#                 # Check if at least two other fields are found near the name
#                 name_index = body_text.lower().find(player['name'].lower())
#                 if name_index != -1:
#                     surrounding_text = body_text[max(0, name_index - 10000):min(len(body_text), name_index + 10000)]
#                     fields_found = sum(1 for field in ['position', 'year', 'hometown', 'highSchool'] 
#                                        if player.get(field) and player[field].lower() in surrounding_text.lower())
                    
#                     # Log the number of fields found for debugging
#                     logger.debug(f"Fields found for {player['name']}: {fields_found}")

#                     if fields_found >= 2:
#                         valid_players.append(player)
#                     else:
#                         logger.debug(f"Player {player['name']} rejected: Only {fields_found} fields found")
#                 else:
#                     logger.debug(f"Player {player['name']} rejected: Name not found in body text")
#             else:
#                 logger.debug(f"Player {player['name']} rejected: Fuzzy match ratio below threshold")
#         else:
#             logger.debug(f"Player rejected: Missing name or body text. Player data: {player}")
    
#     logger.info(f"Total players: {len(players)}, Valid players: {len(valid_players)}")
#     return valid_players

def validate_player_data(players, body_html):
    valid_players = []
    body_html_lowercasecase = body_html.lower()
    
    for player in players:
        if player.get('name') and body_html:
            name_lowercase = player['name'].lower()
            
            # First, try an exact match (case-insensitive)
            if player['name'] in body_html:
                logger.debug(f"Exact match found for {player['name']}")
                valid_players.append(player)
                continue

            # If no exact match, try fuzzy matching
            match_ratio = fuzz.partial_ratio(name_lowercase, body_html_lowercasecase)
            logger.debug(f"Fuzzy match ratio for {player['name']}: {match_ratio}")

            if match_ratio >= 85:
                # Check if at least two other fields are found near the name
                name_index = body_html_lowercasecase.find(name_lowercase)
                if name_index != -1:
                    surrounding_text = body_html_lowercasecase[max(0, name_index - 10000):min(len(body_html_lowercasecase), name_index + 10000)]
                    fields_found = sum(1 for field in ['position', 'year', 'hometown', 'highSchool'] 
                                       if player.get(field) and str(player[field]).lower() in surrounding_text)
                    
                    logger.debug(f"Fields found for {player['name']}: {fields_found}")

                    if fields_found >= 2:
                        valid_players.append(player)
                    else:
                        logger.debug(f"Player {player['name']} rejected: Only {fields_found} fields found")
                else:
                    logger.debug(f"Player {player['name']} rejected: Name not found in body text")
            else:
                logger.debug(f"Player {player['name']} rejected: Fuzzy match ratio below threshold")
        else:
            logger.debug(f"Player rejected: Missing name or body text. Player data: {player}")
    
    logger.info(f"Total players: {len(players)}, Valid players: {len(valid_players)}")
    return valid_players


async def fetch_with_retry(session, url, headers, ssl_context, max_retries=3):
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers, ssl=ssl_context) as response:
                if response.status == 200:
                    return await response.text()
                elif response.status == 525:
                    logger.warning(f"SSL Handshake Failed (Error 525) for {url}. Retrying in 2 seconds...")
                    await asyncio.sleep(2)
                else:
                    logger.warning(f"Failed to fetch {url}. Status code: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
            else:
                return None
    return None

async def gemini_based_scraping(url, school_name, nickname, sheet_name):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
    ]
    
    headers = {'User-Agent': random.choice(user_agents)}
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers, ssl=ssl_context) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    body = soup.body
                    if body is None:
                        logger.warning(f"No body tag found in the HTML from {url}")
                        return None, False, 0, 0
                    body_html = str(body)
                    # Save the body HTML content
                    save_body_html_content(school_name, body_html, sheet_name)
                else:
                    logger.warning(f"Failed to fetch {url}. Status code: {response.status}")
                    return None, False, 0, 0
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None, False, 0, 0
    try:
        current_year = datetime.now().year
        genai.configure(api_key=api_key_manager.get_next_key())
        generation_config = {
            "temperature": 0.3,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192*16,
        }

        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config
        )
        
        prompt = f"""
        Analyze the HTML content of the college softball roster webpage from {url}. The expected school name is "{school_name}" and the team nickname or name should be related to "{nickname}". Focus ONLY on player information, ignoring any coach or staff data that might be present. Extract the following information for each player:
        - Name
        - Position
        - Year (Fr, So, Jr, Sr, Grad, etc)
        - Hometown
        - High School
        - Graduation Year (calculate based on the player's year and the roster year)
        Determine the roster year. Look for an explicit mention of the roster year on the page (e.g., "2024 Softball Roster"). If not found, assume it's for the upcoming season ({current_year + 1}). Also, if the roster year consists of a range then it's the first year in the range (e.g.,  "2024-2025 Softball Roster" means that the current roster year is 2024).
        For the Graduation Year calculation, use the determined roster year as the base:
        - Freshman (Fr) or First Year: Roster Year + 3
        - Sophomore (So) or Second Year: Roster Year + 2
        - Junior (Jr) or Third Year: Roster Year + 1
        - Senior (Sr) or Fourth Year: Roster Year
        - Graduate (Grad) or Fifth Year: Roster Year
        - If the year is unclear, set to null
        Format the output as a JSON string with the following structure:
        {{
            "success": true/false,
            "reason": "reason for failure" (or null if success),
            "rosterYear": YYYY,
            "players": [
                {{
                    "name": "...",
                    "position": "...",
                    "year": "...",
                    "hometown": "...",
                    "highSchool": "...",
                    "graduationYear": YYYY
                }},
                ...
            ]
        }}
        Set "success" to false if:
        1. No player data is found
        2. Any player is missing one or more of the required fields (name, position, year, hometown, highSchool)
        3. The roster year cannot be determined
        If "success" is false, provide a brief explanation in the "reason" field.
        Important: Ensure all names, including those with non-English characters, are preserved exactly as they appear in the HTML. Do not escape or modify any special characters in names, hometowns, or school names. For example, 'Montañez' should remain as 'Montañez', not 'Monta\\u00f1ez', and "O'ahu" should remain as "O'ahu", not "O\\u2018ahu".
        The response should be a valid JSON string only, without any additional formatting or markdown syntax.
        Note: If no player data is found on the page, do not create any imaginary players or list players that are not present on the page. Instead, set the 'success' field to false and provide a reason indicating that no players were found on the roster page.
        """

        token_response = model.count_tokens(prompt + body_html)
        input_tokens = token_response.total_tokens

        response = await model.generate_content_async([prompt, body_html])
        
        output_tokens = model.count_tokens(response.text).total_tokens

        # Log the raw output immediately after receiving it
        log_raw_gemini_output(school_name, response.text, sheet_name)

        # Clean the response text
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]

        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            
            # Validate the player data
            if result['success'] and 'players' in result:
                valid_players = validate_player_data(result['players'], body_html)
                result['players'] = valid_players
                result['player_count'] = len(valid_players)
              

                result['player_count_mismatch'] = False

                if len(valid_players) == 0:
                    result['success'] = False
                    result['reason'] = "No valid players found after validation"
            else:
                # Ensure players and player_count are set even if not present in the original response
                result['players'] = []
                result['player_count'] = 0
                result['player_count_mismatch'] = False
            
            # Ensure all expected fields are present
            result.setdefault('success', False)
            result.setdefault('reason', "Unknown error")
            result.setdefault('rosterYear', None)
            result.setdefault('players', [])
            result.setdefault('player_count', 0)
            result.setdefault('player_count_mismatch', False)

            return result, result['success'], input_tokens, output_tokens
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse JSON from Gemini response for {school_name}: {json_error}")
            logger.debug(f"Raw response: {cleaned_response}")
            return {
                'success': False,
                'reason': f"Failed to parse JSON: {str(json_error)}",
                'rosterYear': None,
                'players': [],
                'player_count': 0,
                'player_count_mismatch': False
            }, False, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in Gemini-based scraping for {school_name}: {str(e)}")
        return {
            'success': False,
            'reason': f"Error in scraping: {str(e)}",
            'rosterYear': None,
            'players': [],
            'player_count': 0,
            'player_count_mismatch': False
        }, False, 0, 0

async def process_school(school_data, url_column, sheet_name):
    url = school_data[url_column]
    school_name = school_data['School']
    nickname = school_data.get('Nickname', '')
    max_retries = 3
    base_delay = 5  # seconds
    reasons = []
    total_input_tokens = total_output_tokens = 0

    if pd.notna(url):
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing {school_name} (URL: {url}) - Attempt {attempt + 1}")
                result, success, input_tokens, output_tokens = await gemini_based_scraping(url, school_name, nickname, sheet_name)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_tokens = total_input_tokens + total_output_tokens
                logger.info(f"Tokens used for {school_name} {url_column}: {total_tokens}")
                
                if success:
                    logger.info(f"Successfully scraped data for {school_name}")
                    # pre_validation_players = result['players']  # Store pre-validation players
                    # valid_players = validate_player_data(result['players'], result['body_text'])
                    # player_count = len(valid_players)
                    player_count = len(result['players'])
                    if player_count >= 35:
                        with open(f"scraping-results/{sheet_name}_urls_for_manual_review.txt", 'a') as f:
                            f.write(f"{school_name}: {url} - {player_count} players\n")
                        logger.warning(f"Large roster detected for {school_name}: {player_count} players. Added to manual review.")
                    return {
                        'school': school_name,
                        'url': url,
                        'success': True,
                        'reason': None,  # Set to None when successful
                        'rosterYear': result['rosterYear'],
                        # 'players': valid_players,
                        # 'pre_validation_players': pre_validation_players,  # Add this line
                        # 'player_count': player_count,
                        'players': result['players'],
                        'player_count': result['player_count'],
                        'input_tokens': total_input_tokens,
                        'output_tokens': total_output_tokens,
                        'total_tokens': total_tokens
                    }
                else:
                    reason = result['reason'] if result and 'reason' in result else 'Unknown error'
                    reasons.append(f"Attempt {attempt + 1}: {reason}")
                    logger.warning(f"Scraping failed for {school_name} - Attempt {attempt + 1}: {reason}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(delay)
            except Exception as e:
                reason = f"Error: {str(e)}"
                reasons.append(f"Attempt {attempt + 1}: {reason}")
                logger.error(f"Error in processing {school_name}: {reason} - Attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)

        return {
            'school': school_name,
            'url': url,
            'success': False,
            'reason': '; '.join(reasons),
            'rosterYear': None,
            'players': [],
            # 'pre_validation_players': [],  # Add this line for consistency
            'player_count': 0,
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'total_tokens': total_tokens
        }
    else:
        logger.info(f"Skipping {school_name} - No URL provided")
        return {
            'school': school_name,
            'url': 'N/A',
            'success': False,
            'reason': 'No URL provided',
            'rosterYear': None,
            'players': [],
            # 'pre_validation_players': [],  # Add this line for consistency
            'player_count': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

async def process_sheet(sheet_name, df):
    all_results = []
    total_tokens_used = 0

    semaphore = asyncio.Semaphore(15)  # Limit to 15 concurrent requests

    async def process_with_semaphore(row, url_column):
        async with semaphore:
            return await process_school(row, url_column, sheet_name)

    roster_url_column = next((col for col in df.columns if 'Roster URL' in col), None)
    if roster_url_column:
        logger.info(f"\nProcessing {roster_url_column} URLs for sheet: {sheet_name}")
        
        tasks = [process_with_semaphore(row, roster_url_column) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for r in results if r['success'])
        failed_scrapes = len(results) - successful_scrapes
        tokens_used = sum(r['total_tokens'] for r in results)
        total_tokens_used += tokens_used

        logger.info(f"\nResults for {sheet_name} - {roster_url_column}:")
        logger.info(f"Successful scrapes: {successful_scrapes}")
        logger.info(f"Failed scrapes: {failed_scrapes}")
        logger.info(f"Tokens used: {tokens_used}")

        save_results(results, f"scraping-results/{sheet_name}_{roster_url_column}_results.json")
        save_failed_schools(results, f"scraping-results/{sheet_name}_{roster_url_column}_failed_schools.txt")

        all_results.extend(results)
    else:
        logger.warning(f"No 'Roster URL' column found in sheet: {sheet_name}")

    logger.info(f"\nTotal tokens used for {sheet_name}: {total_tokens_used}")
    return all_results, total_tokens_used

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

def save_failed_schools(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    failed_schools = [f"{r['school']}: {r['url']} - Reason: {r['reason']}" for r in results if not r['success']]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed_schools))

async def main():
    global api_key_manager
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-3\Freelancer_Data_Mining_Project.xlsx"
    
    api_key_manager = APIKeyManager(GEMINI_API_KEYS)
    total_tokens_used = 0
    
    try:
        xls = await load_excel_data(input_file)
        if xls is not None:
            for sheet_name in xls.sheet_names:
                if sheet_name != "JUCO - NWAC":
                    continue
                logger.info(f"\nProcessing sheet: {sheet_name}")
                df = pd.read_excel(xls, sheet_name=sheet_name)
                _, sheet_tokens = await process_sheet(sheet_name, df)
                total_tokens_used += sheet_tokens
            logger.info(f"\nTotal tokens used across all sheets: {total_tokens_used}")
        else:
            logger.error("Failed to load Excel file. Exiting.")
    except Exception as e:
        logger.error(f"An error occurred in the main function: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())