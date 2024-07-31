import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
import logging
from datetime import datetime
from config import GEMINI_API_KEYS
import pandas as pd
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class APIKeyManager:
    def __init__(self, api_keys):
        self.api_keys = api_keys
        self.current_index = 0

    def get_next_key(self):
        key = self.api_keys[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return key

api_key_manager = APIKeyManager(GEMINI_API_KEYS)

from urllib.parse import urlparse

def normalize_url(url):
    # Add https:// if no protocol is specified
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse the URL
    parsed = urlparse(url)
    
    # Reconstruct the URL without trailing slash
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}"
    
    return normalized.lower()

def search_excel(query, excel_path):
    xls = pd.ExcelFile(excel_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        
        # Find the column that contains "Roster URL"
        roster_url_column = next((col for col in df.columns if 'Roster URL' in col), None)
        
        # Normalize the query if it looks like a URL
        query_normalized = normalize_url(query) if '/' in query else query.lower()
        
        for index, row in df.iterrows():
            school_match = query.lower() in str(row['School']).lower()
            nickname_match = query.lower() in str(row.get('Nickname', '')).lower()
            
            url_match = False
            if roster_url_column:
                excel_url = str(row.get(roster_url_column, ''))
                if excel_url:
                    excel_url_normalized = normalize_url(excel_url)
                    url_match = query_normalized == excel_url_normalized
            
            if school_match or nickname_match or url_match:
                return {
                    'sheet': sheet_name,
                    'row': index + 2,  # Adding 2 because Excel rows start at 1 and have a header
                    'school': row['School'],
                    'nickname': row.get('Nickname', 'N/A'),
                    'url': row.get(roster_url_column, 'N/A') if roster_url_column else 'N/A'
                }
    return None

async def gemini_based_scraping(url, school_name, nickname):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                    body = soup.body
                    if body is None:
                        logger.warning(f"No body tag found in the HTML from {url}")
                        return None, False, 0, 0
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
            "max_output_tokens": 8192,
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
        Determine the roster year. Look for an explicit mention of the roster year on the page (e.g., "2024 Softball Roster"). If not found, assume it's for the upcoming season ({current_year + 1}).
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
        Important: Ensure all names, including those with non-English characters, are preserved exactly as they appear in the HTML. Do not escape or modify any special characters in names, hometowns, or school names.
        The response should be a valid JSON string only, without any additional formatting or markdown syntax.
        """

        token_response = model.count_tokens(prompt + str(body))
        input_tokens = token_response.total_tokens

        response = await model.generate_content_async([prompt, str(body)])
        
        output_tokens = model.count_tokens(response.text).total_tokens

        # Clean the response text
        cleaned_response = response.text.strip()
        if cleaned_response.startswith('```json'):
            cleaned_response = cleaned_response[7:]
        if cleaned_response.endswith('```'):
            cleaned_response = cleaned_response[:-3]

        cleaned_response = cleaned_response.strip()

        try:
            result = json.loads(cleaned_response)
            result['player_count'] = len(result['players'])
            if len(result['players']) == 0:
                result['success'] = False
                result['reason'] = "No players found in the extracted data"
            return result, result['success'], input_tokens, output_tokens
        except json.JSONDecodeError as json_error:
            logger.error(f"Failed to parse JSON from Gemini response: {json_error}")
            logger.debug(f"Raw response: {cleaned_response}")
            return None, False, input_tokens, output_tokens
    except Exception as e:
        logger.error(f"Error in Gemini-based scraping: {str(e)}")
        return None, False, 0, 0

async def main():
    excel_path = r"C:\Users\dell3\source\repos\school-data-scraper-3\Freelancer_Data_Mining_Project.xlsx"
    if not os.path.exists(excel_path):
        print(f"Error: Excel file not found at {excel_path}")
        return

    query = input("Enter the URL, school name, or nickname to search: ")
    
    search_result = search_excel(query, excel_path)
    
    if search_result:
        print(f"\nFound matching entry in Excel:")
        print(f"Sheet: {search_result['sheet']}")
        print(f"Row: {search_result['row']}")
        print(f"School: {search_result['school']}")
        print(f"Nickname: {search_result['nickname']}")
        print(f"URL: {search_result['url']}")
        
        proceed = input("\nDo you want to proceed with scraping? (yes/no): ").lower()
        if proceed != 'yes':
            print("Scraping cancelled.")
            return
    else:
        print("No matching entry found in the Excel file.")
        return

    url = search_result['url']
    school_name = search_result['school']
    nickname = search_result['nickname']

    print(f"\nScraping {url} for {school_name} ({nickname})...")
    
    max_retries = 3
    base_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            result, success, input_tokens, output_tokens = await gemini_based_scraping(url, school_name, nickname)
            
            print("\nScraping Results:")
            print(f"Success: {success}")
            print(f"Input Tokens: {input_tokens}")
            print(f"Output Tokens: {output_tokens}")
            print(f"Total Tokens: {input_tokens + output_tokens}")
            
            if success and result:
                print(f"\nRoster Year: {result['rosterYear']}")
                print(f"Player Count: {result['player_count']}")
                print("\nJSON Output:")
                print(json.dumps(result, indent=2, ensure_ascii=False))
                break
            else:
                print(f"\nScraping failed. Reason: {result['reason'] if result and 'reason' in result else 'Unknown error'}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    print("Max retries reached. Scraping unsuccessful.")
        except Exception as e:
            logger.error(f"An error occurred: {str(e)}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
            else:
                print("Max retries reached. Scraping unsuccessful.")

if __name__ == "__main__":
    asyncio.run(main())