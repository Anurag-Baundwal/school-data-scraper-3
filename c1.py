import asyncio
import aiohttp
import pandas as pd
from bs4 import BeautifulSoup
import json
import google.generativeai as genai
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor
from config import GEMINI_API_KEYS, GOOGLE_API_KEY, SEARCH_ENGINE_ID
import random
import logging
from urllib.parse import urlparse
import os

# TODO: Change logging level
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

# Configure Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless=new")
chrome_options.add_argument("--start-maximized")
chrome_options.add_argument("--disable-extensions")
chrome_options.add_argument("--disable-gpu")
chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

class WebDriverPool:
    def __init__(self, max_drivers=10):
        self.max_drivers = max_drivers
        self.drivers = asyncio.Queue()
        self.count = 0
        self.semaphore = asyncio.Semaphore(max_drivers)

    async def get_driver(self):
        async with self.semaphore:
            if self.drivers.empty():
                if self.count < self.max_drivers:
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                    self.count += 1
                    return driver
                else:
                    return await self.drivers.get()
            else:
                return await self.drivers.get()

    async def return_driver(self, driver):
        await self.drivers.put(driver)

    async def cleanup(self):
        while not self.drivers.empty():
            driver = await self.drivers.get()
            driver.quit()

driver_pool = WebDriverPool(max_drivers=10)

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, str):
            return obj.encode('utf-8').decode('unicode_escape')
        return super().default(obj)

async def load_excel_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        return xls
    except Exception as e:
        logger.error(f"Error loading Excel file: {e}")
        return None

async def extract_relevant_html(soup, multiplier=1):
    body = soup.find('body')
    if not body:
        return ""
    
    lines = body.prettify().split('\n')
    relevant_parts = []
    
    i = 0
    while i < len(lines):
        if 'softball' in lines[i].lower():
            start = max(0, i - (10 * multiplier))
            end = min(len(lines), i + (21 * multiplier))
            part = lines[start:end]
            
            while True:
                coach_index = next((idx for idx, line in enumerate(part) if 'coach' in line.lower()), -1)
                if coach_index == -1 or end >= len(lines):
                    break
                end = min(len(lines), end + (15 * multiplier))
                part = lines[start:end]
            
            relevant_parts.append('\n'.join(part))
            i = end
        else:
            i += 1
    
    return '\n'.join(relevant_parts)

async def gemini_based_scraping(url, school_name):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.101 Safari/537.36'
    ]
    
    headers = {'User-Agent': random.choice(user_agents)}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    html_content = await response.text()
                    soup = BeautifulSoup(html_content, 'html.parser')
                else:
                    logger.warning(f"Failed to fetch {url}. Status code: {response.status}")
                    return None, False, 0, 0
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None, False, 0, 0

    for attempt in range(4):  # 4 attempts: 1x, 2x, 4x, and full body
        try:
            if attempt < 3:
                relevant_html = await extract_relevant_html(soup, multiplier=2**attempt)
            else:
                relevant_html = soup.find('body').prettify()

            genai.configure(api_key=api_key_manager.get_next_key())
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = f"""
            Analyze the HTML content of the coaching staff webpage for {school_name} and extract information ONLY for softball *coaches* (head coach and assistant coaches - sometimes you'll see interim coaches too. Include those). They will typically be found under a softball section, and will usually only be 3-4 in number. Do not include coaches from other sports or general staff members. Extract the following information for each softball coach:
            - Name
            - Title
            - Email address (if available)
            - Phone number (If available. Sometimes it will be in the section heading (eg:Softball - Phone: 828-262-7310))
            - Twitter/X handle (if available)
            Note: Phone number is always 10 digits. If some part is in the section heading and some part is in the row for the particular coach, piece together the information to find the full phone number.
            Determine if the scraping was successful or not. If not, provide a reason from the following options:
            - broken link (ie, 404 or page doesn't contain required data)
            - bot detection (ie, verify you're a human, captcha, that sort of stuff)
            - incomplete data (only some of the fields are present on the screen and the rest require additional clicks)
            - other 
            Format the output as a JSON string with the following structure:
            {{
                "success": true/false,
                "reason": "reason for failing to scrape data" (or null if success),
                "coachingStaff": [
                    {{
                        "name": "...",
                        "title": "...",
                        "email": null,
                        "phone": null,
                        "twitter": null
                    }},
                    ...
                ]
            }}
            If you can find any softball coaching staff information, even if incomplete, set "success" to true and include the available data. If no softball coaches are found, set "success" to false and provide the reason "no softball coaches found".
            Important: Ensure all names, including those with non-English characters, are preserved exactly as they appear in the HTML. Do not escape or modify any special characters in names or other fields.
            The response should be a valid JSON string only, without any additional formatting or markdown syntax.
            """

            token_response = model.count_tokens(prompt + relevant_html)
            input_tokens = token_response.total_tokens

            response = await model.generate_content_async([prompt, relevant_html])
            
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
                if result['success']:
                    return result, result['success'], input_tokens, output_tokens
                elif attempt == 3:  # If it's the last attempt and still not successful
                    return result, result['success'], input_tokens, output_tokens
            except json.JSONDecodeError:
                if attempt == 3:
                    logger.error(f"Failed to parse JSON from Gemini response for {school_name} after all attempts")
                    return None, False, input_tokens, output_tokens
        except Exception as e:
            if attempt == 3:
                logger.error(f"Error in Gemini-based scraping for {school_name} after all attempts: {str(e)}")
                return None, False, 0, 0

    return None, False, 0, 0  # This line should never be reached, but it's here for completeness

async def process_school(school_data, url_column):
    url = school_data[url_column]
    school_name = school_data['School']
    max_retries = 3
    base_delay = 5  # seconds

    if pd.notna(url):
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing {school_name} (URL: {url}) - Attempt {attempt + 1}")
                result, success, input_tokens, output_tokens = await gemini_based_scraping(url, school_name)
                total_tokens = input_tokens + output_tokens
                logger.info(f"Tokens used for {school_name} {url_column}: {total_tokens}")
                
                if success and result:
                    logger.info(f"Successfully scraped data for {school_name}")
                    return {
                        'school': school_name,
                        'url': url,
                        'success': True,
                        'coachingStaff': result['coachingStaff'],
                        'input_tokens': input_tokens,
                        'output_tokens': output_tokens,
                        'total_tokens': total_tokens
                    }
                else:
                    logger.warning(f"Scraping failed for {school_name} - Attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        await asyncio.sleep(delay)
                    else:
                        return {
                            'school': school_name,
                            'url': url,
                            'success': False,
                            'reason': result['reason'] if result else 'Unknown error after retries',
                            'input_tokens': input_tokens,
                            'output_tokens': output_tokens,
                            'total_tokens': total_tokens
                        }
            except Exception as e:
                logger.error(f"Error in processing {school_name}: {str(e)} - Attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    await asyncio.sleep(delay)
                else:
                    return {
                        'school': school_name,
                        'url': url,
                        'success': False,
                        'reason': f'Exception: {str(e)}',
                        'input_tokens': 0,
                        'output_tokens': 0,
                        'total_tokens': 0
                    }
    else:
        logger.info(f"Skipping {school_name} - No URL provided")
        return {
            'school': school_name,
            'url': 'N/A',
            'success': False,
            'reason': 'No URL provided',
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0
        }

async def process_sheet(sheet_name, df):
    all_results = []
    total_tokens_used = 0

    semaphore = asyncio.Semaphore(4)  # Limit to 4 concurrent requests

    async def process_with_semaphore(row, url_column):
        async with semaphore:
            return await process_school(row, url_column)

    for url_column in ['Staff Directory']:
        logger.info(f"\nProcessing {url_column} URLs for sheet: {sheet_name}")
        
        tasks = [process_with_semaphore(row, url_column) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)

        successful_scrapes = sum(1 for r in results if r['success'])
        failed_scrapes = len(results) - successful_scrapes
        tokens_used = sum(r['total_tokens'] for r in results)
        total_tokens_used += tokens_used

        logger.info(f"\nResults for {sheet_name} - {url_column}:")
        logger.info(f"Successful scrapes: {successful_scrapes}")
        logger.info(f"Failed scrapes: {failed_scrapes}")
        logger.info(f"Tokens used: {tokens_used}")

        save_results(results, f"scraping-results/{sheet_name}_{url_column}_results.json")
        save_failed_schools(results, f"scraping-results/{sheet_name}_{url_column}_failed_schools.txt")

        all_results.extend(results)

    logger.info(f"\nTotal tokens used for {sheet_name}: {total_tokens_used}")
    return all_results, total_tokens_used

def save_results(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=CustomJSONEncoder, ensure_ascii=False, indent=2)

def save_failed_schools(results, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    failed_schools = [f"{r['school']}: {r['url']} - Reason: {r.get('reason', 'Unknown')}" for r in results if not r['success']]
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(failed_schools))

async def main():
    global api_key_manager
    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-2\Freelancer_Data_Mining_Project.xlsx"
    
    api_key_manager = APIKeyManager(GEMINI_API_KEYS)
    total_tokens_used = 0
    
    try:
        xls = await load_excel_data(input_file)
        if xls is not None:
            for sheet_name in xls.sheet_names:
                # if sheet_name != "JUCO - USCAA": # -------------------- remove this later
                #     continue
                logger.info(f"\nProcessing sheet: {sheet_name}")
                df = pd.read_excel(xls, sheet_name=sheet_name)
                _, sheet_tokens = await process_sheet(sheet_name, df)
                total_tokens_used += sheet_tokens
            logger.info(f"\nTotal tokens used across all sheets: {total_tokens_used}")
        else:
            logger.error("Failed to load Excel file. Exiting.")
    finally:
        await driver_pool.cleanup()

if __name__ == "__main__":
    asyncio.run(main())

# TODO: Implement timing - ie how long does it take to scrape a sheet
# TODO: modify extract_relevant_html to further reduce token usage https://claude.ai/chat/f1e942f4-c1bd-414a-ae1b-fe7f2273ec1f
# TODO: create more api keys