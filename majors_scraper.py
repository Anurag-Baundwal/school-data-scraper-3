# college majors scraper
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import logging
import asyncio
import aiohttp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random
import base64
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import google.generativeai as genai
import json
import re
from pymongo import MongoClient
from datetime import datetime
import shutil
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# MongoDB setup
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "college_sports_db"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
divisions_collection = db["divisions"]
scraping_logs_collection = db["scraping_logs"]

#------------------- API KEYS ----------------------------------------
# Python code to load and parse the environment variables:
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Parse Gemini API keys as a list
GEMINI_API_KEYS = os.getenv('GEMINI_API_KEYS').split(',')

# Load other environment variables
OXYLABS_USERNAME = os.getenv('OXYLABS_USERNAME')
OXYLABS_PASSWORD = os.getenv('OXYLABS_PASSWORD')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
SEARCH_ENGINE_ID = os.getenv('SEARCH_ENGINE_ID')

#--------------------------------------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# List of user agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

def google_search(query, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=query, cx=cse_id, **kwargs).execute()
    return res['items']

def extract_majors(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    majors = []

    # Type 1: div with style="padding:15px;"
    majors_div = soup.find('div', style="padding:15px;")
    if majors_div:
        majors = [h3.text.strip() for h3 in majors_div.find_all('h3')]

    # Type 2: table with id='majortable'
    if not majors:
        majors_table = soup.find('table', id='majortable')
        if majors_table:
            for row in majors_table.find_all('tr')[1:]:  # Skip header row
                cells = row.find_all('td')
                if cells:
                    majors.append(cells[0].text.strip())

    # Type 3: span elements with class="parent-line"
    if not majors:
        parent_lines = soup.find_all('span', class_="parent-line")
        for line in parent_lines:
            major = line.find('p')
            if major:
                majors.append(major.text.strip())

    # Type 4: Look for lists (ul, ol) containing potential majors
    if not majors:
        for list_elem in soup.find_all(['ul', 'ol']):
            items = list_elem.find_all('li')
            if len(items) > 5:  # Assuming a list of majors would have more than 5 items
                majors.extend([item.text.strip() for item in items])

    # Type 5: Look for header elements (h1, h2, h3) containing "major" or "program"
    if not majors:
        headers = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for header in headers:
            if 'major' in header.text.lower() or 'program' in header.text.lower():
                next_elem = header.find_next_sibling()
                if next_elem and next_elem.name in ['ul', 'ol']:
                    majors.extend([item.text.strip() for item in next_elem.find_all('li')])

    # If still no majors found, try to find any p tags within the main content
    if not majors:
        content_div = soup.find('div', id='MajorsOffered')
        if content_div:
            majors = [p.text.strip() for p in content_div.find_all('p') if p.text.strip()]

    return list(set(majors))  # Remove duplicates

async def take_screenshot_async(url, context, max_retries=5):
    user_agent = random.choice(USER_AGENTS)
    
    for attempt in range(max_retries):
        try:
            page = await context.new_page()
            await page.set_extra_http_headers({"User-Agent": user_agent})
            await page.goto(url, wait_until='networkidle', timeout=60000)
            await asyncio.sleep(random.uniform(3, 8))
            screenshot = await page.screenshot(full_page=True, type='jpeg', quality=100)
            await page.close()
            return screenshot, None
        except PlaywrightTimeoutError:
            if attempt == max_retries - 1:
                return None, "Timeout error after multiple attempts"
            await asyncio.sleep(random.uniform(5, 10))
        except Exception as e:
            return None, str(e)
        finally:
            if 'page' in locals():
                await page.close()
    return None, "Max retries reached"

async def extract_majors_visual(screenshot, max_retries=3):
    for attempt in range(max_retries):
        try:
            api_key = random.choice(GEMINI_API_KEYS)
            genai.configure(api_key=api_key)

            model = genai.GenerativeModel('gemini-1.5-flash')
            
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(screenshot).decode('utf-8')
                }
            ]
            
            prompt = """
            Analyze the screenshot of a college majors webpage and extract the following information:
            - List of all undergraduate majors or programs offered
            
            Be thorough in your search. Look for any text that might represent a major or program of study.
            If you can't find an explicit list of majors, look for related terms like 'programs', 'degrees', or 'fields of study'.
            
            Format the output as a JSON string with the following structure:
            {
                "success": true/false,
                "reason": "reason for failure" (or null if success),
                "majors": [
                    "Major 1",
                    "Major 2",
                    ...
                ]
            }
            
            If you can't find any majors, set success to false and provide a reason.
            """
            
            response = model.generate_content([prompt, image_parts[0]])
            
            if not response.text:
                raise ValueError("Empty response from Gemini API")

            # Clean up the response text
            response_text = response.text.strip()
            # Remove any markdown code block indicators
            response_text = re.sub(r'```json\s*|\s*```', '', response_text)
            
            # Parse the JSON
            data = json.loads(response_text)
            
            return data
        except Exception as e:
            logger.warning(f"Visual scraping attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                return {
                    "success": False,
                    "reason": f"Error in visual scraping after {max_retries} attempts: {str(e)}",
                    "majors": []
                }
            await asyncio.sleep(random.uniform(1, 3))

async def visual_scrape_fallback(url, school, context):
    screenshot, error = await take_screenshot_async(url, context)
    if error or not screenshot:
        # Perform Google search
        search_query = f"{school} undergraduate majors programs"
        try:
            search_results = google_search(search_query, GOOGLE_API_KEY, SEARCH_ENGINE_ID, num=10)
            for result in search_results:
                search_url = result['link']
                search_screenshot, search_error = await take_screenshot_async(search_url, context)
                if not search_error and search_screenshot:
                    majors_data = await extract_majors_visual(search_screenshot)
                    if majors_data["success"] and majors_data["majors"]:
                        return url, school, majors_data["majors"], None, None, search_url
                else:
                    # Use Playwright as a backup
                    try:
                        async with async_playwright() as p:
                            browser = await p.chromium.launch()
                            page = await browser.new_page()
                            await page.goto(search_url, wait_until="networkidle", timeout=60000)
                            search_screenshot = await page.screenshot(full_page=True, type='jpeg', quality=100)
                            await browser.close()
                            
                            if search_screenshot:
                                majors_data = await extract_majors_visual(search_screenshot)
                                if majors_data["success"] and majors_data["majors"]:
                                    return url, school, majors_data["majors"], None, None, search_url
                    except Exception as playwright_error:
                        logger.warning(f"Playwright backup failed for {search_url}: {str(playwright_error)}")
            
            return url, school, None, f"Visual scraping failed for all search results", screenshot, None
        except HttpError as e:
            return url, school, None, f"Google search API error: {str(e)}", screenshot, None
    
    majors_data = await extract_majors_visual(screenshot)
    
    if majors_data["success"] and majors_data["majors"]:
        return url, school, majors_data["majors"], None, None, url
    else:
        return url, school, None, f"Visual scraping failed: {majors_data['reason']}", screenshot, None
        
async def fetch_url(session, url, school, context, max_retries=5, base_timeout=10):
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.google.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }
    
    for attempt in range(max_retries):
        try:
            timeout = base_timeout * (2 ** attempt)
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status == 200:
                    content = await response.text()
                    majors = extract_majors(content)
                    if majors:
                        return url, school, majors, None, None, url
                    else:
                        # If no majors found, try visual scraping
                        return await visual_scrape_fallback(url, school, context)
                else:
                    # For any non-200 status, try visual scraping
                    return await visual_scrape_fallback(url, school, context)
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                return await visual_scrape_fallback(url, school, context)
        except Exception as e:
            if attempt == max_retries - 1:
                return await visual_scrape_fallback(url, school, context)
        
        await asyncio.sleep(random.uniform(1, 3))

    return await visual_scrape_fallback(url, school, context)

async def process_chunk(chunk, context):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for _, row in chunk.iterrows():
            url = row['Undergraduate Majors URL']
            school = row['School']
            if pd.notna(url):
                tasks.append(fetch_url(session, url, school, context))
        return await asyncio.gather(*tasks)

async def process_chunk_wrapper(chunk):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        results = await process_chunk(chunk, context)
        await browser.close()
        return results

def save_screenshot(screenshot, school, url):
    if screenshot:
        filename = f"failed_screenshots/{school.replace(' ', '_')}_{url.split('//')[-1].replace('/', '_')}.jpg"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(screenshot)
        logger.info(f"Screenshot saved for {school}: {filename}")

def save_results(all_majors, failed_urls, sheet_name, division_name, search_used, google_search_results):
    # Update MongoDB
    for school, (original_url, majors, scraped_url) in all_majors.items():
        divisions_collection.update_one(
            {"name": division_name, "colleges.name": school},
            {
                "$set": {
                    "colleges.$.majors.list": majors,
                    "colleges.$.majors.last_updated": datetime.now()
                }
            }
        )

        # Log successful scraping
        scraping_logs_collection.insert_one({
            "division": division_name,
            "college_name": school,
            "data_type": "majors",
            "timestamp": datetime.now(),
            "status": "success",
            "url_scraped": scraped_url
        })

    # Log failed scraping attempts
    for school, url, reason in failed_urls:
        scraping_logs_collection.insert_one({
            "division": division_name,
            "college_name": school,
            "data_type": "majors",
            "timestamp": datetime.now(),
            "status": "failed",
            "reason": reason,
            "url_scraped": url,
            "google_search_used": search_used.get(school, False)
        })

    # Save to Excel and CSV
    results_data = []
    max_majors = max([len(majors) for _, majors, _ in all_majors.values()] + [0])
    columns = ['School', 'Num_Majors', 'Scraping_Method', 'URL_Used'] + [f'Major_{i+1}' for i in range(max_majors)]

    for school, (original_url, majors, scraped_url) in all_majors.items():
        sorted_majors = sorted(majors)
        row_data = {
            'School': school, 
            'Num_Majors': len(sorted_majors),
            'Scraping_Method': 'Google Search' if search_used.get(school, False) else 'Provided URL',
            'URL_Used': scraped_url
        }
        for i, major in enumerate(sorted_majors, 1):
            row_data[f'Major_{i}'] = major
        results_data.append(row_data)

    for school, url, _ in failed_urls:
        if school not in [row['School'] for row in results_data]:
            results_data.append({
                'School': school, 
                'Num_Majors': 0,
                'Scraping_Method': 'Failed',
                'URL_Used': url
            })

    results_df = pd.DataFrame(results_data, columns=columns)
    results_df = results_df.sort_values('School')

    # Save to a single Excel file with multiple sheets
    excel_file = 'scraped_majors_data.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a' if os.path.exists(excel_file) else 'w') as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    # Save to CSV
    results_df.to_csv(f'scraped_majors_data_{sheet_name}.csv', index=False)

    # Save failed URLs to Excel and text file
    failed_df = pd.DataFrame(failed_urls, columns=['School', 'URL', 'Reason'])
    failed_df['Google_Search_URL'] = failed_df['School'].map(google_search_results)
    failed_df = failed_df.sort_values('School')
    
    with pd.ExcelWriter('failed_urls.xlsx', engine='openpyxl', mode='a' if os.path.exists('failed_urls.xlsx') else 'w') as writer:
        failed_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    with open("failed_urls.txt", "a") as f:
        f.write(f"\n{'=' * 50}\n")
        f.write(f"Failed URLs for sheet: {sheet_name}\n")
        f.write(f"{'=' * 50}\n\n")
        for _, row in failed_df.iterrows():
            f.write(f"{row['School']} - {row['URL']} - {row['Reason']}\n")
            if pd.notna(row['Google_Search_URL']):
                f.write(f"  Google Search URL: {row['Google_Search_URL']}\n")
        f.write("\n")

async def main():
    # Clear existing output files and screenshots
    output_files = ['scraped_majors_data.xlsx', 'failed_urls.xlsx', 'failed_urls.txt']
    for file in output_files:
        if os.path.exists(file):
            os.remove(file)
    
    if os.path.exists('failed_screenshots'):
        shutil.rmtree('failed_screenshots')
    os.makedirs('failed_screenshots', exist_ok=True)

    input_file = r"C:\Users\dell3\source\repos\school-data-scraper-3\Freelancer_Data_Mining_Project.xlsx"   
    xls = pd.ExcelFile(input_file)
    
    for sheet_name in xls.sheet_names:
        all_majors = {}
        failed_urls = []
        search_used = {}
        google_search_results = {}  # New dictionary to store Google search results
        
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Processing sheet: {sheet_name}")
        logger.info(f"{'=' * 50}\n")
        
        df = pd.read_excel(input_file, sheet_name=sheet_name)
        
        chunk_size = 10
        chunks = [df[i:i+chunk_size] for i in range(0, df.shape[0], chunk_size)]
        
        for chunk_index, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {chunk_index + 1}/{len(chunks)}")
            retry_count = 0
            max_retries = 3
            
            while retry_count < max_retries:
                try:
                    results = await process_chunk_wrapper(chunk)
                    for url, school, majors, error, screenshot, scraped_url in results:
                        if majors:
                            all_majors[school] = (url, majors, scraped_url)
                            logger.info(f"Processed: {school} - Found {len(majors)} majors")
                            search_used[school] = url != scraped_url
                        else:
                            failed_urls.append((school, url, error))
                            logger.warning(f"Failed: {school} - {url} - {error}")
                            save_screenshot(screenshot, school, url)
                            search_used[school] = True
                            
                            # Log Google search results for failed URLs
                            if scraped_url and scraped_url != url:
                                google_search_results[school] = scraped_url
                    
                    # If successful, break the retry loop
                    break
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_index + 1}: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying chunk {chunk_index + 1} (Attempt {retry_count + 1})")
                        await asyncio.sleep(random.uniform(5, 10))
                    else:
                        logger.error(f"Failed to process chunk {chunk_index + 1} after {max_retries} attempts")
            
            # Add a delay between chunks to avoid overwhelming the server
            await asyncio.sleep(random.uniform(2, 5))
        
        save_results(all_majors, failed_urls, sheet_name, sheet_name, search_used, google_search_results)
        
        logger.info(f"\nFinished processing sheet: {sheet_name}")
        logger.info(f"Total schools processed: {len(all_majors) + len(failed_urls)}")
        logger.info(f"Successful scrapes: {len(all_majors)}")
        logger.info(f"Failed scrapes: {len(failed_urls)}")
        logger.info(f"{'=' * 50}\n")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds")