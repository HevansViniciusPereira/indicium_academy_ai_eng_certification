import os
#import json
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
#import re
from typing import List, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.pdfgen import canvas
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
# from googlesearch import search # For news, you'd use a service like Serper or similar


def get_csv_file_details(url: str) -> List[Dict[str, str]]:
    """
    Scrapes a webpage URL, finds all links ending in .csv, and extracts their details.
    
    Args:
        url (str): The URL of the webpage to scrape.
                          
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing the file name and the full URL.
    """
    results = []
    
    try:
        # Fetch the HTML content
        print(f"Fetching content from: {url}")
        headers = {'User-Agent': 'AgenticCSVScraper/1.0'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        
        # Parsing the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        csv_links = []
        # Finding all <a> tags (links)
        links = soup.find_all('a', href=True)
        # Finding the csv link
        for link in links:
            if link.text.strip()[:5] == "https":
                csv_links.append(link.text.strip())
        
        if not csv_links:
            return [{"status": "No CSV files found.", "details": "The scraper did not find any links ending in .csv on the page."}]
            
        return csv_links

    except requests.exceptions.RequestException as e:
        return [{"status": "Web Scraping Failed", "details": f"Network/HTTP Error: {e}"}]
    except Exception as e:
        return [{"status": "Parsing Error", "details": f"An unexpected error occurred: {e}"}]

# Open DataSUS base URL
url = "https://opendatasus.saude.gov.br/dataset/srag-2021-a-2024/resource/20c49de3-ddc3-4b76-a942-1518eaae9c91"

###########################################################
# URL of data in CSV
url_default = get_csv_file_details(url)
###########################################################

@tool
def download_file_if_missing(
        url: str = url_default,
        local_path: str = "../data/"
        ) -> str:
    """
    Downloads a file from a specified URL only if it does not already exist 
    at the given local_path.
    
    Args:
        url (str): The URL of the file to download.
        local_path (str): The full path including the filename where the 
                          file should be saved.
                          
    Returns:
        str: A status message indicating success, failure, or if the file was skipped.
    """
    
    # Checking local existence
    if os.path.exists(local_path):
        return f"✅ File already exists locally at: {local_path}. Download skipped."

    print(f"File not found locally. Attempting download from: {url}")
    
    # Attempting download
    try:
        # Creating directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Use streaminging to handle potentially large files
        with requests.get(url, stream=True) as r:
            r.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return f"🎉 Successfully downloaded and saved file to: {local_path}"
        
    except requests.exceptions.RequestException as e:
        return f"❌ Download failed due to a network/HTTP error: {e}"
    except Exception as e:
        return f"❌ An unexpected error occurred during file saving: {e}"

@tool
def analyze_csv(file_path: str, focus_column: str) -> str:
    """Reads a CSV, calculates descriptive statistics, and finds key trends."""
    try:
        df = pd.read_csv(file_path)
        stats = df[focus_column].describe().to_string()
        key_metrics = f"Descriptive Statistics for {focus_column}:\n{stats}"
        # Simple trend check (e.g., is the mean increasing/decreasing over a 'Date' column)
        return key_metrics
    except Exception as e:
        return f"Error during data analysis: {e}"


# Graphics Tool
def generate_case_time_series_charts(file_path: str, date_col: str = 'Date', case_col: str = 'Cases') -> str:
    """
    Generates two time-series charts from a CSV: 
    1) Cases in the last 30 days.
    2) Cases in the last 12 months.

    Args:
        file_path (str): Path to the CSV file containing historical data.
        date_col (str): The name of the column containing date/time information.
        case_col (str): The name of the column containing case counts.
                          
    Returns:
        str: A status message listing the paths of the generated image files.
    """
    
    output_dir = "output/graphics"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Load and prepare data
        df = pd.read_csv(file_path)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()

        # 2. Define cutoff dates based on the latest date in the data
        latest_date = df.index.max()
        cutoff_30d = latest_date - pd.Timedelta(days=30)
        cutoff_12m = latest_date - pd.DateOffset(years=1)

        # --- Plot 1: Last 30 Days ---
        df_30d = df.loc[df.index >= cutoff_30d]
        path_30d = os.path.join(output_dir, "cases_last_30_days.png")
        
        plt.figure(figsize=(10, 5))
        df_30d[case_col].plot(kind='line', marker='o', linestyle='-')
        plt.title(f'Case Trends: Last 30 Days (ending {latest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('Date')
        plt.ylabel(f'Number of {case_col}')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path_30d)
        plt.close()

        # --- Plot 2: Last 12 Months ---
        df_12m = df.loc[df.index >= cutoff_12m]
        path_12m = os.path.join(output_dir, "cases_last_12_months.png")

        # Resample data to monthly sums for a cleaner 12-month view
        df_monthly = df_12m[case_col].resample('M').sum()
        
        plt.figure(figsize=(12, 6))
        df_monthly.plot(kind='bar', color='skyblue')
        plt.title(f'Monthly Case Totals: Last 12 Months (ending {latest_date.strftime("%Y-%m")})')
        plt.xlabel('Month')
        plt.ylabel(f'Total {case_col} (Monthly)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path_12m)
        plt.close()
        
        return f"🎉 Successfully generated and saved two charts:\n1. 30-Day Trend: {path_30d}\n2. 12-Month Trend: {path_12m}"

    except FileNotFoundError:
        return f"❌ Error: CSV file not found at {file_path}"
    except KeyError as e:
        return f"❌ Error: Required column {e} not found in the CSV file."
    except Exception as e:
        return f"❌ An unexpected error occurred during chart generation: {e}"


##########################################################
# Initialize the Serper API wrapper as the search tool
# serper_search = GoogleSerperAPIWrapper(
#     # Optional: set search type to 'news' for better results relevant to the report
#     type="news"
# )


# Extracting the date of the data publication
end_date = url_default[0].split(".csv")[0][-10:]
# Getting the date 30 days before data publication
start_date = datetime.strptime(end_date, "%d-%m-%Y")  - timedelta(days=30)
start_date = start_date.strftime("%d-%m-%Y")
###########################################################


# # News Search Tool
# @tool
# def search_online_news(
#     query: str = "Síndrome Respiratória Aguda Grave",
#     num_results: int = 5,
#     start_date: str = start_date,
#     end_date: str = end_date) -> str:
#     """
#     Performs a real-time news search using the Serper API for relevant articles.
    
#     Args:
#         query (str): The search query to find news about.
#         num_results (int): The maximum number of results to fetch (default is 5).
        
#     Returns:
#         str: A JSON string containing the summarized news results.
#     """
#     print(f"Searching news for: {query} after:{start_date} before:{end_date} (Max {num_results} results)")
    
#     try:
#         # Executing the search
#         raw_results: Dict = serper_search.results(query, num_results=num_results)

#         # Extracting and formatting key information from the 'news' array
#         news_items = []
#         if 'news' in raw_results:
#             for item in raw_results['news']:
#                 # Ensure the data is clean and relevant
#                 news_items.append({
#                     "title": item.get('title', 'N/A'),
#                     "snippet": item.get('snippet', 'No summary available.'),
#                     "source": item.get('source', 'N/A'),
#                     "link": item.get('link', 'N/A')
#                 })
        
#         # Returning a clean, parsable JSON string for the LLM to synthesize
#         if news_items:
#             return json.dumps(news_items, indent=2)
#         else:
#             return "No relevant news articles were found for the query."

#     except Exception as e:
#         return f"Error executing Serper news search: {e}"

# PDF Generation Tool
def create_pdf_report(
        content_sections: list,
        image_paths: list,
        title: str = f"Análise de Quantidade de Casos de SRAG entre {start_date} and {end_date}",
        output_file: str = "../outputs/SRAG_report.pdf"
        ) -> str:
    """Generates a professional PDF document from provided text and images."""

    try:
        c = canvas.Canvas(output_file)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(72, 800, title)
        
        y_position = 780
        for section in content_sections:
            c.setFont("Helvetica-Bold", 12)
            c.drawString(72, y_position, section['heading'])
            y_position -= 15
            
            c.setFont("Helvetica", 10)
            text = c.beginText(72, y_position)
            text.textLines(section['body'])
            c.drawText(text)
            y_position -= (len(section['body'].split('\n')) * 12 + 20) # Estimate space
        
        # Image inclusion
        for i, img_path in enumerate(image_paths):
            c.drawImage(img_path, 72, 100 + i * 150, width=400, height=150)

        c.save()
        return f"PDF report generated successfully at: {output_file}"
    except Exception as e:
        return f"Error generating PDF: {e}"


# Calculating metrics
def calculate_epidemiology_rates(
    current_cases: int,
    previous_cases: int,
    current_deaths: int,
    cumulative_confirmed_cases: int,
    icu_occupancy: int,
    icu_capacity: int,
    total_population: int,
    people_vaccinated: int
) -> Dict[str, Union[float, str]]:
    """
    Calculates four key epidemiological rates based on provided data.

    Args:
        current_cases (int): The number of new cases reported in the current period (e.g., today).
        previous_cases (int): The number of new cases reported in the previous period (e.g., yesterday).
        current_deaths (int): The number of new deaths reported in the current period.
        cumulative_confirmed_cases (int): The total, cumulative number of confirmed cases.
        icu_occupancy (int): The number of occupied ICU beds by case patients.
        icu_capacity (int): The total number of available ICU beds.
        total_population (int): The total population of the area.
        people_vaccinated (int): The number of people fully vaccinated.

    Returns:
        Dict[str, Union[float, str]]: A dictionary containing the four calculated rates, 
                                      or error messages if denominators are zero.
    """
    
    results = {}

    # Rate of Case Increase (Percentage Change)
    # Measures the acceleration/deceleration of the epidemic wave.
    if previous_cases > 0:
        rate_increase = ((current_cases - previous_cases) / previous_cases) * 100
        results['case_increase_rate (%)'] = round(rate_increase, 2)
    else:
        results['case_increase_rate (%)'] = "N/A (Previous cases were zero)"

    # Death Rate
    if cumulative_confirmed_cases > 0:
        death_rate = (current_deaths / cumulative_confirmed_cases) * 100
        results['death_rate_cfr (%)'] = round(death_rate, 2)
    else:
        results['death_rate_cfr (%)'] = "N/A (Total confirmed cases were zero)"

    # ICU Occupancy Rate
    if icu_capacity > 0:
        icu_rate = (icu_occupancy / icu_capacity) * 100
        results['icu_occupancy_rate (%)'] = round(icu_rate, 2)
    else:
        results['icu_occupancy_rate (%)'] = "N/A (ICU capacity is zero)"

    # Population Vaccination Rate
    if total_population > 0:
        vac_rate = (people_vaccinated / total_population) * 100
        results['population_vaccination_rate (%)'] = round(vac_rate, 2)
    else:
        results['population_vaccination_rate (%)'] = "N/A (Total population is zero)"

    return results
