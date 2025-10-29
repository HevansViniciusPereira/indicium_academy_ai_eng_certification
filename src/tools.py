import os
import json
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import List, Dict, Union
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from dotenv import load_dotenv


load_dotenv()

server_api_key = os.getenv("SERPER_API_KEY")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# Initialize the Serper API wrapper as the search tool
serper_search = GoogleSerperAPIWrapper(type="news")


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
        print(f"Procurando dados mais recentes no endereço {url}")
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
            return [{"status": "Nenhum arquivo CSV encontrado.", "details": "O scraper não encontrou nenhum link de arquivo .csv na página raiz."}]
            
        print(f"Dados mais recentes encontrados no endereço {csv_links[0]}")
        return csv_links[0]

    except requests.exceptions.RequestException as e:
        return [{"status": "Web Scraping Falhou", "details": f"Network/HTTP Error: {e}"}]
    except Exception as e:
        return [{"status": "Parsing Error", "details": f"Um erro inesperado ocorreu: {e}"}]

def getting_dates(url_str: str) -> str:
    """Returns a start_date and an end_date strings in format %d-%m-%Y.
    Examples:
    https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2025/INFLUD25-20-10-2025.csv -> 20-10-2025
    https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SRAG/2024/INFLUD25-13-02-2024.csv -> 13-02-2024
    """
    # Extracting the publication date
    end_date = url_str.split(".csv")[0][-10:]
    # Getting the date 30 days before data publication
    start_date = datetime.strptime(end_date, "%d-%m-%Y") - timedelta(days=365)
    start_date = start_date.strftime("%d-%m-%Y")

    return start_date, end_date

def download_file_if_missing(
        download_url: str,
        local_path: str
        ) -> str:
    """
    Downloads a file from a specified URL only if it does not already exist 
    at the given local_path.
    
    Args:
        download_url (str): The URL of the file to download.
        local_path (str): The full path including the filename where the 
                          file should be saved.
                          
    Returns:
        str: A status message indicating success, failure, or if the file was skipped.
    """

    _, end_date = getting_dates(url_str=download_url)
    local_path = local_path + "SRAG-" + end_date + ".csv"

    # Checking local existence
    if os.path.exists(local_path):
        print(f"✅ File already exists locally at: {local_path}. Download skipped.")
        return None

    print(f"File not found locally. Attempting download from: {download_url}")
    
    try:
        response = requests.get(download_url)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(local_path, 'wb') as f:
            f.write(response.content)
        print(f"CSV downloaded successfully to {local_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading CSV: {e}")
        exit()
        
        print(f"🎉 Successfully downloaded and saved file to: {local_path}")
        return None
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Download failed due to a network/HTTP error: {e}")
        return None
    except Exception as e:
        print(f"❌ An unexpected error occurred during file saving: {e}")
        return None

# Graphics Tool
def generate_case_time_series_charts(local_path: str, date_col: str = 'DT_NOTIFIC') -> str:
    """
    Generates two time-series charts from a CSV: 
    1) Cases in the last 30 days.
    2) Cases in the last 12 months.

    Args:
        local_path (str): Path to folder containing the CSV file.
        date_col (str): The name of the column containing date/time information.
                          
    Returns:
        str: A status message listing the paths of the generated image files.
    """
    
    output_dir = "output/graphics"
    os.makedirs(output_dir, exist_ok=True)

    try:
        file_path = local_path + os.listdir(local_path)[0]
        df = pd.read_csv(file_path, delimiter=";", usecols=["NU_NOTIFIC","DT_NOTIFIC"])
        
        df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], format="%Y-%m-%d")  
    
        df = (
            df
            .dropna()
            .set_index(date_col)
            .groupby(by=date_col)
            .agg("count")
            .rename(columns={df.columns[0]: 'TotalCases'})
            .sort_index()
        )

        # 2. Define cutoff dates based on the latest date in the data
        latest_date = df.index.max()
        cutoff_30d = latest_date - pd.Timedelta(days=30)
        cutoff_12m = latest_date - pd.DateOffset(years=1)

        # Plot 1: Last 30 Days
        df_30d = df.loc[df.index >= cutoff_30d]
        df_30d = df_30d.reset_index()

        path_30d = os.path.join(output_dir, "cases_last_30_days.png")

        plt.figure(figsize=(12, 6))
        ax = df_30d.plot(
            x="DT_NOTIFIC",
            y="TotalCases",
            kind='line', 
            marker='o', 
            linestyle='-',
            ax=plt.gca()
        )
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.xticks(rotation=45, ha='center')

        plt.title(f'Case In The Last 30 Days (ending {latest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('')
        plt.ylabel(f'Number of Cases')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path_30d)
        plt.close()


        # Plot 2: Last 12 Months
        df_12m = df.loc[df.index >= cutoff_12m]
        path_12m = os.path.join(output_dir, "cases_last_12_months.png")

        # Resample data to monthly sums for a cleaner 12-month view
        df_monthly = df_12m["TotalCases"].resample('M').sum()
        df_monthly = df_monthly.reset_index()
        df_monthly['DT_NOTIFIC'] = df_monthly['DT_NOTIFIC'].dt.strftime('%m-%Y')
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df_monthly,
            x="DT_NOTIFIC",
            y="TotalCases"
        )
        ax.tick_params(left=False, bottom=False)
        ax.set(xlabel=None)
        ax.set(ylabel=None)
        ax.set(yticklabels=[])

        for container in ax.containers:
            ax.bar_label(container)

        plt.title(f'Total Monthly Case: Last 12 Months (ending {latest_date.strftime("%Y-%m")})')
        plt.tight_layout()
        sns.despine(left=True, right=True, bottom=True, top=True)
        plt.savefig(path_12m)
        plt.close()
        
        return f"🎉 Successfully generated and saved two charts:\n1. 30-Day Trend: {path_30d}\n2. 12-Month Trend: {path_12m}"

    except FileNotFoundError:
        return f"❌ Error: CSV file not found at {local_path}"
    except KeyError as e:
        return f"❌ Error: Required column {e} not found in the CSV file."
    except Exception as e:
        return f"❌ An unexpected error occurred during chart generation: {e}"

def search_online_news(
    query: str = "Síndrome Respiratória Aguda Grave",
    num_results: int = 5,
    start_date: str = None,
    end_date: str = None
) -> str:
    """
    Performs a real-time news search using the Serper API for relevant articles,
    saves the results to a JSON file, and returns the file path.
    
    Args:
        query (str): The search query to find news about.
        num_results (int): The maximum number of results to fetch.
        start_date (str): Date string for search filtering (if supported by serper_search).
        end_date (str): Date string for search filtering (if supported by serper_search).
                          
    Returns:
        str: The full file path of the saved JSON file, or an error message.
    """

    news_output_dir = "output/news"
    news_file_name = f"news_context_{end_date}.json"

    # Checking local existence
    if os.path.exists(news_output_dir):
        print(f"✅ News exists locally at: {news_output_dir}. Download skipped.")
        return None

    os.makedirs(news_output_dir, exist_ok=True)
    
    date_info = f" after:{start_date}" if start_date else ""
    date_info += f" before:{end_date}" if end_date else ""

    
    print(f"Searching news for: {query}{date_info} (Max {num_results} results)")
    
    try:
        # 1. Execute the search
        # Pass date filters if applicable. Serper uses custom parameters for date ranges.
        # We'll use the 'tbs' parameter (time before/since) which needs formatting (e.g., 'cdr:1,cd_min:1/1/2024,cd_max:1/31/2024')
        # Since the Serper wrapper doesn't directly expose start/end date, we'll keep the core call simple 
        # but acknowledge the LLM needs to format the query or use custom params if required.
        
        # For this modification, we assume the date filtering is handled either in the query 
        # string or via the search tool's advanced configuration (which we omit for brevity).
        
        raw_results: Dict = serper_search.results(query, num_results=num_results)

        # 2. Extract and format key information
        news_items = []
        if 'news' in raw_results and raw_results['news']:
            for item in raw_results['news']:
                news_items.append({
                    "title": item.get('title', 'N/A'),
                    "snippet": item.get('snippet', 'No summary available.'),
                    "source": item.get('source', 'N/A'),
                    "link": item.get('link', 'N/A')
                })
        
        # 3. Save the results to the specified path
        full_file_path = os.path.join(news_output_dir, news_file_name)
        
        if news_items:
            # Ensure the output directory exists
            os.makedirs(news_output_dir, exist_ok=True)
            
            # Save the structured news items as a JSON file
            with open(full_file_path, 'w', encoding='utf-8') as f:
                json.dump(news_items, f, indent=4, ensure_ascii=False)
                
            print(f"🎉 Relevant news context successfully saved to: {full_file_path}")
            return news_items
        else:
            return "No relevant news articles were found for the query and no file was saved."

    except Exception as e:
        return f"❌ Error executing Serper news search or saving file: {e}"

# Calculating metrics
def calculate_epidemiology_rates(
    local_path: str,
    selected_columns: str,
    end_date: str
) -> Dict[str, Union[float, str]]:
    """
    Calculates four key epidemiological rates based on provided data.

    Args:
        current_cases (int): The number of new cases reported in the current period (e.g., today).

    Returns:
        Dict[str, Union[float, str]]: A dictionary containing the four calculated rates, 
                                      or error messages if denominators are zero.
    """
    
    try:
        file_path = local_path + os.listdir(local_path)[0]
        df = pd.read_csv(file_path, delimiter=";", usecols=selected_columns)
        
        for col in selected_columns:
            if col[:2] == "DT":
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")

    except:
        print("Não foi possível ler o conjunto de dados!")
        return None

    results = {}

    ###############################################
    ## Rate of Case Increase (Percentage Change) ##
    ###############################################

    df_cases = df[["NU_NOTIFIC","DT_NOTIFIC"]]

    df_cases = (
        df_cases
        .dropna()
        .set_index("DT_NOTIFIC")
        .groupby(by="DT_NOTIFIC")
        .agg("count")
        .rename(columns={df_cases.columns[0]: 'TotalCases'})
        .sort_index()
    )

    # Resample data to monthly sums for a cleaner 12-month view
    df_monthly = df_cases["TotalCases"].resample('M').sum()
    df_monthly = df_monthly.reset_index()
    df_monthly['DT_NOTIFIC'] = df_monthly['DT_NOTIFIC'].dt.strftime('%m-%Y')

    previous_cases = df_monthly["TotalCases"].iloc[-2]
    current_cases = df_monthly["TotalCases"].iloc[-1]

    if previous_cases > 0:
        rate_increase = ((current_cases - previous_cases) / previous_cases) * 100
        results['case_increase_rate (%)'] = round(rate_increase, 2)
    else:
        results['case_increase_rate (%)'] = "N/A (Previous cases were zero)"

    ###############################################
    # Death Rate
    ###############################################

    df_deaths = df[["EVOLUCAO"]].dropna().query("EVOLUCAO == 1.0 | EVOLUCAO == 2.0")
    cumulative_confirmed_cases = df_deaths.shape[0]
    current_deaths = df_deaths.query("EVOLUCAO == 2.0").shape[0]

    if cumulative_confirmed_cases > 0:
        death_rate = (current_deaths / cumulative_confirmed_cases) * 100
        results['death_rate_cfr (%)'] = round(death_rate, 2)
    else:
        results['death_rate_cfr (%)'] = "N/A (Total confirmed cases were zero)"

    ###############################################
    # ICU Occupancy Rate
    ###############################################

    df_icu = df[["UTI", "DT_ENTUTI", "DT_SAIDUTI"]].dropna()

    icu_capacity = df_cases = (
        df_icu
        .set_index("DT_ENTUTI")
        .groupby(by="DT_ENTUTI")
        .agg("count")
        .rename(columns={df_cases.columns[0]: 'TotalCases'})
        .sort_index()
    )

    icu_capacity = 100
    icu_occupancy = 50

    if icu_capacity > 0:
        icu_rate = (icu_occupancy / icu_capacity) * 100
        results['icu_occupancy_rate (%)'] = round(icu_rate, 2)
    else:
        results['icu_occupancy_rate (%)'] = "N/A (ICU capacity is zero)"

    ###############################################
    # Population Vaccination Rate
    ###############################################

    df_vacina = df[["VACINA"]]

    total_population = df_vacina.shape[0]
    people_vaccinated = df_vacina.query("VACINA == 1.0").shape[0]

    if total_population > 0:
        vac_rate = (people_vaccinated / total_population) * 100
        results['population_vaccination_rate (%)'] = round(vac_rate, 2)
    else:
        results['population_vaccination_rate (%)'] = "N/A (Total population is zero)"

    file_path = f"output/rates/calc_rates_{end_date}.json"
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

    return results

def analyze_graphic(graphic_12_months_path: str, graphic_30_days_path: str):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        descriptions = []

        for path in [graphic_12_months_path, graphic_30_days_path]:
            prompt = f"""
                Você é um analista epidemiológico especialista.
                Descreva o gráfico {path} em um parágrafo, enfatizando momentos de mais 
                casos e de menos casos e estabelecendo isso com as estações do ano, se o gráfico 
                contiver informações mensais. Se o gráfico tiver informações diárias, elabore uma 
                descrição da situação recente e explore possíveis sazonalidades de curto prazo e 
                tente explicar esse comportamento.
                Escreva também sobre como este gráfico poderia ser utilizado pelo poder público 
                para planejar ações de saúde como, por exemplo, vacinação.

                Evite começar a resposta com 'Como analista epidemiológico'. Vá Direto para 
                a análise e seja sucinto.
            """

            result = llm.invoke(prompt)
            descriptions.append(result.content)

    except Exception as e:
        print(e)

    desc_12_months, desc_30_days = descriptions

    return desc_12_months, desc_30_days

def analyze_metrics(metrics: Dict):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        prompt = f"""
            Você é um analista epidemiológico especialista.
            Escreva um parágrafo sobre Síndrome Respiratória Aguda Grave usando os valores {metrics}.
            Vá Direto para a análise e seja sucinto.
        """

        result = llm.invoke(prompt)

    except Exception as e:
        print(e)

    return result.content

def create_content(news:str, query:str) -> str:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        new_prompt = f"""
            Você é um analista geopolítico e epidemiológico especialista.
            Sua tarefa é sintetizar conteúdo noticioso.
            O tema principal é: '{query}'. A seguir, é apresentado um array JSON 
            dos artigos de notícias mais recentes encontrados:
            --- INÍCIO DOS DADOS DE NOTÍCIAS ---
            {news}
            --- FIM DOS DADOS DE NOTÍCIAS ---
            
            Gere conteúdo para um relatório detalhado no formato Markdown, seguindo estas instruções: 
            1. Inicie com um cabeçalho principal: '## Principais Descobertas e Contexto para "{query}"'.
            2. Escreva um resumo executivo de um parágrafo (com menos de 5 frases).
            3. Crie uma seção '### Desenvolvimentos Recentes' com uma lista de 3 a 5 fatos/eventos-chave em formato de lista com marcadores, citando o título da fonte para cada item.
            4. Conclua com uma seção '### Perspectivas' contendo uma síntese em um parágrafo da tendência ou risco geral.
        """

        result = llm.invoke(new_prompt)
        contents = result.content.split("###")

        return contents

    except Exception as e:
        print(f"{e}")
        
        return None

# query="Notícias sobre Síndrome Respiratória Aguda Grave no Brasil."

# news = search_online_news(
#         query = query,
#         num_results = 5,
#         start_date = "24-10-2024",
#         end_date = "24-10-2025"
#     )

# create_content(news, query)


def generate_pdf_report(
        start_date: str,
        end_date: str,
        summary: str,
        recent_developments: str,
        perspectives: str,
        desc_12_months:str,
        desc_30_days:str,
        desc_metrics: str
        ) -> str:
    """
    Reads data from a JSON file, includes graphics from a folder, and 
    generates a single PDF report.

    Args:
        json_folder (str): Path to the folder containing JSON data files.
        graphic_folder (str): Path to the folder containing image files (PNG/JPG).
        output_path (str, optional): Full path for the output PDF. Defaults to report_output_path/Final_Report.pdf.
                          
    Returns:
        str: A status message with the full path of the generated PDF.
    """

    # Define the output directory and file name
    report_output_path = "output/reports"
    report_name = f"SRAG_Final_Report_{end_date}.pdf"
    #graphic_folder = "output/graphics"
    file_path_12months = "output/graphics/cases_last_12_months.png"
    file_path_30days = "output/graphics/cases_last_30_days.png"

    os.makedirs(report_output_path, exist_ok=True)
    output_path = os.path.join(report_output_path, report_name)

    styles = getSampleStyleSheet()
    story = []

    try:
        # Initialize PDF Document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        
        # Define custom styles
        styles.add(ParagraphStyle(name='TitleStyle', fontSize=16, spaceAfter=25, alignment=1))

        # Adding title
        story.append(Paragraph(f"Paranorama SRAG no período de {start_date} a {end_date}", styles['TitleStyle']))
        story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d-%m-%Y')}", styles['Normal']))
        story.append(Spacer(1, 20))

        # Adding paragraph
        story.append(Paragraph("Resumo Executivo", styles['Heading2']))
        story.append(Paragraph(f"{summary}"))
        story.append(Spacer(1, 12))
                
        # Adding 12 months graphic to the report
        if os.path.exists(file_path_12months):
            # Add Image (scale to fit page width, e.g., 5 inches wide)
            img = Image(file_path_12months, width=5 * 72, height=3 * 72) 
            story.append(img)
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"{desc_12_months}"))
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(f"Warning: Graphic file not found: {desc_12_months}", styles['Normal']))

        # Adding 30 days graphic to the report
        if os.path.exists(file_path_30days):
            # Add Image (scale to fit page width, e.g., 5 inches wide)
            img = Image(file_path_30days, width=5 * 72, height=3 * 72) 
            story.append(img)
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"{desc_30_days}"))
        else:
            story.append(Paragraph(f"Warning: Graphic file not found: {file_path_30days}", styles['Normal']))

        # Adding paragraph
        #story.append(Paragraph("Desenvolvimentos Recentes", styles['Heading2']))
        #story.append(Paragraph(f"{recent_developments}"))
        story.append(Paragraph(f"{desc_metrics}"))

        # Adding paragraph
        story.append(Paragraph("Perspectivas", styles['Heading2']))
        story.append(Paragraph(f"{perspectives}"))

        # Build the PDF
        doc.build(story)
        
        return f"✅ Report successfully generated and saved to: {output_path}"

    except Exception as e:
        return f"❌ Error generating PDF report: {e}"
