import os
import json
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from typing import List, Dict, Union, Tuple, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from dotenv import load_dotenv

############################################################
##################### Loading API Keys #####################
############################################################

load_dotenv()

server_api_key = os.getenv("SERPER_API_KEY")

# Initialize the Serper API wrapper as the search tool
serper_search = GoogleSerperAPIWrapper(type="news")

#############################################################
######################### Functions #########################
#############################################################

# Function to get the most recent link of the CSV data
def get_csv_file_details(root_url: str) -> Union[str, List[Dict[str, str]]]:
    """
    Realiza o web scraping de uma URL raiz para encontrar o link de um arquivo CSV.

    Esta função acessa a URL fornecida, analisa seu conteúdo HTML e procura
    por tags de âncora <a> cujo texto começa com "https". Ela assume que
    o primeiro link encontrado que satisfaz essa condição é o link
    direto para o arquivo CSV desejado.

    Args:
        root_url (str): A URL raiz (página web) onde se espera encontrar o link do arquivo CSV.

    Returns:
        Union[str, List[Dict[str, str]]]:
            - Se bem-sucedido, retorna str contendo a URL completa do arquivo CSV.
            - Se falhar (erro HTTP, erro de rede, ou nenhum link encontrado),
              retorna uma List[Dict[str, str]] com o status e detalhes do erro.

    Dependencies:
        - requests: Para realizar requisições HTTP.
        - bs4 (BeautifulSoup): Para análise de conteúdo HTML.
    """
    
    try:
        # HTTP requisition
        print(f"Procurando dados mais recentes no endereço {root_url}")
        
        # Set a custom User-Agent for request identification
        headers = {'User-Agent': 'AgenticCSVScraper/1.0'}
        
        # Send a GET to the main URL with headers and a 10-second wait limit.
        response = requests.get(root_url, headers=headers, timeout=10)
        
        # It raises an HTTPError if the response status is 4xx (client error) or 5xx (server error).
        response.raise_for_status() 
              
        # Instantiates BeautifulSoup to parse the HTML content of the response.
        soup = BeautifulSoup(response.content, 'html.parser')
        
        csv_links = []
        
        # Finds all anchor tags <a> that have an href attribute.
        links = soup.find_all('a', href=True)
        
        # Iterates over the found links to try and identify the CSV URL.
        for link in links:
            # The `.strip()` removes whitespace before/after the text.
            if link.text.strip()[:5] == "https":
                csv_links.append(link.text.strip())
        
        # If no URL starting with "https" was found in the anchor text.
        if not csv_links:
            return [{"status": "Nenhum arquivo CSV encontrado.", 
                     "details": "O scraper não encontrou nenhum link de arquivo .csv na página raiz que comece com 'https'."}]
            
        # Returns the first link found in the list.
        print(f"Dados mais recentes encontrados no endereço {csv_links[0]}")
        return csv_links[0]
    
    # Captures errors related to the request (network, timeout, HTTP).
    except requests.exceptions.RequestException as e:
        return [{"status": "Web Scraping Falhou", "details": f"Network/HTTP Error: {e}"}]
        
    # Captures any other unexpected errors
    except Exception as e:
        return [{"status": "Parsing Error", "details": f"Um erro inesperado ocorreu: {e}"}]
    
# Function to get the date of data publication and the day one year before
def getting_dates(url_str: str) -> tuple[str, str]:
    """
    Extracts two date strings (start and end dates) from a given URL.

    The function identifies the publication date encoded in the URL (in the format '%d-%m-%Y'),
    then computes a start date exactly one year (365 days) before the publication date.

    Args:
        url_str (str): URL containing a date string in the format '%d-%m-%Y' before the '.csv' extension.

    Returns:
        tuple[str, str]: A tuple containing:
            - start_date (str): Date 365 days before the publication date, formatted as '%d-%m-%Y'.
            - end_date (str): Publication date extracted from the URL, formatted as '%d-%m-%Y'.
    """

    # Extracts the substring before ".csv" and gets the last 10 characters,
    # which correspond to the publication date in the format '%d-%m-%Y'.
    end_date = url_str.split(".csv")[0][-10:]

    # Converts the end date string to a datetime object
    # and subtracts 365 days to get the start date.
    start_date = datetime.strptime(end_date, "%d-%m-%Y") - timedelta(days=365)

    # Formats the start date back into a string.
    start_date = start_date.strftime("%d-%m-%Y")

    # Returns both dates as strings.
    return start_date, end_date

# Function to download the data if it does not already exist locally
def download_file_if_missing(download_url: str, local_path: str = "data/") -> str:
    """
    Downloads a CSV file from a given URL only if it does not already exist locally.

    The function:
      1. Extracts the publication date from the URL using getting_dates.
      2. Builds the full local file path including the date.
      3. Checks if the file already exists locally.
      4. If not, downloads it from the given URL and saves it to disk.
      5. Handles any network or unexpected errors gracefully.

    Args:
        download_url (str): URL of the CSV file to be downloaded.
        local_path (str): Directory path (ending with '/') where the file will be saved.

    Returns: None.
    """

    # Extracts the publication date from the URL to include in the filename
    _, end_date = getting_dates(url_str=download_url)
    local_path = os.path.join(local_path, f"SRAG-{end_date}.csv")

    # Check if the file already exists locally
    if os.path.exists(local_path):
        print(f"✅ O arquivo já existe localmente em: {local_path}. Download ignorado.")
        return None

    print(f"📥 Arquivo não encontrado localmente. Tentando fazer o download de: {download_url}")

    try:
        # Request the file from the URL
        response = requests.get(download_url)
        response.raise_for_status()  # Raises exception for HTTP errors (e.g., 404, 500)

        # Save file to disk in binary mode
        with open(local_path, 'wb') as f:
            f.write(response.content)

        print(f"🎉 Arquivo baixado e salvo com sucesso em: {local_path}")
        return None

    except requests.exceptions.RequestException as e:
        # Handles all HTTP/network-related errors
        print(f"❌ Falha no download devido a um erro de rede ou HTTP: {e}")
        return None

    except Exception as e:
        # Catches any other unexpected errors
        print(f"⚠️ Ocorreu um erro inesperado durante o salvamento do arquivo: {e}")
        return None

# Function to create graphics
def generate_case_time_series_charts(
        local_path: str = "data/",
        output_dir: str = "output/graphics",
        date_col: str = 'DT_NOTIFIC'
        ) -> str:
    """
    Generates two time-series charts from a CSV file containing case records:

    1) Cases in the last 30 days.
    2) Cases in the last 12 months.

    The CSV file must include at least the following columns:
      - NU_NOTIFIC: case identifier.
      - DT_NOTIFIC: date of notification, formatted as %d-%m-%Y.

    The function reads the most recent file found in the given folder,
    aggregates the data by date, and generates the charts under output/graphics/.

    Args:
        local_path (str): Path to the folder containing the CSV file. Default is 'data/'.
        date_col (str): Name of the date column. Default is 'DT_NOTIFIC'.

    Returns:
        str: A status message listing the paths of the generated image files,
             or an error message if the operation fails.
    """
    print("Gerando gráficos.")

    # Create output directory for charts if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Select the first file found in the given folder
        file_path = os.path.join(local_path, os.listdir(local_path)[0])

        # Load CSV and read only the required columns
        df = pd.read_csv(file_path, delimiter=";", usecols=["NU_NOTIFIC", "DT_NOTIFIC"])
        
        # Convert the date column to datetime type
        df["DT_NOTIFIC"] = pd.to_datetime(df["DT_NOTIFIC"], format="%Y-%m-%d")

        # Group by date and count the number of daily notifications
        df = (
            df
            .dropna()
            .set_index(date_col)
            .groupby(by=date_col)
            .agg("count")
            .rename(columns={df.columns[0]: 'TotalCases'})
            .sort_index()
        )

        # Determine time cutoffs for 30 days and 12 months based on the latest date
        latest_date = df.index.max()
        cutoff_30d = latest_date - pd.Timedelta(days=30)
        cutoff_12m = latest_date - pd.DateOffset(years=1)

        # 30 days graphic
        df_30d = df.loc[df.index >= cutoff_30d].reset_index()
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

        # Format x-axis to display daily ticks
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.xticks(rotation=45, ha='center')

        # Add chart labels and style
        plt.title(f'Casos nos últimos 30 dias (terminando em {latest_date.strftime("%Y-%m-%d")})')
        plt.xlabel('')
        plt.ylabel('Número de Casos')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(path_30d)
        plt.close()

        # 12 months graphic
        df_12m = df.loc[df.index >= cutoff_12m]
        path_12m = os.path.join(output_dir, "cases_last_12_months.png")

        # Resample data to monthly sums for cleaner visualization
        df_monthly = df_12m["TotalCases"].resample('M').sum().reset_index()
        df_monthly['DT_NOTIFIC'] = df_monthly['DT_NOTIFIC'].dt.strftime('%m-%Y')
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            data=df_monthly,
            x="DT_NOTIFIC",
            y="TotalCases",
            color='steelblue'
        )

        # Simplify chart appearance
        ax.tick_params(left=False, bottom=False)
        ax.set(xlabel=None, ylabel=None)
        ax.set(yticklabels=[])

        # Add value labels above bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%d')

        plt.title(f'Total de Casos (terminando em {latest_date.strftime("%d-%m-%Y")})')
        plt.tight_layout()
        sns.despine(left=True, right=True, bottom=True, top=True)
        plt.savefig(path_12m)
        plt.close()

        return None

    except FileNotFoundError:
        print(f"❌ Erro: arquivo CSV não encontrado em {local_path}")
        return None
    except KeyError as e:
        print(f"❌ Erro: Colunas {e} não encontradas no arquivo CSV.")
        return None
    except Exception as e:
        print(f"❌ Um erro inesperado ocorreu durante a geração do gráfico: {e}")
        return None

# Function to search for online news about SRAG
def search_online_news(
    start_date: str,
    end_date: str,
    query: str = "Síndrome Respiratória Aguda Grave",
    num_results: int = 5,
    news_output_dir: str = "output/news"
) -> str:
    """
    Performs a real-time news search using the Serper API for relevant articles,
    saves the results to a JSON file, and returns the file path.
    
    Args:
        query (str): The search query to find news about.
        num_results (int): The maximum number of results to fetch.
        start_date (str): Date string for search filtering.
        end_date (str): Date string for search filtering.
        news_output_dir = Directory to save the news.
                          
    Returns:
        str: The full file path of the saved JSON file, or an error message.
    """
    
    news_file_name = f"news_context_{end_date}.json"

    # Checking local existence
    if os.path.exists(news_output_dir + "/" + news_file_name):
        print(f"✅ Existem notícias localmente em: {news_output_dir}. Download ignorado.")
        return None

    os.makedirs(news_output_dir, exist_ok=True)
    
    date_info = f" after:{start_date}" if start_date else ""
    date_info += f" before:{end_date}" if end_date else ""

    print(f"Buscando notícias por: {query}{date_info} (Número máximo de resultados: {num_results})")
    
    try:
        # Execute the search
        raw_results: Dict = serper_search.results(query, num_results=num_results)

        # Extract and format key information
        news_items = []
        if 'news' in raw_results and raw_results['news']:
            for item in raw_results['news']:
                news_items.append({
                    "title": item.get('title', 'N/A'),
                    "snippet": item.get('snippet', 'Nenhum resumo disponível.'),
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
                
            print(f"🎉 Contexto de notícias relevante salvo com sucesso em: {full_file_path}")
            return news_items
        else:
            print("Nenhum artigo de notícia relevante foi encontrado para a consulta e nenhum arquivo foi salvo.")
            return None

    except Exception as e:
        print(f"❌ Erro ao executar a pesquisa de notícias Serper ou ao salvar o arquivo: {e}")
        return None

# Function to calculate epidemiology rates
def calculate_epidemiology_rates(
    local_path: str,
    selected_columns: List[str],
    end_date: str
) -> Dict[str, Union[float, str, Any]]:
    """
    Calculates various epidemiological rates (case increase rate, case fatality rate (CFR),
    ICU occupancy rate, and vaccination rate) from a local notification CSV file.

    The rates are calculated based on notification data within a 30-day period
    ending at `end_date`, except for the case increase rate, which uses monthly variation.
    The final results are saved to a JSON file.

    Args:
        local_path (str): The path to the directory containing the CSV file.
        selected_columns (List[str]): A list of column names that should be read from the CSV file.
        end_date (str): The end date of the analysis period, in "%d-%m-%Y" format.

    Returns:
        results (Dict[str, Union[float, str, Any]]): A dictionary containing the calculated 
            rates or "N/A" messages if division by zero occurs. Returns `None` if
            reading the dataset fails.
    """

    print("Calculando métricas.")
    
    try:
        # Attempt to locate the first file in the directory and construct the full path.
        file_path = local_path + os.listdir(local_path)[0]
        
        # Read the CSV file using semicolon as a delimiter and select only the required columns.
        df = pd.read_csv(file_path, delimiter=";", usecols=selected_columns)
        
        # Convert date columns (those starting with "DT") to datetime objects.
        for col in selected_columns:
            if col[:2] == "DT":
                # Assumes dates are already in YYYY-MM-DD format within the CSV
                df[col] = pd.to_datetime(df[col], format="%Y-%m-%d")

    except Exception as e:
        # In case of reading or conversion failure, print the error and return None.
        print(f"Falha ao ler o conjunto de dados! Erro: {e}")
        return None

    results = {}
    # Calculate the start date (30 days before the end date) for recent analysis.
    start_date = datetime.strptime(end_date, "%d-%m-%Y") - timedelta(days=30)

    #################################################
    ### Rate of Case Increase (Percentage Change) ###
    #################################################

    # Select relevant columns for case count
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

    # Resample data to monthly sums for a cleaner 12-month view.
    df_monthly = df_cases["TotalCases"].resample('M').sum()
    df_monthly = df_monthly.reset_index()
    # Format the date column to Month-Year
    df_monthly['DT_NOTIFIC'] = df_monthly['DT_NOTIFIC'].dt.strftime('%m-%Y')

    # Get the total cases from the second-to-last month and the last month for change calculation.
    previous_cases = df_monthly["TotalCases"].iloc[-2]
    current_cases = df_monthly["TotalCases"].iloc[-1]

    # Calculate the percentage increase rate
    if previous_cases > 0:
        rate_increase = ((current_cases - previous_cases) / previous_cases) * 100
        results['case_increase_rate (%)'] = round(rate_increase, 2)
    else:
        results['case_increase_rate (%)'] = "N/A (O número de casos anteriores era zero.)"

    ###############################################
    #### Death Rate (Case Fatality Rate - CFR) ####
    ###############################################

    df_deaths = df[["DT_NOTIFIC","EVOLUCAO"]]
    # Filter for cases notified in the last month (30 days) and drop nulls.
    df_deaths = df_deaths.query(f"DT_NOTIFIC > '{start_date}'").dropna()
    # Total cases in the period (treated as confirmed/closed)
    total_cases = df_deaths.shape[0]
    # Count of deaths (EVOLUCAO = 2.0)
    current_deaths = df_deaths.query("EVOLUCAO == 2.0").shape[0]

    # Calculate the Case Fatality Rate (CFR)
    if total_cases > 0:
        death_rate = (current_deaths / total_cases) * 100
        results['death_rate_cfr (%)'] = round(death_rate, 2)
    else:
        results['death_rate_cfr (%)'] = "N/A (Total de casos confirmados foi zero.)"

    ###############################################
    ############## ICU Occupancy Rate #############
    ###############################################

    df_icu = df[["UTI","DT_NOTIFIC"]]
    # Filter for cases notified in the last month (30 days)
    df_icu = df_icu.query(f"DT_NOTIFIC > '{start_date}'")
    # Total cases in the period
    total_cases_analyzed = df_icu.shape[0]
    # Count of cases admitted to ICU (UTI = 1.0)
    icu_occupancy = df_icu.query("UTI == 1.0").shape[0]

    # Calculate the ICU Admission Rate (ICU Cases / Total Cases Analyzed)
    if total_cases_analyzed > 0:
        icu_rate = (icu_occupancy / total_cases_analyzed) * 100
        results['icu_occupancy_rate (%)'] = round(icu_rate, 2)
    else:
        results['icu_occupancy_rate (%)'] = "N/A (Total de casos para a análise de UTI foi zero.)"

    # ###############################################
    # Population Vaccination Rate (among analyzed cases)
    # ###############################################

    df_vacina = df[["VACINA","DT_NOTIFIC"]]
    # Filter for cases notified in the last month (30 days)
    df_vacina = df_vacina.query(f"DT_NOTIFIC > '{start_date}'")

    # Total cases analyzed in the period (sample population)
    total_population_analyzed = df_vacina.shape[0]
    # Count of cases that were vaccinated (VACINA = 1.0)
    people_vaccinated = df_vacina.query("VACINA == 1.0").shape[0]

    # Calculate the Vaccination Rate (Vaccinated Cases / Total Cases Analyzed)
    if total_population_analyzed > 0:
        vac_rate = (people_vaccinated / total_population_analyzed) * 100
        results['population_vaccination_rate (%)'] = round(vac_rate, 2)
    else:
        results['population_vaccination_rate (%)'] = "N/A (População total para análise de vacinação é zero.)"
    
    # Define the path for the output JSON file.
    file_path_json = f"output/rates/calc_rates_{end_date}.json"
    directory = os.path.dirname(file_path_json)
    # Create the directory if it does not exist.
    os.makedirs(directory, exist_ok=True)

    # Save the results dictionary to a JSON file.
    with open(file_path_json, 'w') as f:
        json.dump(results, f, indent=4)

    return results

# Function to the LLM generate a description of the graphics
def analyze_graphic(graphic_12_months_path: str, graphic_30_days_path: str) -> Tuple[str, str]:
    """
    Analyzes two graphics (12 months and 30 days) using the Gemini 2.5 Flash
    model to generate epidemiological descriptions.

    The LLM model acts as an expert epidemiological analyst, describing
    case trends and suggesting how the data can be used by public health
    authorities for health planning.

    Args:
        graphic_12_months_path (str): The path to the 12-month graphic file.
        graphic_30_days_path (str): The path to the 30-day graphic file.

    Returns:
        desc_12_months (str): A string containing the description of the 12-month graphic.
        desc_30_days (str): A string containing the description of the 30-day graphic.
    """
    print("Gerando descrição dos gráficos.")

    # List to store the generated descriptions.
    descriptions = []
    
    try:
        # Initializes the LLM.
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Iterates over the paths of the graphics.
        for path in [graphic_12_months_path, graphic_30_days_path]:
            # Assembling the prompt for the LLM.
            prompt = f"""
                Você é um analista epidemiológico especialista.
                Descreva o gráfico {path} em um parágrafo, enfatizando momentos de mais 
                casos e de menos casos e estabelecendo isso com as estações do ano, se o gráfico 
                contiver informações mensais. Se o gráfico tiver informações diárias, elabore uma 
                descrição da situação recente, explore possíveis sazonalidades de curto prazo e 
                tente explicar esse comportamento.
                Escreva também sobre como este gráfico poderia ser utilizado pelo poder público 
                para planejar ações de saúde como, por exemplo, vacinação.

                Evite começar a resposta com 'Como analista epidemiológico'. Vá Direto para 
                a análise e seja sucinto.
            """

            # Invokes the LLM model with the prompt.
            result = llm.invoke(prompt)
            # Stores the content of the LLM's response.
            descriptions.append(result.content)

        # Unpacks the list of descriptions into 12-month and 30-day variables.
        desc_12_months, desc_30_days = descriptions

    except Exception as e:
        # Captures and prints any exception that occurred.
        print(e)
        return None
    
    except ValueError:
        # Handles the case where unpacking fails.
        print("Erro: descrição de 12 meses indisponível", "Erro: descrição de 30 dias indisponível")
        return None

    # Returns the two descriptions generated by the LLM.
    return desc_12_months, desc_30_days

# Function to create a description based on calculates metrics
def analyze_metrics(metrics: Dict[str, Any]) -> str:
    """
    Analyzes a dictionary of epidemiological metrics related
    to Síndrome Respiratória Agura Grave (SRAG) using the Gemini 2.5 Flash
    model to generate a brief summary analysis.

    Args:
        metrics (Dict): A dictionary containing the metrics and data to be analyzed.
                    Examples may include case counts, incidence rates,
                    or bed occupancy percentages.

    Return:
        result.content (List[str]): A string containing the epidemiological analysis of SRAG generated by the LLM.
    """
    print("Gerando descrição das métricas.")

    try:
        # Initializes the Google Large Language Model (LLM).
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # Assembling the prompt for the LLM.
        prompt = f"""
            Você é um analista epidemiológico especialista.
            Escreva um parágrafo sobre Síndrome Respiratória Aguda Grave (SRAG) 
            usando os valores {metrics}.
            Vá Direto para a análise e seja sucinto.
        """

        # Invokes the LLM model with the prompt.
        result = llm.invoke(prompt)

    except Exception as e:
        # Captures and prints any exception.
        print(e)
        # In case of an error, return a failure message to guarantee the return type.
        print(f"Erro ao executar a análise do LLM: {e}")
        return None

    # Returns the content of the analysis generated by the LLM.
    return result.content

# Function to create the report based on news.
def create_content(news: str, query: str) -> Optional[List[str]]:
    """
    Synthesizes a set of news articles into a structured report using the
    Gemini 2.5 Flash model, acting as an expert geopolitical and
    epidemiological analyst.

    The generated report is in Markdown format and is structured into sections
    (Executive Summary, Recent Developments, and Perspectives). It is returned
    as a list of strings, separated by the "###" marker.

    Args:
        news (str): A string containing a JSON array of the most recent news articles.
        query (str): The main topic string used to search for the news.

    Returns:
        contents (List[str]): A list of strings, where each item represents a section of the
             report (separated by the '###' marker). Returns None on error.
    """
    print("Gerando o conteúdo do relatório.")

    try:
        # Initializes the Google Large Language Model (LLM).
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

        # The prompt defines the LLM's role and the strict formatting rules
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

            Guardrails:
                Produza conteúdo exclusivamente sobre Síndrome Respiratória Aguda Grave.
                Se a {query} contiver outro tipo de assunto, responda apenas que seu objetivo é 
                auxiliar somente na geração de conteúdo sobre Síndrome Respiratória Agura grave.
        """

        # Invokes the LLM model with the prompt to generate the report.
        result = llm.invoke(new_prompt)
        
        # Splits the report content into a list of strings.
        contents = result.content.split("###")

        # Returns the list of report sections.
        return contents

    except Exception as e:
        # Captures and prints any exception.
        print(f"Erro: {e}")
        
        return None
    
# Function to create the report (PDF)
def generate_pdf_report(
        start_date: str,
        end_date: str,
        summary: str,
        #recent_developments: str,
        perspectives: str,
        desc_12_months:str,
        desc_30_days:str,
        desc_metrics: str
        ) -> str:
    """
    Generates a structured PDF report for SRAG (Síndrome Respiratória Aguda Grave)
    using pre-analyzed text descriptions and existing graphic files.

    The function creates a titled, dated document containing an executive summary,
    two graphics (12 months and 30 days) with their corresponding descriptions,
    a metrics analysis, and a perspectives section. It uses the ReportLab library
    for PDF creation.

    Args:
        start_date (str): The start date of the reporting period.
        end_date (str): The end date of the reporting period.
        summary (str): The executive summary paragraph, generated by an LLM based on news/data.
        perspectives (str): The perspectives/risk analysis paragraph, generated by an LLM.
        desc_12_months (str): The LLM-generated description/analysis for the 12-month graphic.
        desc_30_days (str): The LLM-generated description/analysis for the 30-day graphic.
        desc_metrics (str): The LLM-generated analysis of key epidemiological metrics.
                          
    Returns:
        str: A status message indicating success or failure, including the full
             path of the generated PDF on success.
    """

    print("Gerando relatório.")

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
            img = Image(file_path_12months, width=6 * 72, height=3 * 72) 
            story.append(img)
            story.append(Spacer(1, 12))
            story.append(Paragraph(f"{desc_12_months}"))
            story.append(Spacer(1, 12))
        else:
            story.append(Paragraph(f"Warning: Graphic file not found: {desc_12_months}", styles['Normal']))

        # Adding 30 days graphic to the report
        if os.path.exists(file_path_30days):
            # Add Image (scale to fit page width, e.g., 5 inches wide)
            img = Image(file_path_30days, width=6 * 72, height=3 * 72) 
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
