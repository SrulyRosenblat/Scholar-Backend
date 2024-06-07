from openai import OpenAI
import json
import requests
import os
from typing import List, Optional
from dotenv import load_dotenv
import bleach
import markdown
from fastapi import FastAPI, File, UploadFile,Form
from groq import Groq
import urllib.request
import xml.etree.ElementTree as ET

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from collections import defaultdict 
from serpapi import GoogleSearch
import re



load_dotenv()
client = OpenAI()
app = FastAPI()
groqClient = Groq(
    api_key=os.getenv("groq_api")
)
api_key_ns =os.getenv('api_key_ns')
api_key_gs =os.getenv('api_key_gs')


# cohere_api_key = os.getenv("CO_API_URL")

# fireworks.client.api_key = os.getenv("FIREWORKS_API_KEY")


# @app.post("/upload")
# async def upload_files(files: List[UploadFile] = File(...)):
#     uploaded_files = []
#     for file in files:
#         contents = await file.read()
#         # Process the file contents
#         file_contents = contents.decode("utf-8")
#         # Perform any necessary operations with the file contents
#         # ...
#         uploaded_files.append({"filename": file.filename, "content": file_contents})
    

#     return {"message": "Files uploaded successfully", "files": uploaded_files}

def htmlify(text):
    """
    turn markdown into safe html
    """
    safe_tags = [
        "p",
        "strong", "b",
        "em", "i",
        "u",
        "span",
        "br",
        "hr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "ul",
        "ol",
        "li",
        "a",
        "img",
        "blockquote",
        "code",
        "pre",
        "div"
        ]

    safe_attrs = {
        'a':'href'
        
    }

    raw_html =  markdown.markdown(text)

    # clean_html = bleach.clean(raw_html, tags=safe_tags, attributes=safe_attrs, strip=True)

    return raw_html

# def format_output(text):
#     safe_tags = ["p", "strong", "em", "h1", "h2", "h3", "ul", "ol", "li", "a", "span"]
#     html_output = markdown(text)
#     safe_html_output = bleach.clean(html_output, tags=safe_tags)
#     return f'<li class="bot"><div>{safe_html_output}</div></li>'


def ask_model(messages=[],model='gpt-3.5-turbo-0125'):
    if model.startswith('gpt'):
        return client.chat.completions.create(
            model=model,
            messages=messages
        )
    elif model == 'fine-tune':
        return client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:personal::9OsmbglY",
            messages=messages,
            temperature=0.5,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )

    else:
        return groqClient.chat.completions.create(
            model=model,
            messages=messages
        )
# def prepare_snippets(results: list) -> list:
#     """
#     prepare the snippets to be used in the reranking
#     :results: the results to be prepared
#     :return: the prepared snippets
#     """
#     # borrowed from another project i did will make more relevent
#     snippets = []
#     for result in results:
#         host = result['meta_url']['hostname']
#         host = host.replace('www.','').lower().strip()
#         if 'extra_snippets' in result:

#             snippetCount = len(result['extra_snippets'])
#             snippets.extend({'text':snippet,'url':result['url'],'title':result['title'],'snippetCount':snippetCount} for snippet in result['extra_snippets']) 
#         else:
#             snippets.append({'text':result['description'],'url':result['url'],'title':result['title'],'snippetCount':1})
#     return snippets

def prepare_search_documents(results):
    documents = []
    for result in results:
        if 'extra_snippets' in result:
            documents.extend(Document(page_content=snippet, metadata={"url":result['url'],"title":result['title']}) for snippet in result['extra_snippets'] )#{'text':snippet,'url':result['url'],'title':result['title'],'snippetCount':snippetCount} for snippet in result['extra_snippets']) 
        else:
            documents.append(Document(page_content=result['description'], metadata={"url":result['url'],"title":result['title']}))#{'text':result['description'],'url':result['url'],'title':result['title'],'snippetCount':1})
    return documents

# def rerank(docs, query ,top_n=None):
#     results =[]
#     snippets = prepare_snippets(docs)

#     url = "https://api.cohere.ai/v1/rerank"
#     headers = {
#         "accept": "application/json",
#         "content-type": "application/json",
#         "Authorization": f"Bearer {cohere_api_key}"
#     }
#     data = {
#         "model": "rerank-english-v2.0",
#         "query": query,
#         "top_n": top_n,
#         "documents": snippets,
#     }

#     response = requests.post(url, headers=headers, json=data, timeout=200)

#     if response.status_code == 200:
#         response = response.json()
#     else:
#         print(f"Error: {response.status_code}")
#         return []
#     for r in response['results']:
#         results.append(snippets[r['index']])

#     return results


# def stringify_snippets(snippets):
#     s = "Search Results:\n"
#     for result in snippets:
#         s += f'---\n{result["url"]}\n**{result["title"]}**\n{result["text"]}\n---'
#     return s

def stringify_docs(docs):
    result = ''
    pdf_str = ""
    search_str = ""
    for p in docs:
        if p.metadata.get("page",None):
            pdf_str += f"---\nfile name: {p.metadata['source']}\ncontent: {p.page_content}\n---"
        else:
            search_str += f"---\nUrl: {p.metadata['url']}\n**{p.metadata['title']}**\n{p.page_content}\n---"
    if pdf_str: 
        result +="Users documents:\n"+ pdf_str + '\n'
    if search_str:
        result += "Search Results:\n"+ search_str

    return result.strip()

# def ask_chat(message):
#     completion = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": message}
#     ]
#     )   
#     return completion.choices[0].message.content


@app.get("/")
def read_root():
    return "welcome to the API"

# @app.get("/ask/{question}")
# def ask(question: str):
#     return {"response":htmlify(ask_chat(question)),"question":question,"sources":[]}


def buildMessages(conversation):
    conversation_string = '```'
    for message in conversation:
     conversation_string +=  f'\n {"[USER]" if message["role"] == "user" else "[AI]" } {message["content"]}'
    conversation_string += '\n```'
    print(conversation_string)
    messages=[
        {
        "role": "system",
        "content": "Given a conversation, your task is to identify and extract a single, concise phrase that best represents the main topic or question discussed. This phrase should be suitable for use as a search query on Google to find more information related to the conversation. Please provide only the extracted phrase without any additional text or explanations. You may use 'NO SEARCH' only cases where the question or statement requires no knowledge to answer\n\n"
        },
        {
        "role": "user",
        "content": "```\n[USER] What are some good exercises for losing weight?\n[AI] Some effective exercises for losing weight include:\n1. Cardiovascular exercises like running, cycling, swimming, or brisk walking\n2. High-Intensity Interval Training (HIIT)\n3. Strength training exercises like weightlifting or bodyweight exercises\n4. Yoga or Pilates for flexibility and core strength\nRemember, a combination of regular exercise and a balanced diet is the most effective approach for sustainable weight loss.\n\n[USER] How often should I exercise to see results?\n[AI] The frequency of exercise needed to see results varies from person to person, but a general guideline is to aim for at least 150 minutes of moderate-intensity aerobic activity or 75 minutes of vigorous-intensity aerobic activity per week. This can be spread out over several days, such as 30 minutes of exercise, 5 times a week.\nFor strength training, aim for at least 2-3 sessions per week, targeting all major muscle groups. Allow for adequate rest and recovery between workouts.\nConsistency is key, so find a schedule that works for you and stick to it. Remember to start gradually and increase the intensity and duration of your workouts over time to avoid injury and maintain progress.\n\n[USER] Are there any specific diet tips that can help?\n```"
        },
        {
        "role": "assistant",
        "content": "diet tips for weight loss"
        },
        {
        "role": "user",
        "content": "```\n[USER] What are some key features of Python?\n[AI] Python is known for several key features that make it a popular programming language:\n1. Easy to learn and read: Python has a simple and intuitive syntax, making it beginner-friendly and easy to understand.\n2. Versatile: Python can be used for a wide range of applications, including web development, data analysis, machine learning, and more.\n3. Large standard library: Python comes with a comprehensive standard library that provides tools for various tasks, reducing the need for external dependencies.\n4. Third-party packages: Python has a vast ecosystem of third-party packages available through the Python Package Index (PyPI), extending its functionality even further.\n5. Interpreted language: Python is an interpreted language, meaning you can run code without the need for compilation, enabling faster development cycles.\n6. Cross-platform compatibility: Python code can run on various operating systems, including Windows, macOS, and Linux.\n7. Strong community support: Python has a large and active community that contributes to its development, provides support, and creates numerous resources and tutorials.\n\n[USER] How can I install Python on my computer?\n[AI] To install Python on your computer, follow these steps:\n1. Visit the official Python website at https://www.python.org/downloads/.\n2. Click on the download link for the latest stable version of Python suitable for your operating system (e.g., Windows, macOS, or Linux).\n3. Run the installer and follow the installation wizard:\n   - For Windows: Check the option to add Python to the PATH variable during installation, which allows you to run Python from the command prompt.\n   - For macOS: The installer will guide you through the installation process.\n   - For Linux: Python is usually pre-installed, but you can use your distribution's package manager to install the latest version.\n4. Once the installation is complete, you can verify the installation by opening a terminal or command prompt and typing `python --version`. It should display the installed Python version.\n5. You can now start using Python by running the Python interpreter from the terminal or by using an Integrated Development Environment (IDE) like PyCharm, Visual Studio Code, or Jupyter Notebook.\nNote: If you have multiple versions of Python installed, you may need to use `python3` instead of `python` in the command line.\n\n[USER] What are some good resources?```"
        },
        {
        "role": "assistant",
        "content": "good resources to set up and learn Python"
        },
        {
        "role": "user",
        "content": "```\n[USER] whats up?\n```"
        },
        {
        "role": "assistant",
        "content": "NO SEARCH"
        },
        {
        "role": "user",
        "content": "```\n[USER] who was the 3rd 4th and 5th president\n```"
        },
        {
        "role": "assistant",
        "content": "first 5 presidents"
        },
        {
        "role":'user',
        "content": conversation_string
        }
    ]
    return messages

def braveSearch(query, prefix='', count=20):
  # search api
    url = f"https://api.search.brave.com/res/v1/web/search?q={prefix}{query}"

    payload = {
        'count': count,
        'text_decorations':False,
        'rich':True,
        'result_filter':'web',
    }

    headers = {
    'X-Subscription-Token': os.getenv("BRAVE_SUBSCRIPTION_API"),
    'Cookie':os.getenv("BRAVE_COOKIE")
    }

    response = requests.request("GET", url, headers=headers, params=payload,timeout=200)

    try:
        return json.loads(response.text)['web']['results']
    except Exception as e:
        print(e)
        return []
# @app.get("/search_from_conversation/")
# def search_from_llm(conversation : List[dict]):
#     messages = buildMessages(conversation)
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo-0125",
#         messages=messages
#     )
#     q = completion.choices[0].message.content
#     print(q)

#     return q
@app.get("/search_from_conversation/")
def search_from_llm(conversation : List[dict]):
    messages = buildMessages(conversation)
    completion = ask_model(
        model="llama3-8b-8192",
        messages=messages
    )
    q = completion.choices[0].message.content
    print(q)

    return q


# @app.post("/chat/")
# def ask_llm(conversation  : List[dict]):
#     search_results , query = search_from_llm(conversation)

#     if search_results:
#         conversation[-1]['content'] =  stringify_docs(search_results) +"\n\nUser Query:\n" + conversation[-1]['content'] 

#     conversation.insert(0,    {"role": "system", "content": "you are a smart assistant that uses markdown and cites sources"},)
#     print(conversation)
#     completion = client.chat.completions.create(
#         model="gpt-3.5-turbo-0125",
#         messages=conversation
#     )
#     result = htmlify(completion.choices[0].message.content)
#     return {'result': result, 'sources':search_results,"raw-text":completion.choices[0].message.content,'search query':query}



def search_documents(pages,query, k=5):
    '''
    search through pages
    '''
    if pages:
        faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
        docs = faiss_index.similarity_search(query, k=k)
        return docs
    else:
        return []


def split_document_types(documents):
    """
    make dictionary's to separate out the pdfs and websites.
    """
    documents = [document.dict() for document in documents]
    snippets = [doc for doc in documents if doc['metadata'].get('url',False)]
    pdf_snippets = [doc for doc in documents if not doc['metadata'].get('url',False)]
    websites = defaultdict(list)
    pdfs = defaultdict(list)
    for snippet in snippets:
        url = snippet['metadata']['url']
        websites[url].append(snippet)
    for paragraph in pdf_snippets:
        file = paragraph['metadata']['source']
        pdfs[file].append(paragraph)
    for key in pdfs.keys():
        # sort based on page number
        pdfs[key].sort(key = lambda x: x['metadata']['page'] )
    return websites, pdfs

def build_source_sheet(websites,pdfs):
    result = ""
    for url in websites.keys():
        result += f'''### [{websites[url][0]["metadata"]['title']}]({url})\n'''
        for snippet in websites[url]:
            result+=f'''- {snippet['page_content']}\n'''
    for pdf_name in pdfs.keys():
        result += f'''### PDF: {pdf_name}\n'''
        count =len(pdfs[pdf_name])
        i = 0
        for snippet in pdfs[pdf_name]:
            result+=f'''{snippet['page_content']}\n{'...' if i != count -1 else ''}\n'''
            i +=1

    return result

from urllib.parse import urlparse

# def get_domain_name(url):
#     parsed_url = urlparse(url)
#     domain_name = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_url)
#     return domain_name
import re

def extract_main_url_part(url):
    pattern = re.compile(r'^(?:https?:\/\/)?(?:www\.)?([^\/?#]+)')
    match = pattern.match(url)
    if match:
        return match.group(1)
    else:
        return None
def humanSourceSheet(websites):
    if not websites:
        return ''
 
    start = '<br/>\n<div class="flex flex-wrap items-center gap-2 no-underline">\n<p class="py-2 text-sm font-semibold text-gray-700">Learn more:</p>\n'
    middle = ''
    count = 1
    for url in websites.keys():
        middle += f'<a href="{url}"target="_blank" class="flex items-center gap-1 rounded border border-gray-300 px-4 py-2 text-sm font-semibold text-gray-700 no-underline shadow hover:text-blue-600"><span class="text-red-500">{count}</span><span>{extract_main_url_part(url)}</span></a>'
        # middle += f'<p>{count}. <a href="{url}" target="_blank" class="text-blue-600 underline">{websites[url][0]["metadata"]["title"]}</a></p>'
        count += 1
    end = '\n</div>'
    return htmlify(start + middle + end)


    


@app.post("/chat+/")
async def ask_llm_pdf(
    conversation : str = Form(...), internet_search: Optional[bool] = Form(True), files: List[UploadFile] = File(None)):
    """
    api to interact with our ai
    """
    conversation = json.loads(conversation)

    if not internet_search and not files:
        print("skiping all search")
        completion = ask_model(
        model="llama3-8b-8192",
        messages=conversation
        )

        formatted_result = htmlify(completion.choices[0].message.content)
        print("Formatted result:", formatted_result)
        return {'result': formatted_result, "raw-text": completion.choices[0].message.content, 'search-query': None, 'sorted-documents': [], 'Last-message': conversation[-1]['content'], 'source_sheet': ''}
    sorted_documents =[]
    # print("convesation",conversation)
    query = search_from_llm(conversation)
    search_results = []
    if query.strip() != "NO SEARCH" and internet_search:
        print('searching internet')
        search_results = braveSearch(query)

    print("files",files)
    pages = prepare_search_documents(search_results)
    # print('pages', pages )
    print('search results:',len(pages))

    if files:
        os.makedirs("temp", exist_ok=True)
        for file in files:
            temp_file_path = f"temp/{file.filename}"
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(await file.read())

            loader = PyPDFLoader(temp_file_path)
            pages.extend(list(loader.load_and_split()))

            os.remove(temp_file_path)
    print('Total pages after adding PDFs:', len(pages))
    sorted_documents = search_documents(pages, query, k=5)
    print("Sorted documents:", sorted_documents)
    websites, pdfs = split_document_types(sorted_documents)
    print("Websites:", websites)
    print("PDFs:", pdfs)
    search_str = ''
    if websites or pdfs:
        search_str = build_source_sheet(websites, pdfs)
        
    print(search_str)
    
    if search_str:
        conversation[-1]['content'] = f"# Possible Sources:\n"+ search_str + "\n\n# User Query:\n" + conversation[-1]['content']
    source_sheet = htmlify(search_str)
    conversation.insert(0, {"role": "system", "content": "you are a smart assistant that uses markdown and cites sources"})
    completion = ask_model(
        model="fine-tune",
        messages=conversation
    )

    formatted_result = htmlify(completion.choices[0].message.content) +'\n' + humanSourceSheet(websites)
    print("Formatted result:", formatted_result)
    return {'result': formatted_result, "raw-text": completion.choices[0].message.content, 'search-query': query, 'sorted-documents': sorted_documents, 'Last-message': conversation[-1]['content'], 'source_sheet': humanSourceSheet(websites)}

@app.post("/search_arxiv/")
def search_arxiv(query : str, max_results : int):
    base_url = "http://export.arxiv.org/api/query?"
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
    }
    url = base_url + urllib.parse.urlencode(params)

    # Fetch the Atom XML data
    with urllib.request.urlopen(url) as response:
        atom_xml = response.read().decode('utf-8')
    
    # Parse the XML data
    root = ET.fromstring(atom_xml)
    
    # Extract relevant information from the XML tree
    results = []
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', '').replace('  ', ' ')
        published = entry.find('{http://www.w3.org/2005/Atom}published').text.strip()
        authors = [author.text.strip() for author in entry.findall('{http://www.w3.org/2005/Atom}author/{http://www.w3.org/2005/Atom}name')]
        abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', '')
        url = entry.find('{http://www.w3.org/2005/Atom}id').text

        # Append the extracted information to the results list
        results.append({
            'url': url,
            'title': title,
            'authors': authors,
            'date': published,
            'abstract': abstract
        })
    
    return results


def find_years(text : str):
    # Define the pattern to search for a four-digit number (year)
    pattern = r'\b\d{4}\b'  # \b represents a word boundary, \d{4} matches four digits
    
    # Use the findall() function to find all matches in the text
    years = re.findall(pattern, text)
    
    return years

@app.post("/search_all")
def search_all(query : str, max_results : int, api_key_ns :str = '',api_key_gs :str = '', arxiv_active:bool = False):  
    count = 0
    results = []
    if api_key_gs:
        count += 1
    if api_key_ns:
        count += 1
    if arxiv_active:
        count += 1
    if count == 0:
        return []

    count_per_item = max_results // count
    if api_key_gs:
        try:
            results.extend(search_google_scholar(query,count_per_item,api_key=api_key_gs))
        except Exception as e:
            print(e)
    if api_key_ns:
        try:
            results.extend(search_nature_springer(query,count_per_item,api_key_ns))
        except Exception as e:
            print(e)
    if arxiv_active:
        results.extend(search_arxiv(query,count_per_item)) 
    return results

@app.post("/search_google_scholar/")
def search_google_scholar(query : str, max_results : int, api_key :str = api_key_gs):  
    # Paramters for the search
    params = {
        "engine": "google_scholar",
        "q": query,
        "hl": "en",
        "num": max_results,
        "api_key": api_key
    }

    # Seach using GoogleScholar API and SERP and store in a dictionary
    search = GoogleSearch(params)
    results = search.get_dict()

    result = []
    
    for i in range(max_results):
        if results["search_metadata"]["status"] == "Success":
            # Extract relevant information from the dictionary
            title = results["organic_results"][i]["title"]
            url = results["search_metadata"]["google_scholar_url"]
            abstract = results["organic_results"][i]["snippet"]
            authors = results["organic_results"][i]["publication_info"]["summary"].split("-")[0][:-1]
            date = find_years(results["organic_results"][i]["publication_info"]["summary"])[0]
    
            # Append the extracted information to the results list
            result.append({
                'url': url,
                'title': title,
                'authors': authors,
                'date': date,
                'abstract': abstract
            })

    return result



@app.post("/search_nature_springer/")
def search_nature_springer(query : str, max_results : int, api_key :str = api_key_ns):
    api_endpoint = "https://api.springernature.com/openaccess/json"
    # Make a GET request to the API endpoint with your API key and search parameters
    response = requests.get(api_endpoint, params={"api_key": api_key, "q": query,"s": 1, "p": max_results},timeout=12)
    result = []
    
    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Check if there are search results
        if "records" in data and data["records"]: 
            # data["records"] contains max_results number of papers so we 
            # will iterate over all and gather important information
            for paper_info in data["records"]:
                title = paper_info["title"]
                url = paper_info["url"][0]["value"]
                abstract = paper_info["abstract"]["p"]
                authors = []
                for author in paper_info["creators"]:
                    authors.append(author["creator"])
                date = paper_info["publicationDate"]

                # Append the extracted information to the results list
                result.append({
                    'url': url,
                    'title': title,
                    'authors': authors,
                    'date': date,
                    'abstract': abstract
                })
                
        else:
            print("No search results found for the specified query.")
            return result
    else:
        print("Error: Unable to retrieve search results from the API.")
        return result
    return result