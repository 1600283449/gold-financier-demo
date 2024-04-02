from pathlib import Path  
from typing import List, Optional
import os
import pdfkit 
import boto3
from file_utils import filing_exists
from fire import Fire
from sec_edgar_downloader._Downloader import Downloader  
# from sec_edgar_downloader import Downloader
from tqdm.contrib.itertools import product  
import os
from flask import Flask, jsonify, request
from flask_cors import CORS
# 引入 CORS，解决跨域问题
import shutil
import time
import openai 
# import pinecone 
from pinecone import Pinecone, ServerlessSpec
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain_community.chat_models import ChatOpenAI

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain_community.llms import CTransformers, Replicate  
from bs4 import BeautifulSoup  
# from langchain_community.vectorstores import Pinecone
# 搜索测试

import streamlit as st
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
import requests
import json 
import pdfminer
from langchain.schema import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
# 搜索测试完
load_dotenv()
app = Flask(__name__)
from langchain_nvidia_ai_endpoints  import NVIDIAEmbeddings,ChatNVIDIA
from langchain_community.document_loaders import WebBaseLoader
# from sec_cik_mapper import StockMapper
from pathlib import Path
CORS(app) 

REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')  
NVIDIA_API_KEY = os.environ.get('NVIDIA_API_KEY')  
S3_BUCKET = os.environ.get('S3_BUCKET')  
S3_ACCESS_KEY = os.environ.get('S3_ACCESS_KEY')  
S3_SECRET_KEY = os.environ.get('S3_SECRET_KEY')  
  
s3 = boto3.client('s3', aws_access_key_id=S3_ACCESS_KEY,
                  aws_secret_access_key=S3_SECRET_KEY)
DEFAULT_OUTPUT_DIR = "data/"
# You can lookup the CIK for a company here: https://www.sec.gov/edgar/searchedgar/companysearch
DEFAULT_CIKS = []
DEFAULT_FILING_TYPES = []

EMAIL = "qq1369556525@gmail.com"
# 搜索测试
# ------------------ 加载环境变量 ------------------ #
load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")  # 获取环境变量中的BROWSERLESS_API_KEY
serper_api_key = os.getenv("SERP_API_KEY")  # 获取环境变量中的SERP_API_KEY

# ------------------ 网络爬虫部分 ------------------ #

# 1. 搜索工具
def search(query):
    url = "https://google.serper.dev/search"  # 定义搜索的URL

    payload = json.dumps({
        "q": query  # 查询参数
    })
    headers = {
        'X-API-KEY': serper_api_key,  # 使用API密钥作为请求头中的X-API-KEY
        'Content-Type': 'application/json'  # 请求内容类型为JSON
    }
    response = requests.request("POST", url, headers=headers, data=payload)  # 发起POST请求
    print(response.text)  # 打印响应文本
    return response.text  # 返回响应文本



# 2. 网站内容抓取工具
# 具有PDF抓取功能的修改代码
def scrape_website(objective: str, url: str):
    """
    网站抓取，根据目标摘要内容，如果内容过大则进行摘要。
    目标是用户指定的原始目标和任务，url是要抓取的网站的URL。
    """
    print("正在抓取网站...")

    # 检查URL是否指向PDF文件
    if url.lower().endswith('.pdf'):
        # 下载PDF
        response = requests.get(url)
        with open("/tmp/temp.pdf", "wb") as f:
            f.write(response.content)

        # 使用pdfminer提取PDF中的文本
        from pdfminer.high_level import extract_text
        text = extract_text("/tmp/temp.pdf")
    else:
        # 针对非PDF URL的现有逻辑
        headers = {
            'Cache-Control': 'no-cache',
            'Content-Type': 'application/json',
        }
        data = {"url": url}
        data_json = json.dumps(data)
        post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
        response = requests.post(post_url, headers=headers, data=data_json)
        
        # 如果请求失败，返回相应的消息
        if response.status_code != 200:
            print(f"HTTP请求失败，状态码为 {response.status_code}")
            return

        # 使用BeautifulSoup提取文本内容
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text()


    # Print and summarize (if needed) the extracted content
    print("CONTENTTTTTT:", text)
    if len(text) > 1:
        output = summary(objective, text)
        return output
    else:
        return text


def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,
        combine_prompt=map_prompt_template,
        verbose=True
    )

    output = summary_chain.run(input_documents=docs, objective=objective)

    return output


class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    objective: str = Field(
        description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    args_schema: Type[BaseModel] = ScrapeWebsiteInput

    def _run(self, objective: str, url: str):
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")



# 3. 使用上述工具创建langchain代理
tools = [
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""As an expert in retrieving and explaining SEC filings for companies, I specialize in providing detailed insights and clear explanations about various filings, such as 10-K, 10-Q, and 8-K reports, from the U.S. Securities and Exchange Commission (SEC). I can assist users in understanding the financial health, business strategies, and potential risks associated with different companies based on their SEC filings. My responses should prioritize accuracy and clarity, ensuring that financial terms and concepts are explained in a user-friendly manner.

As the user upload a SEC filling and ask a question, I shall read the whole document carefully, and answer questions according to the filling and my insights. 

If the user requests for a specific SEC filling with a stock ticker, I shall use the getSECFilings tool provided to fetch the filling and return to the user. 

If the user do not upload a SEC filling, but he asks questions related to a company. I shall firstly use bing search to search for information relevant to the user query,  think carefully, and then give an accurate question to the user. 

When faced with ambiguous queries, I should seek clarification to provide the most relevant information. My personality is informative and professional, aiming to deliver responses that are both educational and accessible to users with varying levels of financial knowledge. """
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}

llm_agent= ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613", max_tokens=1000)
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm_agent, max_token_limit=1000)

agent = initialize_agent(
    tools,
    llm_agent,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    agent_kwargs=agent_kwargs,
    memory=memory,
)
# generate texts
@app.route('/api/agent', methods=['POST'])
def agent_route():
    try:
        querys = request.get_json()
        agents=querys["log"]
        

        result = agent({"input":agents })
   
        
        results=result['output']
        print(results)

        return jsonify({'success': True, 'data': results})  
    except Exception as e:
        app.logger.error(str(e))  # error log
        return jsonify({'success': False, 'errmsg': str(e)}), 500  # return error response
# 搜索测试完
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)  # read PDF file 
        for page in pdf_reader.pages:  # iterate PDF pages
            text += page.extract_text()  # extract texts提取文本
    return text  # return all pdf texts


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    # split text into chunks
    chunks = text_splitter.split_text(text)
    return chunks  


# def get_vectorstores(filename,text_chunks):
#     try:
#         embeddings = NVIDIAEmbeddings()
#         loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")
#         docs = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
#         documents=text_splitter.split_documents(docs)
#         print(documents[0])

#         retriever =FAISS.from_documents(documents,embeddings)
#         print(retriever)
#         return retriever 
#     except Exception as e:
#         app.logger.error(str(e))
#         return None  # failed to create vector store
def get_vectorstore(filename,text_chunks):
    try:
        embeddings = OpenAIEmbeddings()
        pc = Pinecone( api_key=os.environ.get("160ced0a-5ffc-449a-9533-886efba0ef8b"))
        # pinecone.init(api_key="160ced0a-5ffc-449a-9533-886efba0ef8b", environment="us-east-1-aws")
    
        index_name = filename
        print(index_name,789)
        print(pc.list_indexes().names(),745)

        if index_name not in pc.list_indexes().names():
            # we create a new index
            pc.create_index(
                name=index_name,
                metric="cosine",
                dimension=1536 ,
                spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
            )
            vectorstore = PineconeVectorStore.from_texts(text_chunks, embeddings, index_name=index_name) # create new vectorstore
            
        else: # if index exists, load it
            vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)

        return vectorstore
    except Exception as e:
        app.logger.error(str(e))
        return None  # failed to create vector store

def get_similar_docs(query, vectorstore,k=1, score=False):

    similar_docs = vectorstore.similarity_search_with_score(query)
    # similar_docs = vectorstore.similarity_search(query)

    return similar_docs
    

def get_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name='gpt-4')
    if vectorstore is None:
        # Handle situations where vector storage was not created correctly
        return None
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qa
def get_qa_chain_page(vectorstore):
    llm = ChatOpenAI(model_name='gpt-4')
    if vectorstore is None:
        # Handle situations where vector storage was not created correctly
        return None
    
    qapage = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    return qapage

from langchain_core.runnables import chain
# ChatNVIDIA官网方法
# def get_qa_chains(allmodel,vectorstore,question):
#     client = OpenAI(
#     base_url = "https://integrate.api.nvidia.com/v1",
#     api_key = "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"
#     )

#     completion = client.chat.completions.create(
#     model="mistralai/mixtral-8x7b-instruct-v0.1",
#     messages=[{"role":"user","content":question}],
#     temperature=0.5,
#     top_p=1,
#     max_tokens=1024,
#     stream=True
#     )
#     for chunk in completion:
#         print(chunk)
#         if chunk.choices[0].delta.content is not None:
#             print(chunk.choices[0].delta.content, end="")
#     return 1
def get_qa_chains(allmodel,vectorstore,query):
    nvidiaAnswe=[]
# Replicate方法
    # llm = Replicate (
    # #  model = "mistralai/mistral-7b-instruct-v0.1:5fe0a3d7ac2852264a25279d1dfb798acbc4d49711d126646594e212cb821749" , 
    #  model = allmodel
    # )
    # if vectorstore is None:
    #     # Handle situations where vector storage was not created correctly
    #     return None
    # qa = ConversationalRetrievalChain.from_llm( 
    #     llm, 
    #     vectorstore.as_retriever(search_kwargs={ 'k' : 2 }), 
    #     return_source_documents= True
    # )
    # youtobe上的方法
    model = ChatNVIDIA(model=allmodel)
    retriever=vectorstore.as_retriever()
    hyde_template ="""Even if you do not the full answer, generate a one-paragraph hypothetical anto the below question:
{question}"""

    hyde_prompt = ChatPromptTemplate.from_template(hyde_template)

    hyde_retriever = hyde_prompt | model | StrOutputParser ()
    @chain
    def hyde_ret(question):
        hypothetical_document =hyde_retriever.invoke({"question":question})
        return retriever.invoke(hypothetical_document)
    template ="""Answer the question based only on the following context:{context}Question: {question}"""
    prompt = ChatPromptTemplate.from_template(template)
    answer_chain=prompt | model | StrOutputParser ()
    @chain
    def final_chain(question):
        documents =hyde_ret.invoke({"question" :question})
        for s in answer_chain.stream({"question": question,"context": documents}):
            nvidiaAnswe.append(s)
    for s in final_chain.stream(query):
        print(s, end="")
        sentence=""
        sentence = "".join(nvidiaAnswe)  
        return sentence

def _download_filing(
    cik: str, filing_type: str, output_dir: str, limit=None, before=None, after=None
):
    dl = Downloader(company_name=cik, email_address=EMAIL, download_folder=output_dir)
    dl.get(filing_type, cik, limit=limit, before=before, after=after, download_details=True)
    _convert_to_pdf(output_dir)

  
def _convert_to_pdf(output_dir: str):
    """Converts all html files in a directory to pdf files."""

    # NOTE: directory structure is assumed to be:
    # output_dir
    # ├── sec-edgar-filings
    # │   ├── AAPL
    # │   │   ├── 10-K
    # │   │   │   ├── 0000320193-20-000096
    # │   │   │   │   ├── filing-details.html
    # │   │   │   │   ├── filing-details.pdf   <-- this is what we want

    data_dir = Path(output_dir) / "sec-edgar-filings"
    for cik_dir in data_dir.iterdir():
        if not cik_dir.is_dir() or cik_dir.name == ".DS_Store":
            continue
        for filing_type_dir in cik_dir.iterdir():
            if not filing_type_dir.is_dir() or filing_type_dir.name == ".DS_Store":
                continue
            for filing_dir in filing_type_dir.iterdir():
                filing_doc = filing_dir / "primary-document.html" # originally be in .txt
                filing_pdf = filing_dir / "priary-document.pdf" # convert to .pdf
                if filing_pdf.exists():
                    

                    OBJECT_NAME=str(filing_pdf).split("/")[-2]+'.pdf'
                    #OBJECT_NAME="file.pdf"
                    file_path = filing_pdf
                    s3.upload_file(file_path, S3_BUCKET, OBJECT_NAME)
                    object_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{OBJECT_NAME}"
                    return object_url
                if filing_doc.exists() and not filing_pdf.exists():
                    print("- Converting {}".format(filing_doc))
                    input_path = str(filing_doc)
                    output_path = str(filing_pdf)
                  
                    try:
                        # windows 用到
                        #path_wkthmltopdf = r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe'
                        #config = pdfkit.configuration(wkhtmltopdf=path_wkthmltopdf)
                        pdfkit.from_file(input_path, output_path, verbose=True)

                        # pdfkit.from_file(input_path, output_path, verbose=True)
                    except Exception as e:
                        print(f"Error converting {input_path} to {output_path}: {e}")
                        


def main(
    secClk,
    secType,
    afters,
    befores,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    ciks: List[str] = DEFAULT_CIKS,
    file_types: List[str] = DEFAULT_FILING_TYPES,
    before: Optional[str] = None,
    after: Optional[str] = None,
    limit: Optional[int] = 1,
    convert_to_pdf: bool = True,
):
    after=afters
    before=befores
    ciks=secClk
    file_types=secType
    print(ciks,899)
    print(file_types,7899)

    print('Downloading filings to "{}"'.format(Path(output_dir).absolute()))
    print("File Types: {}".format(file_types))
    if convert_to_pdf:
        for symbol, file_type in product(ciks, file_types):
         try:
            if filing_exists(symbol, file_type, output_dir):
                print(f"- Filing for {symbol} {file_type} already exists, skipping")
            else: 
                print(f"- Downloading filing for {symbol} {file_type}")
                _download_filing(symbol, file_type, output_dir, limit, before, after)
         except Exception as e:
            print(
                f"Error downloading filing for symbol={symbol} & file_type={file_type}: {e}"
            )
            
    if convert_to_pdf:
        print("Converting html files to pdf files")
        pdfurl=_convert_to_pdf(output_dir)
        return pdfurl
    i
@app.route('/api/information', methods=['POST'])
def information():
    try:
       querys = request.get_json()
       folder_path = 'data/sec-edgar-filings' # 替换为你的文件夹路径

       shutil.rmtree(folder_path)
       os.mkdir(folder_path)
       print(DEFAULT_CIKS)
       secClk=[]
       secClk.append(querys['secClk'])
  
       secType=[]
       secType.append(querys['secType'])

       afters=querys['time']+"-01-01"
       befores=querys['time']+"-12-31"
 

       pdfurl=Fire(main(secClk,secType,afters,befores))
       print(pdfurl)
       return jsonify({'success': True, 'data':pdfurl})  # return success response and data
    except Exception as e:
        app.logger.error(str(e))  # error log
        return jsonify({'success': False, 'errmsg': str(e)}), 500  # return error response

text_chunks = None  # Split text into small pieces
# upload PDF to c3
@app.route('/api/c3', methods=['POST'])
def upload():
    try:
        # get uploaded file
        file = request.files['file']

        print( file.filename)
        s3.upload_fileobj(file, S3_BUCKET, file.filename)  # upload file object to S3
        url = s3.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': S3_BUCKET,
                'Key': file.filename
            }
        )


        return jsonify({'success': True, 'data': url})  # return success response and content of the pdf
    except Exception as e:
        app.logger.error(str(e))  # error log
        return jsonify({'success': False, 'errmsg': str(e)}), 500  # return error response
# generate texts

@app.route('/api/data', methods=['POST'])
def upload_file():
    try:
        global text_chunks
        # global filename
        file = request.files['file']
        # filename = file.filename.replace(".pdf", "")  
        pdf_docs = request.files.getlist('file')
        # print("The filename is :", filename)
        pdf_texts = get_pdf_text(pdf_docs)  # get pdf texts
        text_chunks = get_text_chunks(pdf_texts)  # get text chunks
        return jsonify({'success': True, 'data': pdf_texts})  
    except Exception as e:
        app.logger.error(str(e))  # error log
        return jsonify({'success': False, 'errmsg': str(e)}), 500  # return error response
        
# send chat message
@app.route('/api/datas', methods=['POST'])
def reciveLog():
    try:
        querys = request.get_json()
        query=querys["log"]
        allmodel=querys["allmodel"]
        print(query)
        filename=querys["pdfname"]
        filename=filename.replace(".pdf", "")
        #print(text_chunks)
        
        vectorstore = get_vectorstore(filename, text_chunks)  # 创建 vectorstore 并保存到全局

        qa = get_qa_chain(vectorstore) # qa for openai chat
        my_time = []  
        qapage = get_qa_chain_page(vectorstore) # qa for openai chat
        start_times = time.time()  
        chatlAnswer = qa({"query": query})
        if chatlAnswer and qa:
            end_time = time.time()  
            # 计算请求时间并打印  
            request_time = end_time - start_times 
            my_time.append({'chatTime':request_time})       
        chatlAnswers = qapage({"query": query+'What is the first specific page obtained from this file for this information'})
        start_time = time.time()  
        chat_history = []
       
        allmodelAnswer = get_qa_chains(allmodel,vectorstore ,query) # qa for all model
        # allmodelAnswer = qas({ 'question' : query, 'chat_history' : chat_history})
        # allmodelAnswer=hyde_retriever.invoke({"question":query})
        if allmodelAnswer:
            end_time = time.time()  
            # 计算请求时间并打印  
            request_time = end_time - start_time 
            my_time.append({'modelTime':request_time})  
        return jsonify({'success': True, 'data': {'chatlAnswer': chatlAnswer["result"],'chatlAnswers': chatlAnswers["result"], 'allmodelAnswer': allmodelAnswer,'my_time':my_time}})
    except Exception as e:
        app.logger.error(str(e))  
        return jsonify({'success': False, 'errmsg': str(e)}), 500 
# send chat message for comparison 
@app.route('/api/allcontentsapi', methods=['POST'])
def allcontents():
    try:
        querys = request.get_json()
        query=querys["log"]
        print(query)
        filename=querys["pdfname"]
        filename=filename.replace(".pdf", "")
        #print(text_chunks)
        vectorstore = get_vectorstore(filename,text_chunks)  # create vectorstore and store to global
        qa = get_qa_chain(vectorstore)
        result = qa({"query": query})
        return jsonify({'success': True, 'data': result["result"]})  
    except Exception as e:
        app.logger.error(str(e))  
        return jsonify({'success': False, 'errmsg': str(e)}), 500     

# get s3 list
@app.route('/api/list', methods=['POST'])
def reciveLogs():
    try:
        # s3 = boto3.resource('s3')

        # bucket_name = 'chatpdf-team815'
        # bucket = s3.Bucket(bucket_name)

        # pdf_files = []

        # for obj in bucket.objects.filter(Prefix='path/to/pdf/files/'):
        #     if obj.key.endswith('.pdf'):
        #         pdf_files.append(obj.key)

        # specify the bucket and key of the file to retrieve

        # retrieve files from S3
        response = s3.list_objects(Bucket='chatpdf-team815')        
        return jsonify({'success': True, 'data':response})  # return success response and data
    except Exception as e:
        app.logger.error(str(e))  # error log
        return jsonify({'success': False, 'errmsg': str(e)}), 500  # return error response
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=1234)  # Run the Flask application, allowing debugging mode, supporting remote access, and specifying ports

