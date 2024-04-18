from langchain.callbacks.manager import CallbackManager
import os
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain_core.outputs import LLMResult
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio
from fastapi.responses import StreamingResponse
from typing import  Any, Awaitable,Generator
from langchain_openai import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# region Set Environment Variables
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True

)

os.environ['OPENAI_API_KEY'] = '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'
model_name = 'gpt-4-0125-preview'
temperature = 0.1
user_lang = "中文"
learning_lang = "英文"
embeddings_model = OpenAIEmbeddings()
#endregion

recursive_text_splitter = RecursiveCharacterTextSplitter(
    # separator調整為空白優先，這樣我們的 overlap 才會正常運作
    separators=[" ", "\n"],
    chunk_size = 300,
    chunk_overlap  = 100,
    length_function = len,
)
vectordb = Chroma(persist_directory='./data', embedding_function=OpenAIEmbeddings())

#region custom handler
class My_streamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    tokens=[]

    finish=False
    def on_llm_new_token(self,token,str,**kwargs)-> None:
        self.tokens.append(token)
    
    def  on_llm_end(self, response, **kwargs) -> None:
        self.finish=True
    
    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.tokens.append(str(error))

    def generate_tokens(self)->Generator:
        while not self.finish:
            if self.tokens:
                token = self.tokens.pop(0)
                yield{"data":token}
            else:
                pass
#endregion

# region Create API Function to Handle Request 
async def generator(question):

    # Create callback_handler to async-ly iterate through the output token of the llm
    callback_hadler = AsyncIteratorCallbackHandler()
    
    # Initial lize the llm with callback handler
    llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming=True, callback_manager=CallbackManager([callback_hadler]))
    
    # Assign the vectordb for the chain toi retrive from
    vectordb = Chroma(persist_directory='./data', embedding_function=embeddings_model)
    retriever = vectordb.as_retriever()
    
    # Create query chain
    # qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory,verbose=True)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever,chain_type="stuff",verbose=True)
    
    # Wrap an awaitable with a event to signal when it's done or an exception is raised
    async def wrap_done(fn: Awaitable):
        try:
            await fn
        except Exception as e:
            # handle exception
            print(f"Caught exception: {e}")
        # finally:
        #     # Signal the aiter to stop.
        #     event.set()
    
    # Begin a task that runs in the background
    run = asyncio.create_task(wrap_done(qa.arun(question)))
    # run = asyncio.create_task(qa.arun(question))
    # print(run)
    # Use server-sent-events to stream the response
    async for token in callback_hadler.aiter():
        yield token
    await run

# endregion

# callback=My_streamingStdOutCallbackHandler()
# llm = llm = ChatOpenAI(model_name=model_name, temperature=temperature, streaming=True, callback_manager=CallbackManager([callback]))

# region Create API Request Format 
class Item(BaseModel):
    query: str
# endregion

class Document(BaseModel):
    docs: str

# region Create FastAPI APP
app=FastAPI()
# 設定 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允許所有來源，生產環境應更具體指定
    allow_credentials=True,
    allow_methods=["*"],  # 允許所有方法
    allow_headers=["*"],  # 允許所有頭部
)

# 全局处理所有 OPTIONS 请求
@app.middleware("http")
async def global_options_handler(request: Request, call_next):
    if request.method == "OPTIONS":
        return Response(status_code=204, headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Credentials": "true",
        })
    return await call_next(request)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    print(f"请求方法: {request.method}, 请求路径: {request.url.path}, 响应状态: {response.status_code}")
    return response

# Explicitly handle OPTIONS for specific routes
@app.options("/conversation")
def options_conversation():
    return Response(status_code=204, headers={
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "POST, OPTIONS",
        "Access-Control-Allow-Headers": "*",
        "Access-Control-Allow-Credentials": "true",
    })

# API endpoint and url 
@app.post("/conversation")
async def get_conversation(q:Item):
    ques = q.query
    question = ques+"如果文獻中存在圖片檔名以{圖檔.png}格式包含於回答中，若無則無需提到圖檔，回答中也不須包含文獻中並未提供圖片檔案等敘述"
    return StreamingResponse(generator(question), media_type="text/event-stream")

# endregion    ，


    
# API endpoint and url 
@app.post("/newtext")
async def NewText(q:Document):
    documents = q.docs
    loader = PyPDFLoader(documents)
    pages_new = loader.load_and_split(recursive_text_splitter)
    _ = vectordb.add_documents(pages_new)
    vectordb.persist()
# endregion


if __name__ == '__main__':
    uvicorn.run("RAG_GPT_Flask_stream:app", host="0.0.0.0", port=5000, reload=True)

