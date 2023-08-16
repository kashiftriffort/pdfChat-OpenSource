import os
import sys
import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader

os.environ["REPLICATE_API_TOKEN"] = "r8_QA9VSTsxRjaW04wZK9IjxZwUVfOy8JR3b1eyT"

pinecone.init(api_key="5b939530-ec80-40ec-8e34-16126062dfec", environment="asia-southeast1-gcp-free")


# Load and preprocess the PDF document
#loader = PyPDFLoader("/Users/JilaniK/Downloads/fastfacts-what-is-climate-change.pdf")
#documents = loader.load()

documents = []
for file in os.listdir("docs"):
    if file.endswith(".pdf"):
        pdf_path = "./docs/" + file
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# Split the documents into smaller chunks for preprocessing
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

# Use HuggingFace embeddings for transforming text into numerical vectors
embeddings = HuggingFaceEmbeddings()

# Set up the Pinecone vector database
index_name = "chatpdf"
index = pinecone.Index(index_name)
vectordb = Pinecone.from_documents(texts, embeddings, index_name=index_name)

# Initialize Replicate Llama2 Model

llm = Replicate(
    model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
    input={"temperature": 0.9, "max_length": 3000}
)

# Set up the Conversational Retrieval Chain

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vectordb.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    verbose=True
)

yellow = "\033[0;33m"
green = "\033[0;32m"
white = "\033[0;39m"

# Start chatting with the chatbot
chat_history = []
print(f"{yellow}---------------------------------------------------------------------------------")
print('Welcome to the DocBot. You are now ready to start interacting with your documents')
print('---------------------------------------------------------------------------------')
while True:
    query = input(f"{green}Prompt: ")
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting")
        sys.exit()
    result = qa_chain({"question": query, "chat_history": chat_history})
    print(f"{white}Answer: " + result["answer"] + "\n")
    chat_history.append((query, result["answer"]))
