import chainlit as cl
import time
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2
)

@cl.on_chat_start
async def on_chat_start():
    await cl.Message("📄 Welcome to Ragified PDF!\nPlease upload a PDF to begin.").send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content="📤 Please upload a PDF to begin.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=200,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"📄 Processing `{file.name}`...")
    await msg.send()

    start_time = time.time()

    # Load PDF
    loader = PyPDFLoader(file.path)
    pages = loader.load()

    msg.content = "📄 PDF loaded. Splitting into chunks..."
    await msg.update()

    # Extract text and metadata
    page_text = [page.page_content for page in pages]
    meta_data = [page.metadata for page in pages]
    full_text = "\n\n".join(page_text)

    # Chunking
    chunk_start = time.time()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    chunks = text_splitter.create_documents([full_text])
    chunk_time = time.time() - chunk_start

    msg.content = f"🔗 Chunking done in {chunk_time:.2f}s. Generating embeddings..."
    await msg.update()

    # Embeddings
    msg.content = "🧠 Generating embeddings for chunks..."
    await msg.update()

    try:
        embed_start = time.time()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(chunks, embeddings)

        cl.user_session.set("vector_store", vector_store)
        embed_time = time.time() - embed_start

        msg.content = f"✅ Embeddings done in {embed_time:.2f}s. Vector store ready!"
        await msg.update()

    except Exception as e:
        msg.content = f"❌ Embedding failed: {str(e)}"
        await msg.update()

        total_time = time.time() - start_time
        await cl.Message(f"✅ Ready for questions! (Total time: {total_time:.2f}s)").send()


@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content
    vector_store = cl.user_session.get("vector_store")

    if not vector_store:
        await cl.Message("⚠️ Please upload a PDF file first.").send()
        return

    retrieval_start = time.time()
    result = vector_store.similarity_search(query, k=3)
    retrieval_time = time.time() - retrieval_start

    context = "\n".join([doc.page_content for doc in result])

    prompt = f"""
        You are a helpful assistant answering based only on the resume content below.

        User question: {query}

        Context:
        {context}

        Please answer using only the context above in a concise and relevant manner.
    """

    response_start = time.time()
    response = llm.invoke(prompt)
    response_time = time.time() - response_start

    await cl.Message(
        f"🧠 **Response** (retrieved in {retrieval_time:.2f}s, LLM replied in {response_time:.2f}s):\n\n{response.content}"
    ).send()
