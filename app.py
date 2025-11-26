# ==========================================================
# RAG SYSTEM - VERSI DEBUGGING (CEK MACET DIMANA)
# ==========================================================
import os
import sys
import shutil

# --- FIX KHUSUS WINDOWS ---
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Cek Library
try:
    import gradio as gr
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import Chroma
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.chains import RetrievalQA
except ImportError as e:
    sys.exit(f"❌ Error Library: {e}")

# 1. SETUP API KEY (PASTIKAN BENAR)
os.environ["OPENAI_API_KEY"] = "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"

# 2. LOAD PDF
pdf_filename = "SIBI - Sistem Informasi Perbukuan Indonesia.pdf"
pdf_path = os.path.join(os.getcwd(), pdf_filename)
persist_dir = os.path.join(os.getcwd(), "chroma_db")

print("="*40)
print("       MODE DIAGNOSA PERMASALAHAN")
print("="*40)

if not os.path.exists(pdf_path):
    print("❌ File PDF tidak ditemukan!")
    sys.exit()

# Bersihkan DB lama
if os.path.exists(persist_dir):
    try:
        shutil.rmtree(persist_dir)
    except:
        pass

try:
    print("1. [START] Membaca PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(pages)
    
    print("2. [START] Menghubungi OpenAI untuk Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    print("3. [START] Membuat Database Lokal...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("✓ Database SIAP!")

    # Chain
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    # --- FUNGSI CHAT DENGAN LAPORAN ---
    def chat_function(message, history):
        print(f"\n[USER BERTANYA]: {message}")
        try:
            print("   -> Langkah A: Mencari potongan teks di PDF...")
            # Kita tes pencarian dulu
            docs = vectorstore.similarity_search(message, k=3)
            print(f"   -> Hasil A: Ditemukan {len(docs)} referensi.")
            
            print("   -> Langkah B: Mengirim data ke ChatGPT (OpenAI)...")
            # Proses utama
            result = qa_chain.invoke({"query": message})
            print("   -> Langkah C: ChatGPT sudah membalas! Mengirim ke browser...")
            
            answer = result["result"]
            sources = result["source_documents"]
            
            response = f"{answer}\n\n---\n\n📚 **Sumber Referensi:**\n\n"
            for i, doc in enumerate(sources, 1):
                page = doc.metadata.get('page', '?')
                text = doc.page_content[:150].replace("\n", " ").strip()
                response += f"**[{i}] Halaman {page}:**\n{text}...\n\n"
            
            return response
        except Exception as e:
            print(f"❌ ERROR TERJADI: {e}")
            return f"❌ Maaf, ada error: {str(e)}"

    print("\n🚀 Membuka Browser... Silakan tanya sesuatu.")
    demo = gr.ChatInterface(
        fn=chat_function,
        title="🎓 Chatbot SIBI (Mode Debug)",
        description=f"Jika loading lama, cek terminal VS Code untuk melihat dia macet dimana.",
    )
    demo.launch(inbrowser=True)

except Exception as e:
    print(f"\n❌ ERROR UTAMA: {e}")