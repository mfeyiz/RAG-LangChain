import os
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from langchain_nomic import NomicEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


class BankAIService:
    def __init__(self):
        env_path = Path(__file__).parent.parent.parent / '.env'
        load_dotenv(dotenv_path=env_path)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(current_dir, "..", "vector_db", "faiss_index")

        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
        
        try:
            self.vectorstore = FAISS.load_local(
                db_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
        except Exception as e:
            print(f"Hata: Veritabanı yüklenemedi. {e}")
            self.vectorstore = None
        
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        
        self.last_search_results = []
        
        self.tools = [{
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": "Dokümantasyon veritabanında arama yapar. Sadece teknik sorular, ürün bilgileri veya belirli dokümanlardaki bilgi gerektiren sorular için kullan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Arama sorgusu"
                        }
                    },
                    "required": ["query"]
                }
            }
        }]

    def search_documents(self, query):
        docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=5)
        
        print(f"\nARAMA: {query}")
        for doc, score in docs_with_scores:
            print(f"Skor: {score:.4f} | Metin: {doc.page_content[:100]}...")
        
        MIN_SCORE_THRESHOLD = 1.1
        relevant_docs = [doc for doc, score in docs_with_scores if score < MIN_SCORE_THRESHOLD]
        
        print(f"Bulunan alakalı doküman: {len(relevant_docs)}")
        
        self.last_search_results = [
            {
                "content": doc.page_content,
                "score": float(score),
                "relevant": bool(score < MIN_SCORE_THRESHOLD)
            }
            for doc, score in docs_with_scores
        ]
        
        if not relevant_docs:
            return "İlgili doküman bulunamadı."
        
        return "\n\n".join([doc.page_content for doc in relevant_docs])

    def ask_stream(self, query):
        self.last_search_results = []
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            print(f"API Key mevcut: {bool(api_key)}")
            print(f"API Key uzunluğu: {len(api_key) if api_key else 0}")
            
            messages = [
                {
                    "role": "system",
                    "content": "Sen bir dokümantasyon asistanısın. Teknik sorular, tanımlar veya açıklamalar istendiğinde MUTLAKA search_documents fonksiyonunu kullanarak dokümanlarda ara. SADECE selamlaşma, teşekkür gibi sosyal etkileşimler için arama yapma. Kullanıcı bir şey soruyor veya açıklama istiyorsa search_documents'i kullan."
                },
                {
                    "role": "user",
                    "content": query
                }
            ]
            
            print(f"İstek gönderiliyor...")
            response = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=messages,
                tools=self.tools,
                stream=False
            )
            print(f"Yanıt alındı")
            
            message = response.choices[0].message
            print(f"Message: {message}")
            print(f"Tool calls: {message.tool_calls}")
            
            if message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": message.tool_calls
                })
                
                all_contexts = []
                for tool_call in message.tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    search_query = function_args.get("query")
                    
                    context = self.search_documents(search_query)
                    all_contexts.append(context)
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": context
                    })
                
                yield f"__SEARCH_RESULTS__{json.dumps(self.last_search_results, ensure_ascii=False)}__END_SEARCH__\n"
                
                final_response = self.client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    stream=True
                )
                
                for chunk in final_response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            else:
                print("Tool call yok, direkt cevap veriliyor")
                stream_response = self.client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=messages,
                    stream=True
                )
                
                for chunk in stream_response:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                        
        except Exception as e:
            print(f"HATA: {e}")
            import traceback
            traceback.print_exc()
            yield f"Hata oluştu: {str(e)}"