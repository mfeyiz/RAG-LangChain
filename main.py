from flask import Flask, request, Response, send_from_directory
import requests
import json 

app = Flask(__name__)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/chat', methods=['POST'])
def chat_endpoint():
    data = request.json
    user_message = data.get("message")
    return Response(chat(user_message), mimetype='text/event-stream')

def chat(user_message):
    url = "http://localhost:11434/api/chat"
    payload = {
        "model": "llama3.1",
        "messages": [{"role": "user", "content": user_message}],
        "stream": True
    }

    # with ifadesi, HTTP bağlantısını yönetir ve otomatik olarak kapatır. Context manager kullanımı
    with requests.post(url, json=payload, stream=True) as r:
        for line in r.iter_lines():
            if line:
                try:
                    chunk = json.loads(line.decode('utf-8'))
                    content = chunk.get('message', {}).get('content', '')
                    if content:
                        # SSE formatında gönder - içeriği olduğu gibi koru
                        yield f"data: {content}\n\n"
                        # Flush yapmak için
                except json.JSONDecodeError:
                    continue
            
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)