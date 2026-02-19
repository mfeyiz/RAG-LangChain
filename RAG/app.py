from flask import Flask, request, Response, send_from_directory, jsonify
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from RAG.services.llm_service import BankAIService 

STATIC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
ai_service = BankAIService()


@app.route('/')
def index():
    return send_from_directory(STATIC_DIR, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


@app.route('/ask', methods=['POST'])
def handle_query():
    data = request.json
    user_query = data.get("query")
    
    if not user_query:
        return jsonify({"error": "Query alanı boş olamaz"}), 400
        
    def generate():
        for chunk in ai_service.ask_stream(user_query):
            yield f"data: {chunk}\n\n"

    return Response(generate(), mimetype='text/event-stream')

            
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)


