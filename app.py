import os
import sys
import threading

from dotenv import load_dotenv
from flask_cors import CORS
from together import Together
from flask_socketio import SocketIO
from flask import Flask, request, jsonify, send_from_directory

from path_handler import PathManager

path_manager = PathManager()
sys.path.append(str(path_manager.get_base_directory()))
load_dotenv()

from src.pipelines.type_hint import ControllerType
from src.pipelines.pipeline.builder import PipelineBuilder
from src.pipelines.config.default import PIPELINE_DEFAULT_CONFIG

pipeline = (
    PipelineBuilder(controller_type=ControllerType.TRAINING)
    .with_logger("digisage", "DigiSage", "development")
    .with_config(PIPELINE_DEFAULT_CONFIG)
    .with_preprocessor(True)
    .with_vectorizer(True)
    .with_vocabulary()
    .with_similarity()
    .with_evaluator(True)
    .build(True)
)
pipeline.run()

inference_pipeline = (
    PipelineBuilder(controller_type=ControllerType.INFERENCE)
    .with_logger("digisage", "DigiSage", "development")
    .with_config(PIPELINE_DEFAULT_CONFIG)
    .with_preprocessor()
    .with_vectorizer()
    .with_vocabulary()
    .with_similarity()
    .with_evaluator()
    .build()
)

client = Together(
    api_key=os.getenv("APIKEY", "92dd4356e04ea782af22b37f22673f1a69166f3c005ba97e3197c9997fd02721")
)

llm_prompt = '''
--- Objective:
You are a customer service expert. You must answer user's question using the provided set of questions and answers. 

--- User Input:
{query}

--- Context:
{retrieval}

--- Rules:
- Answer with concise and polite sentences in Persian.
'''.strip()

# Retrieval and LLM response generation functions
def get_ir_results(query):
    result = inference_pipeline.run(query)
    
    response = [
        {
            "question": result["retrieved_question"][i],
            "answer": result["retrieved_answer"][i],
            "category": result["retrieved_category"][i],
        }
        for i in range(len(result["retrieved_question"]))
    ]
    
    return response

def get_llm_response(query, ir_results):
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": llm_prompt.format(
                    retrieval=ir_results,
                    query=query)
            }
        ],
    )
    
    return response.choices[0].message.content

app = Flask(
    __name__,
    static_folder="ui"
)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="http://127.0.0.1:5000")

@app.route('/submit', methods=['POST'])
def submit_query():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "No query provided"}), 400

    threading.Thread(target=process_query, args=(query,)).start()
    
    return jsonify({"status": "Processing started"}), 202

@app.route('/', methods=['GET'])
def hi():
    return send_from_directory(
        directory="ui", 
        path="index.html"
    )

def process_query(query):
    ir_results = get_ir_results(query)
    socketio.emit('ir_results', {'ir_results': ir_results})
    
    llm_response = get_llm_response(query, ir_results)
    socketio.emit('llm_response', {'llm_response': llm_response})

@socketio.on('connect')
def on_connect():
    print("Client connected")

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)