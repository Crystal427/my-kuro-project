from flask import Flask, request, Response, jsonify
from openai import OpenAI
import requests
import json
import sseclient

app = Flask(__name__)

OPENAI_API_KEY = ""
OPENAI_API_BASE = "https://aihubmix.com/v1"

client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)

def process_messages(messages):
    system_content = ""
    new_messages = []
    expected_role = "user"
    
    for message in messages:
        if message['role'] == 'system':
            system_content += message['content'] + "\n"
            continue
        
        while message['role'] != expected_role:
            new_messages.append({'role': expected_role, 'content': '请注意下面的指令'})
            expected_role = "assistant" if expected_role == "user" else "user"
        
        if message['role'] == 'user' and not new_messages:
            message['content'] = system_content + "\n" + message['content']
        
        new_messages.append(message)
        expected_role = "assistant" if expected_role == "user" else "user"
    
    # 如果消息列表为空，添加一个空的user消息
    if not new_messages:
        new_messages.append({'role': 'user', 'content': system_content.strip()})
    # 确保消息列表以user结束
    elif new_messages[-1]['role'] == 'assistant':
        new_messages.append({'role': 'user', 'content': 'Nothing'})
    
    return new_messages

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    try:
        data = request.get_json(force=True)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in request body"}), 400

    # 处理消息
    data['messages'] = process_messages(data.get('messages', []))
    
    stream = data.get('stream', False)
    
    if not stream:
        try:
            response = client.chat.completions.create(**data)
            return jsonify(response.model_dump()), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        def generate():
            try:
                response = client.chat.completions.create(**data)
                for chunk in response:
                    yield f"data: {json.dumps(chunk.model_dump())}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"
        
        return Response(generate(), mimetype='text/event-stream')

@app.route('/v1/models', methods=['GET'])
def list_models():
    try:
        models = client.models.list()
        return jsonify(models.model_dump()), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):
    url = f"{OPENAI_API_BASE}/{path}"
    headers = {key: value for (key, value) in request.headers if key != 'Host'}
    headers['Authorization'] = f"Bearer {OPENAI_API_KEY}"

    resp = requests.request(
        method=request.method,
        url=url,
        headers=headers,
        data=request.get_data(),
        params=request.args,
        allow_redirects=False,
        stream=True
    )

    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items()
               if name.lower() not in excluded_headers]

    if 'text/event-stream' in resp.headers.get('Content-Type', ''):
        def generate():
            client = sseclient.SSEClient(resp)
            for event in client.events():
                yield f"data: {event.data}\n\n"
        return Response(generate(), mimetype='text/event-stream')
    
    return Response(resp.content, resp.status_code, headers)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)