import requests
import json

def ask_local_llm(query):
    url = "http://localhost:11434/api/generate"

    payload = {
        "model": "llama3.2",
        "prompt": query
    }

    response = requests.post(url, json=payload, stream=True)

    final_text = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8'))
            if "response" in data:
                final_text += data["response"]

    return final_text


if __name__ == "__main__":
    question = input("Ask your question: ")
    answer = ask_local_llm(question)
    print("\n=== Answer ===\n")
    print(answer)




# What requests is used for
# - Sending HTTP requests: GET, POST, PUT, DELETE, etc.
# - Accessing web APIs: Call REST APIs and retrieve JSON or other data.
# - Downloading content: Fetch HTML pages, images, or files from the internet.
# - Sending data: Submit form data, upload files, or send JSON payloads.
# - Handling responses: Read status codes, headers, and response bodies.


#- json → Used to parse JSON strings into Python dictionaries.
 
