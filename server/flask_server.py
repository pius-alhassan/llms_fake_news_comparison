from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import re
import google.generativeai as generativeai
import json
import os

app = Flask (__name__)

CORS(app)

api_key = os.getenv("GEMINI_TWO_FIVE_PRO_KEY)")
api_key = "AIzaSyABHOR14yehwTgZGDr1I0NZ9ljU6tqK4EQ"

generativeai.configure(api_key=api_key)

model = generativeai.GenerativeModel ("gemini-2.5-pro")

def strip_markdown_json(text):
    """
    Removes Markdown JSON fences like ```json ... ``` so json.loads() won't fail.
    """
    cleaned = re.sub(r"^```json", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"^```", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE)
    return cleaned.strip()


@app.route("/ask", methods=["POST"])
def ask_gemini():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        response = model.generate_content(question)
        answer = response.text if hasattr(response, "text") else response.candidates[0].content.parts[0].text
        return jsonify({"answer": answer})
    except Exception as e:
        print("Flask server error:", e)
        error_message = str(e)

        if "429" in error_message or "quota" in error_message.lower():
            return jsonify({"error": "Rate limited. Please wait a minute and try again"}), 429
        else:
            return jsonify({"error": str(e)}), 500

@app.route("/clean", methods=["POST"])
def clean_text():
    data = request.get_json()
    raw_data = data.get("text", "")
    if not raw_data:
        return jsonify({"error": "No text provided"}), 400
    try:
        prompt = (
            "Clean this text by fixing spelling, removing unnecessary characters,"
            "and formatting into clear sentences. Do not add extra information \n\n"
            f"Text: {raw_data}"
        )
        response = model.generate_content(prompt)
        cleaned_response = response.text.strip()

        with open("../data/cleaned_response.txt","w", encoding="utf-8") as writer:
            writer.write(cleaned_response)
            return jsonify({
                "message": "Response text cleaned and save",
                "preview": cleaned_response[:200]
            })
    except Exception as e:
        print("Error in cleaning operation", e)
        return jsonify({"error": str(e)}), 500
    
@app.route("/extract", methods=["GET"])
def extract_detail():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "..", "data")
    
    cleaned_path = os.path.join(DATA_DIR, "cleaned_response.txt")
    extracted_path = os.path.join(DATA_DIR, "extracted_text.json")
    if not os.path.exists(cleaned_path):
        return jsonify({"error": "cleaned text not found. Please run clean text operation again"}), 400
    try:
        with open(cleaned_path, "r", encoding="utf-8") as reader:

            cleaned_response = reader.read()
            prompt = (
                "Extract all dates, names, and locations from the following text."
                "Return the result as valid JSON with keys: dates, names, locations. \n\n"
                f"{cleaned_response}"
            )
            response = model.generate_content(prompt)
            extracted_text = response.text.strip()
            cleaned_json_text = strip_markdown_json(extracted_text)
            try:
                extracted_text_json = json.loads(cleaned_json_text)
            except json.JSONDecodeError:
                extracted_text_json = {"raw_output": extracted_text}

            with open(extracted_path, "w", encoding="utf-8") as writer:
                json.dump(extracted_text_json, writer, indent=2)

            return jsonify({
                "message": "Information extracted and save successfully",
                "data": extracted_text_json
            })
    except Exception as e:
        print("Error in text extration operation", e)
        return jsonify({"error":str (e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)


