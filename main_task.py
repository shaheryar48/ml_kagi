from flask import Flask, request, jsonify, render_template
import ctranslate2
from transformers import AutoTokenizer
import time
import os

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Initialize Flask app
app = Flask(__name__)


# Load the model and tokenizer
translator = ctranslate2.Translator("optimized_model4", device="cpu", inter_threads=4, intra_threads=4)
model_name = "lapp0/flan-t5-small-query-expansion-merged-lr-2e-4-ep-30"
tokenizer = AutoTokenizer.from_pretrained(model_name)



# Set up the template folder
TEMPLATE_FOLDER = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(TEMPLATE_FOLDER, exist_ok=True)
app.template_folder = TEMPLATE_FOLDER

@app.route('/inference', methods=['POST'])
def inference():
    try:
        # Get input data from the request
        data = request.json
        input_texts = data.get("input_texts", [])

        if not input_texts:
            return jsonify({"error": "No input texts provided"}), 400

        # Tokenize the input texts
        input_tokens = [tokenizer.tokenize(text) for text in input_texts]

        # Add end-of-sequence token if necessary
        input_tokens = [tokens + [tokenizer.eos_token] for tokens in input_tokens]

        # Record the start time
        start_time = time.time()

        # Perform translation
        results = translator.translate_batch(
            input_tokens,
            beam_size=1,  # Greedy decoding
            return_scores=False,
            max_decoding_length=24  # Limit length for faster results
        )

        # Record the end time
        end_time = time.time()

        # Decode the output
        output_texts = [tokenizer.convert_tokens_to_string(result.hypotheses[0]) for result in results]

        # Calculate the elapsed time
        elapsed_time = (end_time - start_time) * 1000

        # Return the results
        return jsonify({
            "output_texts": output_texts,
            "elapsed_time_ms": elapsed_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
def query():
    try:
        # Get input query from the request
        data = request.json
        query_text = data.get("query_text", "")

        if not query_text:
            return jsonify({"error": "No query text provided"}), 400

        # Tokenize the query
        query_tokens = tokenizer.tokenize(query_text) + [tokenizer.eos_token]

        # Record the start time
        start_time = time.time()

        # Perform translation
        result = translator.translate_batch(
            [query_tokens],
            beam_size=1,  # Greedy decoding
            return_scores=False,
            max_decoding_length=24  # Limit length for faster results
        )

        # Record the end time
        end_time = time.time()

        # Decode the output
        output_text = tokenizer.convert_tokens_to_string(result[0].hypotheses[0])

        # Calculate the elapsed time
        elapsed_time = (end_time - start_time) * 1000

        # Return the query result
        return jsonify({
            "query_text": query_text,
            "response_text": output_text,
            "elapsed_time_ms": elapsed_time
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        query_text = request.form.get('query_text', '')
        if not query_text:
            return render_template('index.html', response="Please enter a query.")

        # Tokenize the query
        query_tokens = tokenizer.tokenize(query_text) + [tokenizer.eos_token]

        # Perform translation
        start_time = time.time()
        result = translator.translate_batch(
            [query_tokens],
            beam_size=1,  # Greedy decoding
            return_scores=False,
            max_decoding_length=24  # Limit length for faster results
        )
        end_time = time.time()

        # Decode the output
        output_text = tokenizer.convert_tokens_to_string(result[0].hypotheses[0])
        elapsed_time = (end_time - start_time) * 1000

        # Return the result to the web page
        return render_template('index.html', response=output_text, elapsed_time=f"{elapsed_time:.2f} ms")

    return render_template('index.html')

# Create a simple HTML template
index_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Query Inference</title>
</head>
<body>
    <h1>Model Query Interface</h1>
    <form method="POST" action="/">
        <label for="query_text">Enter your query:</label><br>
        <input type="text" id="query_text" name="query_text" required><br><br>
        <button type="submit">Submit</button>
    </form>

    {% if response %}
        <h2>Response:</h2>
        <p>{{ response }}</p>
        <p>Elapsed Time: {{ elapsed_time }}</p>
    {% endif %}
</body>
</html>
"""

# Save the template
with open(os.path.join(TEMPLATE_FOLDER, 'index.html'), 'w') as f:
    f.write(index_html)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)