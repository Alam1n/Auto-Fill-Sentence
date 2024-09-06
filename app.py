from flask import Flask, request, jsonify, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

# Initialize Flask app
app = Flask(__name__)

# Define the path to your fine-tuned model and tokenizer
model_dir = "Sentence Completion\gpt2-autocomplete"

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained(model_dir)
tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

# Create a text generation pipeline
text_generator = pipeline('text-generation', model=model, tokenizer=tokenizer)

def predict_next_word(text, top_k=5):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    predictions = outputs.logits[:, -1, :]
    top_k_predictions = predictions.topk(top_k).indices[0].tolist()
    
    suggested_words = [tokenizer.decode([pred]) for pred in top_k_predictions]
    return suggested_words

def predict_next_sentence(text, max_length=50):
    generated_text = text_generator(text, max_length=max_length, num_return_sequences=1)
    full_text = generated_text[0]['generated_text']
    
    # Split the generated text into sentences and return the first complete sentence
    sentences = full_text.split('.')
    if len(sentences) > 1:
        return sentences[0] + '.'
    else:
        return full_text
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    
    word_suggestions = predict_next_word(text)
    sentence_suggestion = predict_next_sentence(text)
    
    response = {
        'word_suggestions': word_suggestions,
        'sentence_suggestion': sentence_suggestion
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
