import gradio as gr
from transformers import pipeline

# Initialize the chatbot using Hugging Face's pre-trained GPT model (e.g., distilgpt2 for efficiency)
chatbot = pipeline("text-generation", model="distilgpt2", tokenizer="distilgpt2")

# Function to generate chatbot responses
def chatbot_response(user_input):
    """Generate a response to the user's input."""
    # Use the text-generation pipeline to generate a reply
    response = chatbot(user_input, max_length=100, num_return_sequences=1)
    return response[0]['generated_text']

# Create Gradio interface for the chatbot
interface = gr.Interface(
    fn=chatbot_response,  # The function to generate the response
    inputs=[gr.Textbox(label="Ask a Question", placeholder="Type your query here...")],  # User input (query)
    outputs=[gr.Textbox(label="Chatbot Response")],  # The generated response (output)
    title="AI Chatbot for Customer Support",  # Interface title
    description="This chatbot will provide responses to customer support queries.",  # Description
    live=True  # Enable live interaction
)

# Launch the Gradio interface
interface.launch()
