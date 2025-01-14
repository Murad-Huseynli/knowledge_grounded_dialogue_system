import gradio as gr
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)

# Assuming 'my_custom_model' is your fine-tuned model, and you have a suitable 'get_custom_model_response' function
from finetuned_back import get_model_response

finetuned = "wiki-llama-2-7b"
model = AutoModelForCausalLM.from_pretrained(finetuned)
tokenizer = AutoTokenizer.from_pretrained(finetuned)

# Global variable to maintain conversation history
conversation_history = [{"role": "system", "content": "Start of conversation."}]

def send(user_input):
    global conversation_history
    if user_input:
        # Get custom model response
        model_response = get_model_response(user_input, conversation_history, model, tokenizer)

        # Update conversation history
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": model_response})

        # Format the conversation for display
        conversation_display = format_conversation_for_display(conversation_history)
        return conversation_display

def format_conversation_for_display(conversation_history):
    display_text = ""
    for entry in conversation_history:
        if entry["role"] == "user":
            display_text += f"<strong style='font-family: Arial, sans-serif;'>You: {entry['content']}</strong><br>"
        else:
            parts = entry['content'].split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    display_text += f"Model: {part}<br>"
                else:
                    display_text += f"<pre><code>{part}</code></pre><br>"
    return display_text

# GUI setup using Gradio
iface = gr.Interface(
    fn=send,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="html",
    title="Wiki-LLAMA-2 Chatbot",
    description="Chat with your fine-tuned AI model with QLORA",
    theme="Soft"
)

# Run the Gradio app
iface.launch(share=True)
