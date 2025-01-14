import gradio as gr
from baseline_back import get_model_response
from llama import Llama

# To run:
# python -m torch.distributed.launch /root/llama-1/testing_demo_new.py

# Initialize the model
generator = Llama.build(
    ckpt_dir="llama/llama-2-7b-chat/",
    tokenizer_path="llama/tokenizer.model",
    max_seq_len=2048,
    max_batch_size=8,
)

# Global variable to maintain conversation history
conversation_history = [{"role": "system", "content": "Start of conversation."}]

def send(user_input):
    global conversation_history
    if user_input:
        # Get model response and update conversation history
        model_response, updated_conversation_history = get_model_response(
            user_input, conversation_history, generator, None, 0.6, 0.9)

        # Update the global conversation history
        conversation_history = updated_conversation_history

        # Format the conversation for display
        conversation_display = format_conversation_for_display(conversation_history)
        return conversation_display

def format_conversation_for_display(conversation_history):
    display_text = ""
    for entry in conversation_history:
        if entry["role"] == "user":
            # User input in bold and a distinct font
            display_text += f"<strong style='font-family: Arial, sans-serif;'>You: {entry['content']}</strong><br>"
        else:
            # Split the response on triple backticks
            parts = entry['content'].split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    # Regular text
                    display_text += f"Model: {part}<br>"
                else:
                    # Code block
                    display_text += f"<pre><code>{part}</code></pre><br>"
    return display_text


# GUI setup using Gradio
iface = gr.Interface(
    fn=send,
    inputs=gr.Textbox(lines=2, placeholder="Type your message here..."),
    outputs="html",
    title="Wiki-LLama-2 Chatbot",
    description="Chat with the Wiki-Finetuned fast AI model",
    theme="Soft"
)

# Run the Gradio app
iface.launch(share=True)
