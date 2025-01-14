from typing import List, Optional

import fire

from llama import Llama, Dialog
from transformers import pipeline
from transformers import AutoTokenizer

# Extract this part into a function
def get_model_response(user_input, conversation_history, model, tokenizer):
    # Format the prompt
    conversation_history.append({"role": "user", "content": user_input})

    # Initialize the pipeline
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)


    # Generate response
    result = pipe(user_input)

    # Extract the generated text
    model_response = result[0]['generated_text']

    return model_response


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,       # 0.6 by default
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    conversation_history: List[Dialog] = [{"role": "system", "content": "Start of conversation."}]

    while True:
        # User input
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Append user input to the conversation history as 'user' role
        conversation_history.append({"role": "user", "content": user_input})

        # Generate response
        results = generator.chat_completion(
            [conversation_history],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract model response and append to the conversation history as 'assistant' role
        model_response = results[0]['generation']['content']
        conversation_history.append({"role": "assistant", "content": model_response})

        # Display response
        print(f"> Model: {model_response}")
        print("\n==================================\n")

if __name__ == "__main__":
    fire.Fire(main)
