import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name="gpt2", device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    print(f"Using device: {device}")

    try:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    except Exception as e:
        raise RuntimeError(f"Error loading model/tokenizer: {e}")

    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return tokenizer, model, device


def generate_response(tokenizer, model, device, prompt, max_new_tokens=50):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=False,
        )

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Extract generated tokens (only the new tokens after the prompt)
        generated_tokens = output_ids[0, input_ids.shape[-1]:]

        if generated_tokens.numel() == 0:
            # Fallback: decode whole output
            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            # Remove prompt prefix if present
            if prompt and full_text.startswith(prompt):
                return full_text[len(prompt):].strip()
            return full_text.strip()

        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return response

    except Exception as e:
        return f"Sorry, an error occurred during generation: {e}"


def main():
    model_name = "gpt2"
    if len(sys.argv) > 1:
        model_name = sys.argv[1]

    tokenizer, model, device = load_model(model_name)

    print("AI Assistant: Hello! Type 'quit' to exit.")

    conversation_history = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAI Assistant: Goodbye!")
            break

        if user_input.lower() == "quit":
            print("AI Assistant: Goodbye!")
            break

        # Build prompt with optional short history
        if conversation_history:
            history = "\n".join(conversation_history[-4:])
            prompt = f"{history}\nUser: {user_input}\nAI:"
        else:
            prompt = f"User: {user_input}\nAI:"

        response = generate_response(tokenizer, model, device, prompt)

        # Update history
        conversation_history.append(f"User: {user_input}")
        conversation_history.append(f"AI: {response}")

        print("AI Assistant:", response)


if __name__ == "__main__":
    main()
