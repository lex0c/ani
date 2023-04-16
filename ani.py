import sys
import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor


MODEL_CACHE = {}


def load_model(model_name):
    if model_name in MODEL_CACHE:
        return MODEL_CACHE[model_name]

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    MODEL_CACHE[model_name] = (tokenizer, model)

    return tokenizer, model


def generate_text(prompt, model, tokenizer, max_length=50, temperature=1.0, num_return_sequences=1, top_p=0.9, top_k=50):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)
    pad_token_id = tokenizer.eos_token_id

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            temperature=temperature,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            top_p=top_p,
            top_k=top_k,
            do_sample=True
        )

    return [tokenizer.decode(output_seq, skip_special_tokens=True) for output_seq in output]


def generate_text_wrapper(args):
    prompt, model, tokenizer, max_length, temperature, num_return_sequences, top_p, top_k = args
    return generate_text(prompt, model, tokenizer, max_length, temperature, num_return_sequences, top_p, top_k)


def main():
    parser = argparse.ArgumentParser(description="Generate text using AI")
    parser.add_argument("prompts", nargs="+", help="Text prompts")
    parser.add_argument("-m", "--model", default="gpt2", choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], help="Model size")
    parser.add_argument("-l", "--length", type=int, default=50, help="Maximum length of generated text")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("-n", "--num-sequences", type=int, default=1, help="Number of sequences to return")
    parser.add_argument("-p", "--top-p", type=float, default=0.9, help="Nucleus sampling: cumulative probability threshold")
    parser.add_argument("-k", "--top-k", type=int, default=50, help="Top-K sampling: number of highest-probability words to sample from")
    parser.add_argument("-c", "--processes", type=int, default=1, help="Number of processes to use")
    args = parser.parse_args()

    if args.length < 1 or args.length > 1024:
        print("Error: Length must be between 1 and 1024.")
        sys.exit(1)

    if args.temperature < 0.1 or args.temperature > 10.0:
        print("Error: Temperature must be between 0.1 and 10.0.")
        sys.exit(1)

    try:
        tokenizer, model = load_model(args.model)
    except Exception as e:
        print(f"Error loading GPT-2 model: {e}")
        sys.exit(1)

    try:
        generator = generate_text_wrapper if args.processes == 1 else generate_text

        with ThreadPoolExecutor(max_workers=args.processes) as executor:
            results = list(executor.map(generator, [(prompt, model, tokenizer, args.length, args.temperature, args.num_sequences, args.top_p, args.top_k) for prompt in args.prompts]))

        print()

        for result in results:
            print(result)
    except Exception as e:
        print(f"Error generating text: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

