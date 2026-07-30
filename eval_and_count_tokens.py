import json, argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", type=str, required=True)
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_examples", type=int, default=20)   # light
    p.add_argument("--max_new_tokens", type=int, default=40)
    p.add_argument("--no_repeat_ngram_size", type=int, default=3)
    p.add_argument("--repetition_penalty", type=float, default=1.3)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device}")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_dir)
    model = GPT2LMHeadModel.from_pretrained(args.model_dir).to(device)
    model.eval()

    rows = []
    with open(args.data_path) as f:
        for i, line in enumerate(f):
            if i >= args.max_examples:
                break
            rows.append(json.loads(line))

    correct = 0
    total_tokens = 0
    results_log = []

    for i, row in enumerate(rows):
        prompt = f"Q: {row['question']}\nReasoning:"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                     pad_token_id=tokenizer.eos_token_id, do_sample=False,
                                     no_repeat_ngram_size=args.no_repeat_ngram_size,
                                     repetition_penalty=args.repetition_penalty)

        gen_ids = output[0][prompt_len:]
        n_gen = len(gen_ids)
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        true_answer = str(row["answer"]).strip().lower()
        is_correct = true_answer in gen_text.strip().lower()

        correct += int(is_correct)
        total_tokens += n_gen
        results_log.append({"question": row["question"], "true_answer": row["answer"],
                             "generated_text": gen_text, "n_generated_tokens": n_gen, "correct": is_correct})

    accuracy = correct / len(rows) if rows else 0.0
    avg_tokens = total_tokens / len(rows) if rows else 0.0

    print(f"\nRESULTS for {args.model_dir}")
    print(f"  Examples: {len(rows)}  Accuracy: {accuracy:.2%}  Avg tokens: {avg_tokens:.1f}")
    print("Use accuracy/avg_tokens as inputs to breakeven_calculator.py")

    with open(f"{args.model_dir}/eval_results.json", "w") as f:
        json.dump({"accuracy": accuracy, "avg_generated_tokens": avg_tokens,
                    "n_examples": len(rows), "per_example": results_log}, f, indent=2)


if __name__ == "__main__":
    main()
