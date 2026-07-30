import json, argparse, random


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="../data")
    p.add_argument("--n_train", type=int, default=60)   # light default for CPU
    p.add_argument("--n_eval", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with open(args.input_path) as f:
        data = json.load(f)
    print(f"Loaded {len(data)} examples.")

    random.seed(args.seed)
    random.shuffle(data)  # FIX: avoid biased ordered split

    needed = args.n_train + args.n_eval
    if len(data) < needed:
        raise ValueError(f"Only {len(data)} examples, need {needed}.")

    train_rows = data[:args.n_train]
    eval_rows = data[args.n_train:needed]

    def write_baseline(rows, path):
        with open(path, "w") as f:
            for row in rows:
                out = {"question": row["question"], "reasoning": " ".join(row["steps"]), "answer": row["answer"]}
                f.write(json.dumps(out) + "\n")

    def write_coconut(rows, path):
        with open(path, "w") as f:
            for row in rows:
                out = {"question": row["question"], "reasoning_steps": row["steps"], "answer": row["answer"]}
                f.write(json.dumps(out) + "\n")

    write_baseline(train_rows, f"{args.output_dir}/train_baseline.jsonl")
    write_baseline(eval_rows, f"{args.output_dir}/eval_baseline.jsonl")
    write_coconut(train_rows, f"{args.output_dir}/train_coconut.jsonl")
    write_coconut(eval_rows, f"{args.output_dir}/eval_coconut.jsonl")
    print(f"Wrote {len(train_rows)} train / {len(eval_rows)} eval examples (shuffled, seed={args.seed}).")


if __name__ == "__main__":
    main()
