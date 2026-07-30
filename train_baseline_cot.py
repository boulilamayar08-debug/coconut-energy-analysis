import json, argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from energy_logger import EnergyLogger, REFERENCE_TDP_WATTS


class ReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, max_examples=None, max_length=640):
        self.examples = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                row = json.loads(line)
                text = f"Q: {row['question']}\nReasoning: {row['reasoning']}\nA: {row['answer']}{tokenizer.eos_token}"
                self.examples.append(text)
        self.tokenizer, self.max_length = tokenizer, max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.examples[idx], truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_examples", type=int, default=60)   # light
    p.add_argument("--epochs", type=int, default=2)          # light
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device_name", type=str, default="cpu_laptop", choices=list(REFERENCE_TDP_WATTS.keys()))
    p.add_argument("--output_dir", type=str, default="../baseline_cot_model")
    p.add_argument("--max_length", type=int, default=640)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[baseline_cot] device={device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    dataset = ReasoningDataset(args.data_path, tokenizer, args.max_examples, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    logger = EnergyLogger(args.device_name, REFERENCE_TDP_WATTS[args.device_name])
    logger.start()

    model.train()
    step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            epoch_loss += loss.item(); step += 1
            if step % 10 == 0:
                print(f"  epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"[epoch {epoch}] avg loss: {epoch_loss/len(loader):.4f}")
        logger.checkpoint(f"epoch_{epoch}_done")

    logger.stop()
    result = logger.report()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.save(f"{args.output_dir}/energy_log.json")

    with open(f"{args.output_dir}/run_metadata.json", "w") as f:
        json.dump({"method": "baseline_cot", "num_examples": len(dataset), "epochs": args.epochs,
                    "batch_size": args.batch_size, "total_steps": total_steps, **result}, f, indent=2)
    print("Done -> run eval_and_count_tokens.py next.")


if __name__ == "__main__":
    main()
