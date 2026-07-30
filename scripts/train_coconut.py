"""
Simplified single-file Coconut reimplementation (Hao et al. 2024 concept).
NOT the official facebookresearch/coconut repo. Latent substitution is
applied only during TRAINING; eval_and_count_tokens.py generates the
<|latent|> token as an ordinary discrete token at inference -- state this
as a limitation in your methods section.
"""
import json, argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from energy_logger import EnergyLogger, REFERENCE_TDP_WATTS

LATENT_TOKEN = "<|latent|>"


class StagedReasoningDataset(Dataset):
    def __init__(self, path, tokenizer, stage, max_examples=None, max_length=640):
        self.rows = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_examples and i >= max_examples:
                    break
                self.rows.append(json.loads(line))
        self.tokenizer, self.stage, self.max_length = tokenizer, stage, max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        steps = row["reasoning_steps"]
        n_latent = min(self.stage, len(steps))
        latent_part = (LATENT_TOKEN + " ") * n_latent
        text_part = " ".join(steps[n_latent:])
        full_text = f"Q: {row['question']}\nReasoning: {latent_part}{text_part}\nA: {row['answer']}{self.tokenizer.eos_token}"
        enc = self.tokenizer(full_text, truncation=True, max_length=self.max_length,
                              padding="max_length", return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def add_latent_token(tokenizer, model):
    if LATENT_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": [LATENT_TOKEN]})
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(LATENT_TOKEN)


def replace_latent_embeddings_with_hidden_state(model, input_ids, inputs_embeds, latent_id):
    """Swap each <|latent|> embedding for the PRECEDING position's hidden state.
    FIX: use zero-padded shift, not torch.roll (roll wraps the last position
    around to position 0, injecting garbage/EOS hidden state at sequence start)."""
    with torch.no_grad():
        hidden = model.transformer(inputs_embeds=inputs_embeds, output_hidden_states=True).hidden_states[-1]
    latent_mask = (input_ids == latent_id)
    new_embeds = inputs_embeds.clone()
    shifted_hidden = torch.zeros_like(hidden)
    shifted_hidden[:, 1:, :] = hidden[:, :-1, :]  # position 0 has no predecessor -> stays zero
    new_embeds[latent_mask] = shifted_hidden[latent_mask]
    return new_embeds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--max_examples", type=int, default=60)     # light
    p.add_argument("--max_latent_stage", type=int, default=2)  # light (was 3)
    p.add_argument("--epochs_per_stage", type=int, default=1)  # light (was 2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--device_name", type=str, default="cpu_laptop", choices=list(REFERENCE_TDP_WATTS.keys()))
    p.add_argument("--output_dir", type=str, default="../coconut_model")
    p.add_argument("--max_length", type=int, default=640)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[coconut] device={device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    latent_id = add_latent_token(tokenizer, model)
    model.to(device)

    logger = EnergyLogger(args.device_name, REFERENCE_TDP_WATTS[args.device_name])
    logger.start()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    # match baseline: same LR schedule so only the mechanism differs
    steps_per_epoch = -(-args.max_examples // args.batch_size)
    total_steps = steps_per_epoch * args.epochs_per_stage * (args.max_latent_stage + 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for stage in range(args.max_latent_stage + 1):
        print(f"\n=== stage {stage}/{args.max_latent_stage} ===")
        dataset = StagedReasoningDataset(args.data_path, tokenizer, stage, args.max_examples, args.max_length)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

        model.train()
        for epoch in range(args.epochs_per_stage):
            epoch_loss = 0.0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                inputs_embeds = model.transformer.wte(input_ids)
                if stage > 0:
                    inputs_embeds = replace_latent_embeddings_with_hidden_state(model, input_ids, inputs_embeds, latent_id)

                loss = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels).loss
                loss.backward()
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                epoch_loss += loss.item()
            print(f"  stage {stage} epoch {epoch} avg loss: {epoch_loss/len(loader):.4f}")
        logger.checkpoint(f"stage_{stage}_done")

    logger.stop()
    result = logger.report()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.save(f"{args.output_dir}/energy_log.json")

    with open(f"{args.output_dir}/run_metadata.json", "w") as f:
        json.dump({"method": "coconut", "num_examples": args.max_examples,
                    "max_latent_stage": args.max_latent_stage, "epochs_per_stage": args.epochs_per_stage,
                    "total_stages": args.max_latent_stage + 1, **result}, f, indent=2)
    print("Done -> run eval_and_count_tokens.py next.")


if __name__ == "__main__":
    main()
