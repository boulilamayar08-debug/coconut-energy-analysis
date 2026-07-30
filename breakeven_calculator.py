import json, argparse


def load_run(path):
    with open(f"{path}/run_metadata.json") as f:
        return json.load(f)


def energy_per_query_kwh(avg_tokens, joules_per_token):
    return (avg_tokens * joules_per_token) / 3_600_000.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline_dir", type=str, required=True)
    p.add_argument("--coconut_dir", type=str, default=None)
    p.add_argument("--coconut_training_kwh", type=float, default=None)
    p.add_argument("--baseline_avg_tokens", type=float, required=True)
    p.add_argument("--coconut_avg_tokens", type=float, required=True)
    p.add_argument("--joules_per_token", type=float, default=2.61)
    args = p.parse_args()

    baseline_kwh = load_run(args.baseline_dir)["energy_kwh"]
    if args.coconut_dir:
        coconut_kwh = load_run(args.coconut_dir)["energy_kwh"]
    elif args.coconut_training_kwh is not None:
        coconut_kwh = args.coconut_training_kwh
    else:
        raise ValueError("Provide --coconut_dir or --coconut_training_kwh")

    extra_training_kwh = coconut_kwh - baseline_kwh
    e_baseline = energy_per_query_kwh(args.baseline_avg_tokens, args.joules_per_token)
    e_coconut = energy_per_query_kwh(args.coconut_avg_tokens, args.joules_per_token)
    savings_per_query = e_baseline - e_coconut

    print("=" * 60)
    print(f"Baseline training: {baseline_kwh:.6f} kWh | Coconut training: {coconut_kwh:.6f} kWh")
    print(f"Extra training cost: {extra_training_kwh:.6f} kWh")
    print(f"Baseline: {args.baseline_avg_tokens} tok -> {e_baseline:.9f} kWh/query")
    print(f"Coconut:  {args.coconut_avg_tokens} tok -> {e_coconut:.9f} kWh/query")
    print(f"Savings/query: {savings_per_query:.9f} kWh")

    # FIX: handle negative extra_training_kwh explicitly (Coconut cheaper on both axes)
    if extra_training_kwh <= 0 and savings_per_query > 0:
        print("\nCoconut is cheaper on BOTH training and inference -- no break-even needed.")
    elif savings_per_query <= 0:
        print("\nCoconut does NOT save inference energy at these token counts -- no break-even exists.")
    else:
        breakeven = extra_training_kwh / savings_per_query
        print(f"\nBREAK-EVEN: {breakeven:,.0f} queries")
    print("=" * 60)

    print("\nSensitivity table:")
    print(f"{'J/token':>10} | {'savings/query':>18} | {'break-even':>15}")
    for jpt, label in [(2.61, "small model"), (5.0, "mid"), (9.35, "small upper"), (3.5, "large deployed")]:
        eb = energy_per_query_kwh(args.baseline_avg_tokens, jpt)
        ec = energy_per_query_kwh(args.coconut_avg_tokens, jpt)
        sav = eb - ec
        if extra_training_kwh <= 0 and sav > 0:
            print(f"{jpt:>10.2f} | {sav:>18.9f} | {'already ahead':>15}   ({label})")
        elif sav > 0:
            be = extra_training_kwh / sav
            print(f"{jpt:>10.2f} | {sav:>18.9f} | {be:>15,.0f}   ({label})")
        else:
            print(f"{jpt:>10.2f} | {sav:>18.9f} | {'no break-even':>15}   ({label})")


if __name__ == "__main__":
    main()
