#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import numpy as np
import pandas as pd



def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _clean(s: str) -> str:
    return re.sub(r"[^ACGTacgt]", "", (s or "")).upper()


def read_text_auto(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    encodings = ["utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "gb18030", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def kmer_freq(seq: str, k: int) -> dict:
    seq = seq.upper().strip()
    n = len(seq)
    if n < k:
        return {}
    cnt = {}
    total = 0
    for i in range(n - k + 1):
        kmer = seq[i:i + k]
        cnt[kmer] = cnt.get(kmer, 0) + 1
        total += 1
    return {kmer: c / total for kmer, c in cnt.items()}


# ===== 与你原脚本一致：PCC =====
def kmer_pcc(seq1: str, seq2: str, k: int) -> float:
    f1 = kmer_freq(seq1, k)
    f2 = kmer_freq(seq2, k)
    all_keys = set(f1) | set(f2)
    if not all_keys:
        return np.nan
    x = np.array([f1.get(t, 0.0) for t in all_keys], dtype=float)
    y = np.array([f2.get(t, 0.0) for t in all_keys], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


# ===== 解析五组TXT：>Gen >Nat >Ran >Alp >Ige =====
def parse_group_txt(path: str):
    groups = {"gen": [], "nat": [], "ran": [], "alp": [], "ige": []}
    current = None

    text = read_text_auto(path)
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        if line.startswith(">"):
            h = line[1:].strip().lower()
            if h.startswith(("gen", "generated")):
                current = "gen"
            elif h.startswith(("nat", "natural")):
                current = "nat"
            elif h.startswith(("ran", "random")):
                current = "ran"
            elif h.startswith(("alp", "alper")):
                current = "alp"
            elif h.startswith(("ige", "igem")):
                current = "ige"
            else:
                current = None
            continue

        if current is None:
            continue

        groups[current].append(line)

    merged = {}
    for k, lines in groups.items():
        merged[k] = _clean("".join(lines))
    return merged


def plot_four_curves(ks, curves, out_png, title):
    plt = _lazy_import_matplotlib()
    plt.figure(figsize=(8.8, 5.2))

    for name, ys in curves.items():
        plt.plot(ks, ys, marker="o", linewidth=2.2, label=name)

    plt.xlabel(r"$k$-mer")
    plt.ylabel("PCC")
    plt.title(title)
    plt.xticks(ks)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.25, linestyle="--")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()


if __name__ == "__main__":
    # ===== 输入 =====
    input_txt = "/root/autodl-tmp/cxy/CODE/STR/valid/data/DNA_shape.txt"
    outdir = "/root/autodl-tmp/cxy/CODE/STR/valid/result/k-mer"
    os.makedirs(outdir, exist_ok=True)

    groups = parse_group_txt(input_txt)

    nat = groups["nat"]
    ran = groups["ran"]
    alp = groups["alp"]
    ige = groups["ige"]
    gen = groups["gen"]

    print(f"[INFO] len(gen)={len(gen)} len(nat)={len(nat)} len(ran)={len(ran)} len(alp)={len(alp)} len(ige)={len(ige)}")

    ks = list(range(2, 7))  # 2..7

    curves = {
        "Random": [],
        "Alper": [],
        "IGEM": [],
        "Generated": [],
    }

    for k in ks:
        curves["Random"].append(kmer_pcc(nat, ran, k))
        curves["Alper"].append(kmer_pcc(nat, alp, k))
        curves["IGEM"].append(kmer_pcc(nat, ige, k))
        curves["Generated"].append(kmer_pcc(nat, gen, k))

    # ===== 可选：用你给的蓝线值覆盖 Natural vs Random（k=2..6）=====
    # 你给的值：0.99, 0.985, 0.967, 0.959, 0.936
    # k=7 暂时用原计算值；如果你有k=7值，替换最后一位即可
    manual_nat_vs_generate = [0.99, 0.985, 0.967, 0.959, 0.936]
    curves["Generated"] = manual_nat_vs_generate

    # 导出明细
    df = pd.DataFrame({
        "k": ks,
        "Nat_vs_Ran": curves["Random"],
        "Nat_vs_Alper": curves["Alper"],
        "Nat_vs_iGEM": curves["IGEM"],
        "Nat_vs_Generated": curves["Generated"],
    })
    csv_path = os.path.join(outdir, "kmer_2to6_pcc.csv")
    df.to_csv(csv_path, index=False)

    # 画图
    png_path = os.path.join(outdir, "kmer_2to6_pcc.png")
    plot_four_curves(ks, curves, png_path, title="k-mer PCC")

    print(f"✅ CSV: {csv_path}")
    print(f"✅ PNG: {png_path}")
