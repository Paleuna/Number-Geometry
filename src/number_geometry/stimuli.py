"""Stimulus construction used for the numerical-task activation dataset."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

NUM_WORDS_MAP = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def get_num_repr(n: int, fmt: str) -> str:
    return NUM_WORDS_MAP[n] if fmt == "english" else str(n)


def pluralize(noun: str, n: int) -> str:
    return noun if n == 1 else noun + "s"


def add_entry(dataset, task, subset, val, fmt, text, custom_idx=None):
    target = get_num_repr(val, fmt)
    tokens = text.split()
    if custom_idx is not None:
        idx = len(tokens) + custom_idx if custom_idx < 0 else custom_idx
    else:
        idx = next((i for i, tok in enumerate(tokens) if tok.strip(".,:!?\"'") == target), -1)
    if idx < 0 or idx >= len(tokens):
        raise ValueError(f"Target {target!r} not found at a valid position in: {text!r}")
    dataset.append({"task": task, "subset": subset, "val": val, "format": fmt, "target_token": target, "text": text, "target_idx": idx})


def generate_unified_dataset():
    """Generate the handcrafted tasks: 9 numbers × 2 formats × 11 task conditions × 5 sentences."""
    dataset = []
    primes = {2, 3, 5, 7}
    for fmt in ["digit", "english"]:
        for val in range(1, 10):
            target = get_num_repr(val, fmt)
            zero = get_num_repr(0, fmt)
            one = get_num_repr(1, fmt)
            noun = pluralize("apple", val)

            for text in [
                f"I have a total of {target} {noun}",
                f"The number of items is {target}",
                f"There are in total {target} {noun}",
                f"The total count is {target}",
                f"Observed item count: {target}",
            ]:
                add_entry(dataset, "quantity", "control", val, fmt, text)

            parity = "odd" if val % 2 else "even"
            for text in [
                f"The set of {parity} numbers includes {target}",
                f"Examples of {parity} integers are {target} and others",
                f"Known {parity} digits are {target} etc",
                f"Talking about {parity} values like {target}",
                f"Select the {parity} number : {target}",
            ]:
                add_entry(dataset, "parity", parity, val, fmt, text)

            lower = get_num_repr(val - 1, fmt)
            for text in [
                f"Numbers greater than {lower} include {target}",
                f"Values larger than {lower} are {target} and others",
                f"Any integer exceeding {lower} is {target}",
                f"A number bigger than {lower} is {target}",
                f"Count integers above {lower} : {target}",
            ]:
                add_entry(dataset, "comparison", "greater", val, fmt, text)

            upper = get_num_repr(val + 1, fmt)
            for text in [
                f"Numbers smaller than {upper} include {target}",
                f"Values less than {upper} are {target} and others",
                f"Any integer below {upper} is {target}",
                f"A number lower than {upper} is {target}",
                f"Count integers under {upper} : {target}",
            ]:
                add_entry(dataset, "comparison", "smaller", val, fmt, text)

            add_templates = [
                f"The sum of {target} and {zero} is {target}",
                f"The result of adding {target} to {zero} is {target}",
                f"{zero} added to {target} gives {target}",
                f"Answer to {zero} plus {target} is {target}",
                f"Equation : {zero} plus {target} equals {target}",
            ]
            for text in add_templates:
                add_entry(dataset, "addition", "pre_equal", val, fmt, text)
                add_entry(dataset, "addition", "post_equal", val, fmt, text, custom_idx=-1)

            mul_templates = [
                f"The product of {target} and {one} is {target}",
                f"The result of multiplying {target} by {one} is {target}",
                f"{one} multiplied to {target} gives {target}",
                f"Answer to {one} times {target} is {target}",
                f"Equation : {one} times {target} equals {target}",
            ]
            for text in mul_templates:
                add_entry(dataset, "multiplication", "pre_equal", val, fmt, text)
                add_entry(dataset, "multiplication", "post_equal", val, fmt, text, custom_idx=-1)

            prime_label = "prime" if val in primes else "composite"
            for text in [
                f"List of {prime_label} numbers contains {target}",
                f"Found a {prime_label} integer : {target}",
                f"Example of {prime_label} value is {target}",
                f"The set of {prime_label}s contains {target}",
                f"Identify the {prime_label} digit : {target}",
            ]:
                add_entry(dataset, "prime", prime_label, val, fmt, text)

            prev = get_num_repr(val - 1, fmt)
            for text in [
                f"The number after {prev} is {target}",
                f"Successor of {prev} is {target}",
                f"Counting up from {prev} gives {target}",
                f"Next integer after {prev} is {target}",
                f"The value following {prev} is {target}",
            ]:
                add_entry(dataset, "successor", "next", val, fmt, text, custom_idx=-1)

            nxt = get_num_repr(val + 1, fmt)
            for text in [
                f"The number before {nxt} is {target}",
                f"Predecessor of {nxt} is {target}",
                f"Counting down from {nxt} gives {target}",
                f"Previous integer before {nxt} is {target}",
                f"The value preceding {nxt} is {target}",
            ]:
                add_entry(dataset, "predecessor", "prev", val, fmt, text, custom_idx=-1)
    return dataset


def prepare_corpus_data(df: pd.DataFrame, task_name: str, fmt: str):
    dataset = []
    for col in df.columns:
        val = int(col)
        texts, idxs = df[col].iloc[0], df[col].iloc[1]
        expected = get_num_repr(val, fmt)
        for text, idx in zip(texts, idxs):
            dataset.append({"task": task_name, "subset": "real_data", "val": val, "format": fmt, "text": text, "target_idx": int(idx), "target_token": expected})
    return dataset


def load_all_stimuli(corpus_dir: str | Path):
    corpus_dir = Path(corpus_dir)
    digit_insert = pd.read_json(corpus_dir / "real_insert_digit.json")
    english_insert = pd.read_json(corpus_dir / "real_insert_english.json")
    digit_sample = pd.read_json(corpus_dir / "real_sample_digit.json")
    english_sample = pd.read_json(corpus_dir / "real_sample_english.json")
    real_insert = prepare_corpus_data(digit_insert, "real_insert", "digit") + prepare_corpus_data(english_insert, "real_insert", "english")
    real_sample = prepare_corpus_data(digit_sample, "real_sample", "digit") + prepare_corpus_data(english_sample, "real_sample", "english")
    return real_insert + real_sample + generate_unified_dataset()
