from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _rapidfuzz_levenshtein
except Exception:  # pragma: no cover - optional acceleration
    _rapidfuzz_levenshtein = None


def parse_krn_content(
    krn: str,
    ler_parsing: bool = False,
    cer_parsing: bool = False,
) -> list[str]:
    if cer_parsing:
        krn = krn.replace("\n", " <b> ")
        krn = krn.replace("\t", " <t> ")
        tokens = krn.split(" ")
        characters: list[str] = []
        for token in tokens:
            if token not in ["<b>", "<t>"]:
                characters.append(token)
            else:
                characters.extend(list(token))
        return characters

    if ler_parsing:
        krn_lines = krn.split("\n")
        for index, line in enumerate(krn_lines):
            line = line.replace("\n", " <b> ")
            line = line.replace("\t", " <t> ")
            krn_lines[index] = line
        return krn_lines

    krn = krn.replace("\n", " <b> ")
    krn = krn.replace("\t", " <t> ")
    return krn.split(" ")


def levenshtein(a: list[str], b: list[str]) -> int:
    if _rapidfuzz_levenshtein is not None:
        return int(_rapidfuzz_levenshtein.distance(a, b))

    n, m = len(a), len(b)
    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))
    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current[j] = min(add, delete, change)
    return current[n]


def compute_metric(hypotheses: list[list[str]], ground_truths: list[list[str]]) -> float:
    acc_edit_distance = 0
    acc_length = 0
    for hypothesis, ground_truth in zip(hypotheses, ground_truths):
        acc_edit_distance += levenshtein(hypothesis, ground_truth)
        acc_length += len(ground_truth)
    if acc_length == 0:
        return 0.0
    return 100.0 * acc_edit_distance / acc_length


def get_metrics(hyp_array: list[str], gt_array: list[str]) -> tuple[float, float, float]:
    hyp_cer = []
    gt_cer = []
    hyp_ser = []
    gt_ser = []
    hyp_ler = []
    gt_ler = []

    for hypothesis, ground_truth in zip(hyp_array, gt_array):
        hyp_ler.append(parse_krn_content(hypothesis, ler_parsing=True, cer_parsing=False))
        gt_ler.append(parse_krn_content(ground_truth, ler_parsing=True, cer_parsing=False))

        hyp_ser.append(parse_krn_content(hypothesis, ler_parsing=False, cer_parsing=False))
        gt_ser.append(parse_krn_content(ground_truth, ler_parsing=False, cer_parsing=False))

        hyp_cer.append(parse_krn_content(hypothesis, ler_parsing=False, cer_parsing=True))
        gt_cer.append(parse_krn_content(ground_truth, ler_parsing=False, cer_parsing=True))

    cer = compute_metric(hyp_cer, gt_cer)
    ser = compute_metric(hyp_ser, gt_ser)
    ler = compute_metric(hyp_ler, gt_ler)
    return cer, ser, ler
