from __future__ import annotations

from pathlib import Path

from utils.constants import (
    BREAK_TOKEN,
    KERN_LINE_BREAK,
    KERN_MIDDLE_DOT,
    KERN_SPACE,
    KERN_TAB,
    PAD_TOKEN,
    SPACE_TOKEN,
    TAB_TOKEN,
)


def bekern_text_to_tokens(content: str) -> list[str]:
    content = content.replace(KERN_SPACE, f" {SPACE_TOKEN} ")
    content = content.replace(KERN_MIDDLE_DOT, KERN_SPACE)

    token_lines: list[list[str]] = []
    for line in content.split(KERN_LINE_BREAK):
        tokens = line.replace(KERN_TAB, f" {TAB_TOKEN} ").split(KERN_SPACE)
        if len(tokens) > 1:
            tokens.append(BREAK_TOKEN)
            token_lines.append(tokens)

    return [token for line in token_lines for token in line]


def tokens_to_kern(tokens: list[str]) -> str:
    transcription = "".join(token for token in tokens if token != PAD_TOKEN)
    transcription = transcription.replace(TAB_TOKEN, KERN_TAB)
    transcription = transcription.replace(BREAK_TOKEN, KERN_LINE_BREAK)
    transcription = transcription.replace(SPACE_TOKEN, KERN_SPACE)
    return transcription


def parse_krn_content(
    krn: str,
    ler_parsing: bool = False,
    cer_parsing: bool = False,
) -> list[str]:
    if cer_parsing:
        krn = krn.replace(KERN_LINE_BREAK, f" {BREAK_TOKEN} ")
        krn = krn.replace(KERN_TAB, f" {TAB_TOKEN} ")
        tokens = krn.split(KERN_SPACE)
        characters = []
        for token in tokens:
            if token not in [BREAK_TOKEN, TAB_TOKEN]:
                characters.append(token)
            else:
                characters.extend(list(token))
        return characters

    if ler_parsing:
        krn_lines = krn.split(KERN_LINE_BREAK)
        for index, line in enumerate(krn_lines):
            line = line.replace(KERN_LINE_BREAK, f" {BREAK_TOKEN} ")
            line = line.replace(KERN_TAB, f" {TAB_TOKEN} ")
            krn_lines[index] = line
        return krn_lines

    krn = krn.replace(KERN_LINE_BREAK, f" {BREAK_TOKEN} ")
    krn = krn.replace(KERN_TAB, f" {TAB_TOKEN} ")
    return krn.split(KERN_SPACE)


def save_kern_outputs(output_path: str | Path, transcriptions: list[list[str]]) -> None:
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    for index, content in enumerate(transcriptions):
        (output_path / f"{index}.bekern").write_text(tokens_to_kern(content), encoding="utf-8")
