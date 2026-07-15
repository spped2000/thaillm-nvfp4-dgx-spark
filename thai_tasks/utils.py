"""Helpers for scb10x/thai_exam lm-eval tasks.

Subsets have variable choice columns (ic: a-d, others a-e) and some rows
leave a choice empty; `answer` always names a non-empty choice.
"""

KEYS = ["a", "b", "c", "d", "e"]


def _valid_keys(doc):
    return [
        k
        for k in KEYS
        if doc.get(k) is not None and str(doc[k]).strip() not in ("", "None", "nan")
    ]


def doc_to_text(doc):
    return f"คำถาม: {doc['question'].strip()}\nคำตอบ:"


def doc_to_choice(doc):
    return [str(doc[k]).strip() for k in _valid_keys(doc)]


def doc_to_target(doc):
    return _valid_keys(doc).index(doc["answer"].strip().lower())


def process_docs_wiki(dataset):
    return dataset.filter(lambda d: len(d["text"]) >= 500)


def doc_to_text_letter(doc):
    lines = [f"คำถาม: {doc['question'].strip()}"]
    for k in _valid_keys(doc):
        lines.append(f"{k}. {str(doc[k]).strip()}")
    lines.append("คำตอบ:")
    return "\n".join(lines)


def doc_to_choice_letter(doc):
    return [k for k in _valid_keys(doc)]
