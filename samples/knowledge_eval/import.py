from time import sleep
from typing import Any, Dict, List
from datasets import load_dataset, DatasetDict
from veadk.knowledgebase import KnowledgeBase

import json
import re
import os

def _normalize_documents(value: Any) -> List[str]:
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("passage")
                    or item.get("value")
                )
                if text is None and len(item) == 1:
                    only_val = next(iter(item.values()))
                    if isinstance(only_val, str):
                        text = only_val
                if text is not None:
                    out.append(text)
            elif item is not None:
                out.append(str(item))
        return out
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    return [str(value)]


def extract_fields(example: Dict[str, Any]) -> Dict[str, Any]:
    name = example.get("name") or example.get("id")

    question = example.get("question")

    docs_source = example.get("documents")
    documents = _normalize_documents(docs_source)

    response = example.get("response")
    return {
        "name": name,
        "question": question,
        "documents": " ".join(documents),
        "response": response,
    }

def import_txt(subset: str, kb: KnowledgeBase, extracted: List[Dict[str, Any]]):
    MAX_SIZE_BYTES = 5 * 1024 * 1024  # 5MB viking limit
    content_chunks = []
    current_chunk = ""

    for item in extracted:
        doc_content = item["documents"] + "\n"

        # Check if adding this document would exceed the size limit
        if (
            len(current_chunk.encode("utf-8")) + len(doc_content.encode("utf-8"))
            > MAX_SIZE_BYTES
        ):
            # Add current chunk if it's not empty
            if current_chunk:
                content_chunks.append(current_chunk)
                current_chunk = ""

        current_chunk += doc_content

    # Add the last chunk if it's not empty
    if current_chunk:
        content_chunks.append(current_chunk)

    # Add each chunk with indexed filename
    for index, chunk in enumerate(content_chunks):
        file_name = f"{subset}_{index}"
        kb.add_from_text(chunk, file_name=file_name)
        sleep(10)

def import_jsonl(subset: str, kb: KnowledgeBase, extracted: List[Dict[str, Any]]):
    # 将 extracted 中的数据按 JSONL 格式写入 KnowledgeBase, 对 covidqa 有效
    # 去重：基于 (title, content) 二元组
    seen = set()
    unique_pairs = []
    for item in extracted:
        docs = item["documents"] if isinstance(item["documents"], list) else [item["documents"]]
        for doc in docs:
            for m in re.finditer(r"Title:\s*(.*?)\s*Passage:\s*(.*?)(?=\s*Title:|$)", doc, re.S):
                t, c = m.group(1).strip(), m.group(2).strip()
                key = (t, c)
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append({"title": t, "content": c})
    jsonl_content = "\n".join(json.dumps(pair, ensure_ascii=False) for pair in unique_pairs)
    kb.add_from_text(jsonl_content, file_name=f"{subset}-jsonl")


def _process_subset_sync(
    subset: str, backend: str, split: str = None, app_name: str = "evaluation", import_type: str = "txt"
) -> None:
    kb = KnowledgeBase(
        backend=backend,
        app_name=app_name,
    )
    res = kb.collection_status()
    if not res["existed"]:
        raise ValueError("Collection not existed")

    ds_or_dict = load_dataset("rungalileo/ragbench", subset, split=split)
    extracted: List[Dict[str, Any]] = []

    if isinstance(ds_or_dict, DatasetDict):
        # 多 split
        for _, ds in ds_or_dict.items():
            for ex in ds:
                extracted.append(extract_fields(ex))
    else:
        # 单 split（已经是 Dataset）
        for ex in ds_or_dict:
            extracted.append(extract_fields(ex))

    if import_type == "txt":
        import_txt(subset, kb, extracted)
    elif import_type == "jsonl":
        import_jsonl(subset, kb, extracted)
    else:
        raise ValueError(f"Unknown import_type: {import_type}")


def main(backend: str, index: str) -> None:
    subsets = [
        "covidqa",
        "cuad",
        "delucionqa",
        "emanual",
        "expertqa",
        "finqa",
        "hagrid",
        "hotpotqa",
        "msmarco",
        "pubmedqa",
        "tatqa",
        "techqa",
    ]

    for subset in subsets:
        _process_subset_sync(subset, backend, app_name=index, import_type="txt")

if __name__ == "__main__":
    backend = os.getenv("DATABASE_TYPE")
    index = os.getenv("DATABASE_COLLECTION")
    if not backend or not index:
        raise ValueError("DATABASE_TYPE and DATABASE_COLLECTION must be set")
    main(backend=backend, index=index)
