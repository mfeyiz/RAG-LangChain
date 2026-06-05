import json
import os
from datasets import load_dataset


def download_hotpotqa():
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "hotpotqa")
    os.makedirs(output_dir, exist_ok=True)

    print("Downloading HotpotQA dataset...")
    dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)

    count = 0
    max_samples = 100

    for item in dataset:
        if count >= max_samples:
            break

        context_text = ""
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            context_text += f"\n## {title}\n"
            context_text += " ".join(sentences) + "\n"

        doc = {
            "question": item["question"],
            "answer": item["answer"],
            "context": context_text.strip(),
            "type": item["type"],
            "level": item["level"],
        }

        filename = f"hotpotqa_{count:04d}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        count += 1

    print(f"Downloaded {count} samples to {output_dir}")

    combined_text = ""
    for i in range(count):
        filename = f"hotpotqa_{i:04d}.json"
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            doc = json.load(f)
            combined_text += f"# {doc['question']}\n\n"
            combined_text += f"{doc['context']}\n\n"
            combined_text += f"**Answer:** {doc['answer']}\n\n"
            combined_text += "---\n\n"

    combined_path = os.path.join(os.path.dirname(__file__), "..", "data", "hotpotqa_knowledge_base.txt")
    with open(combined_path, "w", encoding="utf-8") as f:
        f.write(combined_text)

    print(f"Combined knowledge base created: {combined_path}")


if __name__ == "__main__":
    download_hotpotqa()
