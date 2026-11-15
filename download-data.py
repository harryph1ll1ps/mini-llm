from datasets import load_dataset

# download 5000 stories from hugging face
print("==== LOADING DATA ====")
dataset = load_dataset("roneneldan/TinyStories", split="train[:5000]")

with open("corpus.txt", "w", encoding="utf-8") as f:
    for story in dataset:
        f.write(story["text"])
        f.write("\n\n")

print("==== DONE:", len(dataset), "STORIES WRITTEN TO CORPUS.TXT ====")


