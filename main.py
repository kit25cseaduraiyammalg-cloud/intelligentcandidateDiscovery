import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
print("Loading AI Model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Job Description
with open("job_description.txt", "r", encoding="utf-8") as f:
    job_description = f.read()

jd_embedding = model.encode(job_description)

# Target skills
target_skills = [
    "python",
    "machine learning",
    "deep learning",
    "nlp",
    "natural language processing",
    "llm",
    "generative ai",
    "rag",
    "retrieval",
    "embeddings",
    "vector database",
    "pytorch",
    "tensorflow",
    "transformers",
    "langchain"
]

results = []

print("Reading Candidates...")

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:

    for line in f:

        try:

            candidate = json.loads(line)

            candidate_text = json.dumps(candidate)

            # Semantic Similarity
            candidate_embedding = model.encode(candidate_text)

            similarity = cosine_similarity(
                [jd_embedding],
                [candidate_embedding]
            )[0][0]

            # Skill Matching
            text_lower = candidate_text.lower()

            matched_skills = []

            skill_points = 0

            for skill in target_skills:

                if skill in text_lower:
                    matched_skills.append(skill)
                    skill_points += 1

            skill_score = (
                skill_points / len(target_skills)
            ) * 100

            # Experience Detection
            exp_score = 50

            experience = 0

            if "experience" in candidate:
                try:
                    experience = float(
                        candidate["experience"]
                    )
                except:
                    experience = 0

            if 5 <= experience <= 9:
                exp_score = 100
            elif experience > 9:
                exp_score = 80
            else:
                exp_score = 50

            # Final Score
            final_score = (
                similarity * 50
                + skill_score * 0.30
                + exp_score * 0.20
            )

            candidate_id = (
                candidate.get("candidate_id")
                or candidate.get("id")
                or candidate.get("name")
                or "unknown"
            )

            reason = ", ".join(
                matched_skills[:5]
            )

            results.append({
                "Candidate_ID": candidate_id,
                "Semantic_Similarity":
                    round(similarity, 4),
                "Skill_Score":
                    round(skill_score, 2),
                "Experience_Score":
                    exp_score,
                "Final_Score":
                    round(final_score, 2),
                "Reason":
                    reason
            })

        except Exception:
            continue

# Ranking
results = sorted(
    results,
    key=lambda x: x["Final_Score"],
    reverse=True
)

top100 = results[:100]

# Create DataFrame
df = pd.DataFrame(top100)

df.insert(
    0,
    "Rank",
    range(1, len(df) + 1)
)

# Export Excel
df.to_excel(
    "top100_candidates.xlsx",
    index=False
)

print("===================================")
print("Top 100 Candidates Generated")
print("File: top100_candidates.xlsx")
print("===================================")
