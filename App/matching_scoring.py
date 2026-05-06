from typing import Iterable, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def resume_jd_match_percentage(resume_text: str, job_description: str) -> float:
    """
    Returns match percentage (0-100) using TF-IDF + cosine similarity.
    """
    if not resume_text or not job_description:
        return 0.0

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_text, job_description])
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(float(similarity) * 100.0, 2)


def improved_resume_score(
    resume_text: str,
    extracted_skills: Iterable[str],
    target_keywords: List[str],
) -> Tuple[int, dict]:
    """
    Lightweight score (0-100) using:
    - skills presence (40)
    - resume length (30)
    - keyword relevance (30)
    """
    text = (resume_text or "").lower()
    skills = [s.lower() for s in (extracted_skills or [])]

    # 1) Skills presence (0-40)
    skill_hits = sum(1 for k in target_keywords if k.lower() in skills or k.lower() in text)
    skill_score = min(40, int((skill_hits / max(1, len(target_keywords))) * 40))

    # 2) Resume length (0-30), basic healthy range
    word_count = len(text.split())
    if word_count < 150:
        length_score = 10
    elif word_count <= 900:
        length_score = 30
    elif word_count <= 1300:
        length_score = 20
    else:
        length_score = 12

    # 3) Keyword relevance (0-30)
    keyword_hits = sum(1 for k in target_keywords if k.lower() in text)
    relevance_score = min(30, int((keyword_hits / max(1, len(target_keywords))) * 30))

    total_score = max(0, min(100, skill_score + length_score + relevance_score))
    breakdown = {
        "skills_score": skill_score,
        "length_score": length_score,
        "relevance_score": relevance_score,
        "word_count": word_count,
        "matched_keywords": keyword_hits,
    }
    return total_score, breakdown
