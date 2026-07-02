from typing import List, Dict, Any
from pathlib import Path
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class AssessmentRetriever:
    def __init__(self, catalog_path: str):
        self.catalog_path = Path(catalog_path)
        self.assessments = self._load_catalog()

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            lowercase=True,
            ngram_range=(1, 2),
        )

        self.document_matrix = self.vectorizer.fit_transform(
            self._documents()
        )

    def _load_catalog(self) -> List[Dict[str, Any]]:
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cleaned = []

        for item in raw:

            if "solution" in item.get("name", "").lower():
                continue

            cleaned.append(self._convert(item))

        return cleaned

    def _convert(self, item: Dict[str, Any]) -> Dict[str, Any]:

        duration = item.get("duration", "")

        minutes = "".join(ch for ch in duration if ch.isdigit())

        return {
            "id": f"assessment_{item['entity_id']}",
            "name": item["name"],
            "url": item["link"],
            "description": item.get("description", ""),
            "categories": item.get("keys", []),
            "job_levels": item.get("job_levels_raw", ""),
            "languages": item.get("languages_raw", ""),
            "duration": int(minutes) if minutes else None,

            # Added so agent.py works correctly
            "test_types": item.get("test_types", []),
        }

    def _documents(self):

        docs = []

        for item in self.assessments:

            docs.append(
                " ".join(
                    [
                        item["name"],
                        item["description"],
                        " ".join(item["categories"]),
                        item["job_levels"],
                        item["languages"],
                    ]
                )
            )

        return docs

    def search(
        self,
        query: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:

        if not query.strip():
            return self.assessments[:limit]

        vector = self.vectorizer.transform([query])

        scores = cosine_similarity(
            vector,
            self.document_matrix,
        )[0]

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []

        for index, score in ranked[:limit]:

            item = dict(self.assessments[index])

            item["score"] = float(score)

            results.append(item)

        return results

    def get(self, assessment_id: str):

        for item in self.assessments:

            if item["id"] == assessment_id:
                return item

        return None
        