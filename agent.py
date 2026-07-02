from typing import List

from schemas import ChatResponse, Message, Recommendation
from retrieval import AssessmentRetriever
from prompts import SYSTEM_PROMPT, build_user_prompt
from llm_client import call_llm_json

DEFAULT_RETRIEVAL_SIZE = 25
MAX_RETURNED_RECOMMENDATIONS = 10
FORCE_RECOMMENDATION_AFTER = 7


class RecommendationEngine:
    def __init__(self, catalog: AssessmentRetriever):
        self.catalog = catalog

    def respond(self, messages: List[Message]) -> ChatResponse:
        if not messages:
            return self._welcome()

        query = self._build_query(messages)

        candidates = self.catalog.search(
            query,
            limit=DEFAULT_RETRIEVAL_SIZE,
        )

        llm_output = self._query_llm(messages, candidates)

        return self._format_response(
            llm_output,
            candidates,
        )

    def _welcome(self) -> ChatResponse:
        return ChatResponse(
            reply=(
                "Hello! Tell me about the position you're hiring for, "
                "including skills, experience, technologies, or competencies, "
                "and I'll recommend suitable SHL assessments."
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    def _build_query(self, messages: List[Message]) -> str:
        return " ".join(
            m.content.strip()
            for m in messages
            if m.role == "user"
        )

    def _conversation(self, messages: List[Message]) -> str:
        return "\n".join(
            f"{m.role.title()}: {m.content}"
            for m in messages
        )

    def _query_llm(
        self,
        messages: List[Message],
        candidates: list,
    ) -> dict:

        system_prompt = SYSTEM_PROMPT

        if len(messages) >= FORCE_RECOMMENDATION_AFTER:
            system_prompt += """

The conversation is nearing its turn limit.

Do not ask additional clarification questions.

Recommend the most appropriate assessments using available information.

Return action="recommend".
"""

        user_prompt = build_user_prompt(
            self._conversation(messages),
            candidates,
        )

        try:
            return call_llm_json(
                system_prompt,
                user_prompt,
            )

        except Exception as exc:
            return {
                "action": "error",
                "reply": f"Internal error: {exc}",
                "selected_ids": [],
                "end_of_conversation": False,
            }

    def _format_response(
        self,
        result: dict,
        candidates: list,
    ) -> ChatResponse:

        recommendations = []

        if result.get("action") in {"recommend", "compare"}:

            candidate_lookup = {
                c["id"]: c
                for c in candidates
            }

            for assessment_id in result.get("selected_ids", []):

                item = (
                    candidate_lookup.get(assessment_id)
                    or self.catalog.get(assessment_id)
                )

                if item is None:
                    continue

                recommendations.append(
                    Recommendation(
                        name=item["name"],
                        url=item["url"],
                        test_type=", ".join(
                            item.get("test_types", [])
                        ) or "Unspecified",
                    )
                )

        return ChatResponse(
            reply=result.get(
                "reply",
                "Could you provide a little more information?",
            ),
            recommendations=recommendations[
                :MAX_RETURNED_RECOMMENDATIONS
            ],
            end_of_conversation=bool(
                result.get(
                    "end_of_conversation",
                    False,
                )
            ),
        )


def handle_chat(
    messages: List[Message],
    index: AssessmentRetriever,
) -> ChatResponse:

    engine = RecommendationEngine(index)

    return engine.respond(messages)
    
