from __future__ import annotations

from abc import ABC, abstractmethod

from directme.retrieval.retriever import RetrievedContext


class AnswerGenerator(ABC):
    @abstractmethod
    def answer(self, context: RetrievedContext) -> str:
        raise NotImplementedError


class RuleBasedAnswerGenerator(AnswerGenerator):
    """Deterministic fallback generator for demos and tests.

    Surfaces the new relation labels and reachability directly so users can
    verify the egocentric pipeline without spinning up a VLM.
    """

    def answer(self, context: RetrievedContext) -> str:
        lang = context.intent.language
        if not context.items:
            return (
                "没有在空间记忆中找到匹配目标。"
                if lang.startswith("zh")
                else "No matching object was found in spatial memory."
            )

        parts: list[str] = []
        for item in context.items:
            parts.append(f"{item.node.node_id} {item.egocentric['natural_language']}")

        # Reachability question takes precedence.
        if context.intent.wants_reachability:
            reachable_items = [i for i in context.items if i.egocentric.get("reachable")]
            if lang.startswith("zh"):
                if reachable_items:
                    return (
                        f"可及范围（{context.reachable_radius_m:.0f} 米内）有 "
                        f"{len(reachable_items)} 个：" + "；".join(parts) + "。"
                    )
                return f"在 {context.reachable_radius_m:.0f} 米可及范围内未找到目标；" + "；".join(parts) + "。"
            if reachable_items:
                return (
                    f"{len(reachable_items)} object(s) within {context.reachable_radius_m:.0f} m: "
                    + "; ".join(parts) + "."
                )
            return f"No objects within {context.reachable_radius_m:.0f} m. Other matches: " + "; ".join(parts) + "."

        if context.intent.wants_count or context.intent.wants_location:
            if lang.startswith("zh"):
                return f"找到 {context.count} 个匹配目标：" + "；".join(parts) + "。"
            return f"Found {context.count} matching object(s): " + "; ".join(parts) + "."

        if lang.startswith("zh"):
            return "；".join(parts) + "。"
        return "; ".join(parts) + "."
