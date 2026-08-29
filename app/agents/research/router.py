"""为研究步骤选择成本受控且可解释的资料查找方式."""

from app.schemas.research import ResearchStep, RetrievalStrategy, RouteDecision

_FRESHNESS_TERMS = ("最新", "今天", "目前", "current", "latest", "today", "recent", "2026")
_GLOBAL_TERMS = ("趋势", "整体", "主题", "共同", "跨文档", "landscape", "trend", "overall")
_RELATION_TERMS = ("关系", "关联", "属于", "参与", "投资", "谁", "relationship", "connected", "between")


class ResearchRouter:
    """用确定性规则修正 Planner 建议，避免为明显问题再调用一次模型.

    Planner 的建议保留开放语义判断，规则则保证几个不能妥协的边界：涉及最新
    信息必须查网页；跨文档主题优先查社区；明确实体关系优先查局部图。Hybrid
    是用户文档的基础查找方式，在图信息不足时提供原文补充。
    """

    def route(self, step: ResearchStep) -> RouteDecision:
        """返回去重后的查找方式和便于日志/调试展示的理由."""
        text = f"{step.objective} {' '.join(step.search_queries)}".lower()
        strategies = list(step.preferred_strategies)
        reasons = ["保留 Planner 建议"]

        if any(term in text for term in _FRESHNESS_TERMS):
            strategies.append(RetrievalStrategy.WEB)
            reasons.append("问题包含时效性信号")
        if any(term in text for term in _GLOBAL_TERMS):
            strategies.append(RetrievalStrategy.GRAPH_GLOBAL)
            reasons.append("问题需要跨文档主题")
        if any(term in text for term in _RELATION_TERMS):
            strategies.append(RetrievalStrategy.GRAPH_LOCAL)
            reasons.append("问题包含实体关系信号")

        # dict 保留首次出现顺序，让日志和测试结果稳定。
        unique = tuple(dict.fromkeys(strategies))
        return RouteDecision(
            step_id=step.step_id,
            strategies=unique,
            reason="；".join(reasons),
        )


__all__ = ["ResearchRouter"]
