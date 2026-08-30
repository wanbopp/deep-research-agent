"""Prometheus 抓取入口."""

from fastapi import APIRouter, Response

from app.observability import metrics

router = APIRouter(include_in_schema=False)


@router.get("/metrics")
async def prometheus_metrics() -> Response:
    """返回当前 API 进程指标；正文不包含用户输入或业务对象标识."""
    snapshot = metrics.render()
    return Response(content=snapshot.content, headers={"Content-Type": snapshot.content_type})


__all__ = ["router"]
