"""兼容旧命令的 Research Worker 薄入口."""

from app.entrypoints.research_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
