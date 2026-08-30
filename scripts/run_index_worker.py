"""兼容旧命令的一次性 Index Worker 薄入口."""

from app.entrypoints.index_worker import main_until_idle


if __name__ == "__main__":
    raise SystemExit(main_until_idle())
