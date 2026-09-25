from curvspec.protocol.retrieval import RetrievalEvaluator


def build_evaluator(config):
    return RetrievalEvaluator(config)


__all__ = ["RetrievalEvaluator", "build_evaluator"]
