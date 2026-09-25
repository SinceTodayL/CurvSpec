from curvspec.matching.objectives import loss


def build_objective(config):
    return loss(config)


__all__ = ["build_objective"]
