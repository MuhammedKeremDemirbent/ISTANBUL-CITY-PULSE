"""Model eğitimi. MLflow'u bilmez; yalnızca `Regressor` arayüzüne bağlıdır."""

from city_pulse.ml.protocols import Regressor, split_inputs_target
from city_pulse.ml.split import Split


def train_model(model: Regressor, split: Split) -> Regressor:
    """Modeli `split.train` üzerinde eğitir. Destekliyorsa `valid` ile erken durdurma kullanır."""
    features, target = split_inputs_target(split.train)
    if hasattr(model, "fit_eval"):
        valid_features, valid_target = split_inputs_target(split.valid)
        model.fit_eval(features, target, valid_features, valid_target)
    else:
        model.fit(features, target)
    return model
