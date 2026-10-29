"""LIME local explanation: which words of the sentence make the ensemble call the target word a skill."""
import numpy as np
from lime.lime_text import LimeTextExplainer


def explain_skill(extractor, sentence, skill, num_samples=100):
    words, target = sentence["words"], skill["word_range"][0]

    def predict_fn(texts):
        p = np.asarray(extractor.skill_prob([t.split() for t in texts], target), dtype=float)
        return np.column_stack([1 - p, p])

    explainer = LimeTextExplainer(class_names=["not skill", "skill"], split_expression=r"\s+",
                                  bow=False, random_state=42)
    exp = explainer.explain_instance(" ".join(words), predict_fn, labels=(1,),
                                     num_features=len(words), num_samples=num_samples)
    weights = [0.0] * len(words)
    for idx, w in exp.as_map()[1]:
        weights[idx] = float(w)
    return [{"word": w, "weight": weights[i], "is_target": skill["word_range"][0] <= i <= skill["word_range"][1]}
            for i, w in enumerate(words)]
