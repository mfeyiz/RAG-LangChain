from dataclasses import dataclass

from RAG.services.tracing import read_trace


@dataclass
class EvaluationSample:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


def run_ragas_evaluation(samples: list[EvaluationSample]):
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except Exception as exc:
        raise RuntimeError("Install ragas to run RAGAS evaluation.") from exc

    dataset = Dataset.from_dict(
        {
            "question": [sample.question for sample in samples],
            "answer": [sample.answer for sample in samples],
            "contexts": [sample.contexts for sample in samples],
            "ground_truth": [sample.ground_truth for sample in samples],
        }
    )
    return evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
        ],
    )


def run_deepeval_evaluation(samples: list[EvaluationSample]):
    try:
        from deepeval import evaluate
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except Exception as exc:
        raise RuntimeError("Install deepeval to run DeepEval evaluation.") from exc

    test_cases = [
        LLMTestCase(
            input=sample.question,
            actual_output=sample.answer,
            expected_output=sample.ground_truth,
            retrieval_context=sample.contexts,
        )
        for sample in samples
    ]
    return evaluate(
        test_cases,
        metrics=[
            FaithfulnessMetric(),
            AnswerRelevancyMetric(),
        ],
    )


def sample_from_trace(trace_id: str, ground_truth: str = "") -> EvaluationSample:
    events = read_trace(trace_id)
    request = _payload_for(events, "request.received")
    retrieval = _payload_for(events, "researcher.retrieval")
    writer = _payload_for(events, "writer.response")

    contexts = [
        item.get("content", "")
        for item in retrieval.get("results", [])
        if item.get("content")
    ]

    return EvaluationSample(
        question=request.get("query", ""),
        answer=writer.get("answer", ""),
        contexts=contexts,
        ground_truth=ground_truth,
    )


def _payload_for(events: list[dict], name: str) -> dict:
    for event in reversed(events):
        if event.get("event") == name:
            return event.get("payload", {})
    return {}
