"""Optional decision-core bridge. Application policy and execution stay in arc-cua.

Import this module explicitly after installing the ``decision-core`` extra. Each
bound policy is local to one operation, including its context and packing state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from decision_core import (
    CallContext,
    Choice,
    ChoiceAnswer,
    DecisionError,
    DecisionProvider,
    DecisionRequest,
    InvocationError,
    prepare_systemone,
)

from .choice import ChoicePolicy


class DecisionProviderFailure(RuntimeError):
    """Safe consumer outcome, with no raw provider text or automatic recovery."""

    def __init__(self, code: str, *, receipt: Any = None) -> None:
        self.code = code
        self.receipt = receipt
        super().__init__(f"{code}: decision provider failed; no action executed")


def _failure(error: DecisionError) -> DecisionProviderFailure:
    # Do not let a third-party adapter echo arbitrary error text or codes.
    allowed = {
        "invalid_request", "invalid_profile", "unsupported", "context_unrepresentable",
        "invalid_response", "model_mismatch", "invalid_context", "profile_not_allowed",
        "deadline_exceeded", "cancelled", "allowance_exhausted", "allowance_error",
        "provider_unavailable", "model_unavailable", "deployment_mismatch",
        "inspection_failed", "invalid_configuration", "transport_error", "http_error",
        "response_too_large", "egress_denied", "invalid_transport", "invalid_cost_policy",
        "unknown_cost", "permission_denied", "allowance_denied", "reservation_failed",
        "invocation_state", "invocation_failed",
    }
    code = error.code if error.code in allowed else "provider_failure"
    return DecisionProviderFailure(code, receipt=error.receipt if isinstance(error, InvocationError) else None)


@dataclass(frozen=True)
class DecisionCoreTransport:
    """Bind a trusted provider to one host-owned operation context.

    No endpoint, credentials, deadline or allowance comes from model state. The
    provider performs its own final validation and one accounted network attempt.
    """

    provider: DecisionProvider
    context: CallContext
    name = "decision-core"
    supports_images = False
    full_distribution = True

    def ask(
        self,
        state: Mapping[str, Any],
        questions: Mapping[str, Any],
        *,
        images: Sequence[bytes] = (),
    ) -> Mapping[str, Any]:
        try:
            if images:
                raise DecisionError("unsupported", "Images are not supported.")
            converted = {}
            for ident, question in questions.items():
                if (not isinstance(question, Mapping) or
                        set(question) != {"type", "criteria", "instructions"} or
                        question["type"] != "choice"):
                    raise DecisionError("unsupported", "Unsupported consumer question shape.")
                converted[ident] = Choice(question["instructions"], question["criteria"])
            request = DecisionRequest("arc-cua.choice", "1", dict(state), converted)
            # Cover every speculative family before evaluating. No semantic
            # truncation of operations, shortcuts, inputs or targets is allowed.
            prepared = prepare_systemone(request, self.provider.profile)
            batch = self.provider.evaluate(request, context=self.context)
            if (batch.request_sha256 != prepared.request_sha256 or
                    batch.contract_sha256 != prepared.contract_sha256 or
                    batch.profile_id != prepared.profile.id or
                    batch.response_model != prepared.profile.expected_response_model or
                    set(batch.answers) != set(converted) or
                    any(not isinstance(answer, ChoiceAnswer) for answer in batch.answers.values())):
                raise DecisionError("invalid_response", "Provider batch does not match the request.")
            return {
                "answers": {ident: {
                    "choice": answer.choice,
                    "confidence": answer.provider_confidence,
                    "probabilities": dict(answer.probabilities),
                } for ident, answer in batch.answers.items()},
                "model": batch.response_model,
                "usage": {"input_tokens": batch.usage.input_tokens, "output_tokens": batch.usage.output_tokens},
                "decision_core": {
                    "profile_id": batch.profile_id,
                    "request_sha256": batch.request_sha256,
                    "contract_sha256": batch.contract_sha256,
                },
            }
        except DecisionError as error:
            raise _failure(error) from None


@dataclass(frozen=True)
class DecisionCorePolicy:
    """Reusable provider configuration; ``bind`` creates an operation-local policy.

    Keep max_candidates as the existing application's candidate policy. Profile
    caps never silently drop choices. All built families must fit or fail before
    provider IO, even if the model would not consume that family for its action.
    """

    provider: DecisionProvider
    max_candidates: int = 240

    def bind(self, context: CallContext) -> ChoicePolicy:
        profile = self.provider.profile
        # With no separate state cap, total bytes is also a safe state+head cap.
        budget = (profile.state_plus_longest_question_bytes or profile.max_request_bytes,
                  profile.max_request_bytes)
        return ChoicePolicy(
            DecisionCoreTransport(self.provider, context),
            max_candidates=self.max_candidates,
            invalid_retries=0,
            provider_budget=budget,
            request_model=profile.model,
        )
