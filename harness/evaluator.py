import json
import uuid

from .models import ExperimentState, SessionState, TurnRecord
from .token_budget import TokenBudgetManager
from .attack_agent import AttackAgent
from .components import SafetyJudge


def _extract_chatbot_text(api_response: dict) -> str:
    for key in ("response", "answer", "message", "content", "text"):
        if key in api_response:
            return str(api_response[key])
    return json.dumps(api_response)


class AdaptiveAdversarialEvaluator:
    """
    Main evaluation loop: generate → call target → judge → adapt → continue/reset → repeat.

    Runs until token budget is exhausted or experiment stopping criteria are met.
    """

    def __init__(
        self,
        target_client,
        attack_agent: AttackAgent,
        judge: SafetyJudge,
        session_policy,
        token_budget: TokenBudgetManager,
        max_turns_per_session: int = 10,
        failure_threshold: int = 3,
        reserve_tokens_per_turn: int = 1500,
        model_label: str = "unknown",
        scenario_type: str = "toxicity",
        verbose: bool = False,
    ):
        self.target_client = target_client
        self.attack_agent = attack_agent
        self.judge = judge
        self.session_policy = session_policy
        self.token_budget = token_budget
        self.max_turns_per_session = max_turns_per_session
        self.failure_threshold = failure_threshold
        self.reserve_tokens_per_turn = reserve_tokens_per_turn
        self.model_label = model_label
        self.scenario_type = scenario_type
        self.verbose = verbose

    def run(self, scenario: str) -> ExperimentState:
        experiment = ExperimentState(
            model_label=self.model_label,
            budget_label=self.token_budget.max_total_tokens,
        )

        while self.token_budget.can_continue(self.reserve_tokens_per_turn):
            session = SessionState(
                session_id=f"eval_{uuid.uuid4().hex[:12]}",
                scenario=scenario,
                scenario_type=self.scenario_type,
            )
            experiment.sessions.append(session)

            if self.verbose:
                print(f"  [session {len(experiment.sessions)}] starting")

            while (
                session.active
                and len(session.turns) < self.max_turns_per_session
                and self.token_budget.can_continue(self.reserve_tokens_per_turn)
            ):
                self._run_turn(session)

                if self.verbose:
                    last = session.turns[-1]
                    jr = last.judge_result
                    error_suffix = f"  ERROR={last.error!r}" if last.error else ""
                    print(
                        f"    turn {last.turn_id}: "
                        f"failure={jr.get('failure_score', '?')} "
                        f"harm={jr.get('harm_potential', '?')} "
                        f"specificity={jr.get('specificity', '?')} "
                        f"refusal_quality={jr.get('refusal_quality', '?')} "
                        f"tokens_remaining={self.token_budget.remaining_tokens}"
                        + error_suffix
                    )

                if session.best_failure_score >= self.failure_threshold:
                    if self.verbose:
                        print(f"  [session {len(experiment.sessions)}] FAILURE (score={session.best_failure_score})")
                    session.active = False
                    break

                decision = self.session_policy.decide(session)
                session.suspicion_score = float(
                    decision.get("updated_suspicion_score", session.suspicion_score)
                )
                action = decision.get("action", "continue")

                if action == "start_new_session":
                    session.active = False
                    break
                elif action == "stop_experiment":
                    session.active = False
                    self.attack_agent.record_session(session)
                    return experiment

            self.attack_agent.record_session(session)

        return experiment

    def _run_turn(self, session: SessionState) -> None:
        user_input = ""
        chatbot_text = ""
        judge_result: dict = {}
        strategy: dict = {}
        error = ""

        try:
            turn = self.attack_agent.next_turn(session)
            strategy = turn["strategy"]

            if turn["done"]:
                session.active = False
                return

            user_input = turn["user_input"]

            target_response = self.target_client.send(
                session_id=session.session_id,
                user_input=user_input,
            )
            chatbot_text = _extract_chatbot_text(target_response)
            judge_result = self.judge.judge(user_input=user_input, chatbot_response=chatbot_text)

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        failure_score = int(judge_result.get("failure_score", 0))
        refusal_score = int(judge_result.get("refusal_score", 0))

        session.best_failure_score = max(session.best_failure_score, failure_score)
        session.repeated_refusals = (session.repeated_refusals + 1) if refusal_score > 0 else 0

        session.turns.append(TurnRecord(
            session_id=session.session_id,
            turn_id=len(session.turns) + 1,
            user_input=user_input,
            chatbot_response=chatbot_text,
            judge_result=judge_result,
            strategy_before_turn=strategy,
            error=error,
        ))
