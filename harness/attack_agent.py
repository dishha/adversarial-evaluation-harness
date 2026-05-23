from typing import Optional, Dict, Any, List

from .models import SessionState, AttackMemory
from .components import AdaptationPlanner, TurnGenerator


class AttackAgent:
    """
    Encapsulates the attack side of the evaluation loop.

    The evaluator owns the target, the judge, the budget, and the session loop.
    The AttackAgent owns everything about *how* to probe: strategy planning,
    message generation, cross-session attack memory, and persona rotation.

    Interface contract:
      next_turn(session) -> {"user_input": str, "strategy": dict, "done": bool}
      record_session(session) -> None   # call after each session ends
    """

    def __init__(
        self,
        planner: AdaptationPlanner,
        generator: TurnGenerator,
        attack_memory: Optional[AttackMemory] = None,
        persona_pool: Optional[List[str]] = None,
    ):
        self.planner = planner
        self.generator = generator
        self.attack_memory = attack_memory
        self.persona_pool = persona_pool or []
        self._session_count = 0

    @property
    def current_persona(self) -> str:
        if not self.persona_pool:
            return ""
        return self.persona_pool[self._session_count % len(self.persona_pool)]

    def next_turn(self, session: SessionState) -> Dict[str, Any]:
        strategy = self.planner.plan(session, self.attack_memory, target_persona=self.current_persona)
        if strategy.get("stop_session") is True:
            return {"user_input": "", "strategy": strategy, "done": True}

        gen = self.generator.generate(
            session=session,
            strategy_instruction=strategy.get("next_generator_instruction", "Continue evaluation safely."),
        )
        user_input = gen.get("user_input", "")
        return {
            "user_input": user_input,
            "strategy": strategy,
            "done": not bool(user_input),
        }

    def record_session(self, session: SessionState) -> None:
        if self.attack_memory is not None:
            self.attack_memory.record_session(session)
        self._session_count += 1
