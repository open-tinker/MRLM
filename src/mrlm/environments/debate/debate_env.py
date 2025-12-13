"""
Multi-agent debate environment.

This environment supports training agents to engage in structured debates,
where agents argue for opposing positions on a given topic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import random

from mrlm.core.base import BaseEnvironment, SimulatedEnvironment
from mrlm.core.types import Message, MessageRole, Observation, Reward, EnvironmentMode


class DebatePosition(Enum):
    """Position in a debate."""

    PRO = "pro"  # Supporting the proposition
    CON = "con"  # Opposing the proposition


@dataclass
class DebateTopic:
    """
    A debate topic with proposition and background.

    Attributes:
        proposition: The statement being debated
        background: Context or background information
        pro_key_points: Key arguments for the pro side
        con_key_points: Key arguments for the con side
        metadata: Additional metadata
    """

    proposition: str
    background: str = ""
    pro_key_points: List[str] = field(default_factory=list)
    con_key_points: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DebateEnvironment(SimulatedEnvironment):
    """
    Multi-agent debate environment.

    In this environment, an agent debates a topic either for (PRO) or
    against (CON) a proposition. The agent receives rewards based on
    the quality of arguments, persuasiveness, and factual accuracy.

    The environment can operate in two modes:
    - Single agent: One LLM debates against a simulated opponent
    - Multi-agent: Two LLMs debate each other (requires external coordination)
    """

    def __init__(
        self,
        topics: Optional[List[DebateTopic]] = None,
        max_turns: int = 6,
        mode: EnvironmentMode = EnvironmentMode.CLIENT,
        judge: Optional["DebateJudge"] = None,
        opponent_model: Optional[Any] = None,
    ):
        """
        Initialize debate environment.

        Args:
            topics: List of debate topics (uses defaults if None)
            max_turns: Maximum number of turns per side
            mode: Environment mode (CLIENT or SERVER)
            judge: Judge to evaluate debate quality
            opponent_model: Optional opponent model (for training)
        """
        super().__init__(mode=mode)

        self.topics = topics or self._get_default_topics()
        self.max_turns = max_turns
        self.judge = judge
        self.opponent_model = opponent_model

        # Current debate state
        self.current_topic: Optional[DebateTopic] = None
        self.current_position: Optional[DebatePosition] = None
        self.turn_count = 0
        self.debate_history: List[Tuple[str, str]] = []  # (speaker, argument)

    def _get_default_topics(self) -> List[DebateTopic]:
        """Get default debate topics."""
        return [
            DebateTopic(
                proposition="Artificial intelligence will have a net positive impact on society",
                background="AI technology is rapidly advancing and being integrated into many aspects of life.",
                pro_key_points=[
                    "Automation increases productivity and efficiency",
                    "AI can solve complex problems (disease, climate)",
                    "Improves accessibility and quality of life",
                ],
                con_key_points=[
                    "Job displacement and economic inequality",
                    "Privacy and surveillance concerns",
                    "Risk of misuse and autonomous weapons",
                ],
            ),
            DebateTopic(
                proposition="Remote work should be the default for knowledge workers",
                background="The COVID-19 pandemic accelerated adoption of remote work.",
                pro_key_points=[
                    "Improved work-life balance and flexibility",
                    "Reduced commute time and environmental impact",
                    "Access to global talent pool",
                ],
                con_key_points=[
                    "Reduced collaboration and innovation",
                    "Difficulty separating work and personal life",
                    "Not suitable for all roles and individuals",
                ],
            ),
            DebateTopic(
                proposition="Social media platforms should be legally liable for user-generated content",
                background="Social media platforms host billions of pieces of content daily.",
                pro_key_points=[
                    "Incentivizes better content moderation",
                    "Reduces spread of misinformation and harmful content",
                    "Protects vulnerable users",
                ],
                con_key_points=[
                    "Stifles free speech and innovation",
                    "Impractical to moderate all content",
                    "Platforms are intermediaries, not publishers",
                ],
            ),
        ]

    def reset(self) -> Observation:
        """
        Reset environment for a new debate.

        Returns:
            Initial observation with debate topic
        """
        # Select random topic and position
        self.current_topic = random.choice(self.topics)
        self.current_position = random.choice([DebatePosition.PRO, DebatePosition.CON])

        self.turn_count = 0
        self.debate_history = []
        self._state = {
            "topic": self.current_topic,
            "position": self.current_position,
            "turn_count": 0,
        }

        # Create initial message
        position_str = "support" if self.current_position == DebatePosition.PRO else "oppose"
        initial_message = Message(
            role=MessageRole.SYSTEM,
            content=f"""You are participating in a structured debate.

Topic: {self.current_topic.proposition}

Background: {self.current_topic.background}

Your position: You {position_str} this proposition.

Instructions:
1. Present clear, logical arguments for your position
2. Use evidence and reasoning to support your claims
3. Address potential counterarguments
4. Be persuasive and well-structured
5. You have {self.max_turns} turns to make your case

Please provide your opening argument (1-2 paragraphs).""",
        )

        return Observation(
            messages=[initial_message],
            state=self._state.copy(),
            done=False,
        )

    def step(self, action: Message) -> Tuple[Observation, Reward]:
        """
        Process agent's argument and return next state.

        Args:
            action: Agent's argument

        Returns:
            Tuple of (observation, reward)
        """
        # Extract argument
        argument = action.content

        # Record in history
        speaker = "Agent" if self.turn_count % 2 == 0 else "Opponent"
        self.debate_history.append((speaker, argument))

        self.turn_count += 1
        self._state["turn_count"] = self.turn_count

        # Check if debate is complete
        done = self.turn_count >= self.max_turns * 2  # Both sides get max_turns

        if done:
            # Debate complete - final evaluation
            reward = self._evaluate_debate()
            next_obs = Observation(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content=f"Debate complete. Your final score: {reward.value:.2f}",
                    )
                ],
                state=self._state.copy(),
                done=True,
                info={"debate_history": self.debate_history},
            )
        else:
            # Generate opponent's response (if not multi-agent)
            if self.opponent_model is None:
                opponent_arg = self._generate_opponent_argument()
                self.debate_history.append(("Opponent", opponent_arg))
                self.turn_count += 1
                self._state["turn_count"] = self.turn_count

                feedback = Message(
                    role=MessageRole.USER,
                    content=f"Opponent's argument: {opponent_arg}\n\nProvide your rebuttal and next argument.",
                )
            else:
                feedback = Message(
                    role=MessageRole.USER,
                    content="Provide your next argument.",
                )

            # Intermediate reward
            reward = self._evaluate_argument(argument)
            next_obs = Observation(
                messages=[feedback],
                state=self._state.copy(),
                done=False,
            )

        return next_obs, reward

    def _generate_opponent_argument(self) -> str:
        """Generate a simulated opponent argument."""
        # Get opposing position's key points
        if self.current_position == DebatePosition.PRO:
            key_points = self.current_topic.con_key_points
            position = "oppose"
        else:
            key_points = self.current_topic.pro_key_points
            position = "support"

        if not key_points:
            return f"I {position} the proposition for various reasons."

        # Use a random key point
        point = random.choice(key_points)
        return f"I {position} this proposition. {point}. This is a critical consideration in this debate."

    def _evaluate_argument(self, argument: str) -> Reward:
        """
        Evaluate a single argument.

        Args:
            argument: The argument text

        Returns:
            Reward for the argument
        """
        if self.judge is not None:
            return self.judge.evaluate_argument(
                argument=argument,
                topic=self.current_topic,
                position=self.current_position,
            )

        # Default: simple length-based scoring
        score = min(len(argument) / 200.0, 1.0)  # Reward longer, detailed arguments
        return Reward(
            value=score,
            components={"length": score},
            info={"argument_length": len(argument)},
        )

    def _evaluate_debate(self) -> Reward:
        """
        Evaluate the complete debate.

        Returns:
            Final reward for the debate
        """
        if self.judge is not None:
            return self.judge.evaluate_debate(
                debate_history=self.debate_history,
                topic=self.current_topic,
                position=self.current_position,
            )

        # Default: average of argument lengths
        agent_args = [arg for speaker, arg in self.debate_history if speaker == "Agent"]
        avg_length = sum(len(arg) for arg in agent_args) / max(len(agent_args), 1)
        score = min(avg_length / 200.0, 1.0)

        return Reward(
            value=score,
            components={"average_quality": score},
            info={"num_arguments": len(agent_args)},
        )

    def close(self):
        """Clean up resources."""
        pass
