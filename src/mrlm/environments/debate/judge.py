"""
Debate judges for evaluating argument quality.

Judges can be rule-based or LLM-based.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import re

from mrlm.core.types import Reward
from mrlm.environments.debate.debate_env import DebateTopic, DebatePosition


class DebateJudge(ABC):
    """Abstract base class for debate judges."""

    @abstractmethod
    def evaluate_argument(
        self,
        argument: str,
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """
        Evaluate a single argument.

        Args:
            argument: The argument text
            topic: The debate topic
            position: The debater's position

        Returns:
            Reward for the argument
        """
        pass

    @abstractmethod
    def evaluate_debate(
        self,
        debate_history: List[Tuple[str, str]],
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """
        Evaluate the complete debate.

        Args:
            debate_history: List of (speaker, argument) tuples
            topic: The debate topic
            position: The debater's position

        Returns:
            Final reward for the debate
        """
        pass


class RuleBasedJudge(DebateJudge):
    """
    Rule-based judge using heuristics.

    Evaluates arguments based on:
    - Length and detail
    - Presence of evidence/reasoning keywords
    - Structure and coherence
    - Relevance to topic
    """

    def __init__(
        self,
        min_length: int = 100,
        max_length: int = 500,
        evidence_keywords: Optional[List[str]] = None,
        reasoning_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize rule-based judge.

        Args:
            min_length: Minimum desired argument length
            max_length: Maximum desired argument length
            evidence_keywords: Keywords indicating evidence
            reasoning_keywords: Keywords indicating logical reasoning
        """
        self.min_length = min_length
        self.max_length = max_length

        self.evidence_keywords = evidence_keywords or [
            "research shows",
            "studies indicate",
            "according to",
            "evidence suggests",
            "data shows",
            "statistics",
            "for example",
            "for instance",
        ]

        self.reasoning_keywords = reasoning_keywords or [
            "therefore",
            "thus",
            "because",
            "consequently",
            "as a result",
            "this means",
            "implies that",
            "follows that",
        ]

    def evaluate_argument(
        self,
        argument: str,
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """Evaluate a single argument using heuristics."""
        components = {}

        # Length score
        length = len(argument)
        if length < self.min_length:
            length_score = length / self.min_length
        elif length > self.max_length:
            length_score = 1.0 - (length - self.max_length) / self.max_length
            length_score = max(0.5, length_score)
        else:
            length_score = 1.0
        components["length"] = length_score

        # Evidence score
        argument_lower = argument.lower()
        evidence_count = sum(
            1 for keyword in self.evidence_keywords if keyword in argument_lower
        )
        evidence_score = min(evidence_count / 2, 1.0)  # Up to 2 evidence markers
        components["evidence"] = evidence_score

        # Reasoning score
        reasoning_count = sum(
            1 for keyword in self.reasoning_keywords if keyword in argument_lower
        )
        reasoning_score = min(reasoning_count / 2, 1.0)  # Up to 2 reasoning markers
        components["reasoning"] = reasoning_score

        # Structure score (sentences, paragraphs)
        sentences = len(re.findall(r'[.!?]+', argument))
        structure_score = min(sentences / 5, 1.0)  # Up to 5 sentences
        components["structure"] = structure_score

        # Relevance score (mentions topic keywords)
        topic_keywords = topic.proposition.lower().split()
        topic_keywords = [w for w in topic_keywords if len(w) > 3]  # Filter short words
        relevance_count = sum(
            1 for keyword in topic_keywords if keyword in argument_lower
        )
        relevance_score = min(relevance_count / 3, 1.0)
        components["relevance"] = relevance_score

        # Overall score (weighted average)
        total_score = (
            0.2 * length_score
            + 0.25 * evidence_score
            + 0.25 * reasoning_score
            + 0.15 * structure_score
            + 0.15 * relevance_score
        )

        return Reward(
            value=total_score,
            components=components,
            info={
                "argument_length": length,
                "evidence_markers": evidence_count,
                "reasoning_markers": reasoning_count,
                "sentences": sentences,
            },
        )

    def evaluate_debate(
        self,
        debate_history: List[Tuple[str, str]],
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """Evaluate complete debate."""
        # Extract agent's arguments
        agent_args = [arg for speaker, arg in debate_history if speaker == "Agent"]

        if not agent_args:
            return Reward(value=0.0)

        # Evaluate each argument
        argument_rewards = [
            self.evaluate_argument(arg, topic, position) for arg in agent_args
        ]

        # Average scores
        avg_score = sum(r.value for r in argument_rewards) / len(argument_rewards)

        # Consistency bonus: check if arguments build on each other
        consistency_score = self._evaluate_consistency(agent_args)

        # Coverage: check if key points are addressed
        coverage_score = self._evaluate_coverage(agent_args, topic, position)

        # Final score
        final_score = 0.6 * avg_score + 0.2 * consistency_score + 0.2 * coverage_score

        return Reward(
            value=final_score,
            components={
                "average_argument": avg_score,
                "consistency": consistency_score,
                "coverage": coverage_score,
            },
            info={
                "num_arguments": len(agent_args),
                "total_words": sum(len(arg.split()) for arg in agent_args),
            },
        )

    def _evaluate_consistency(self, arguments: List[str]) -> float:
        """
        Evaluate consistency across arguments.

        Simple heuristic: check for shared vocabulary.
        """
        if len(arguments) < 2:
            return 1.0

        # Get word sets for each argument
        word_sets = [set(arg.lower().split()) for arg in arguments]

        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(word_sets) - 1):
            intersection = word_sets[i] & word_sets[i + 1]
            union = word_sets[i] | word_sets[i + 1]
            if union:
                sim = len(intersection) / len(union)
                similarities.append(sim)

        return sum(similarities) / len(similarities) if similarities else 0.5

    def _evaluate_coverage(
        self, arguments: List[str], topic: DebateTopic, position: DebatePosition
    ) -> float:
        """
        Evaluate how well arguments cover key points.

        Args:
            arguments: List of arguments
            topic: Debate topic
            position: Debater's position

        Returns:
            Coverage score (0-1)
        """
        # Get relevant key points
        if position == DebatePosition.PRO:
            key_points = topic.pro_key_points
        else:
            key_points = topic.con_key_points

        if not key_points:
            return 0.5  # Neutral if no key points defined

        # Check how many key points are mentioned
        combined_args = " ".join(arguments).lower()
        covered = 0

        for point in key_points:
            # Check if keywords from point appear in arguments
            point_words = [w for w in point.lower().split() if len(w) > 3]
            if any(word in combined_args for word in point_words):
                covered += 1

        return covered / len(key_points) if key_points else 0.5


class LLMJudge(DebateJudge):
    """
    LLM-based judge using a language model to evaluate arguments.

    This provides more nuanced evaluation but requires an LLM.
    """

    def __init__(self, model, tokenizer):
        """
        Initialize LLM judge.

        Args:
            model: Language model for evaluation
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer

    def evaluate_argument(
        self,
        argument: str,
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """Evaluate argument using LLM."""
        # Create evaluation prompt
        position_str = "supporting" if position == DebatePosition.PRO else "opposing"
        prompt = f"""Evaluate the following debate argument on a scale of 0-1.

Topic: {topic.proposition}
Position: {position_str}

Argument:
{argument}

Criteria:
1. Clarity and structure (0-0.2)
2. Evidence and reasoning (0-0.3)
3. Persuasiveness (0-0.3)
4. Relevance to topic (0-0.2)

Provide only a numerical score between 0 and 1."""

        # Generate evaluation (simplified - in practice you'd parse the response)
        # For now, return a placeholder
        # TODO: Implement actual LLM evaluation
        return Reward(value=0.5, components={"llm_score": 0.5})

    def evaluate_debate(
        self,
        debate_history: List[Tuple[str, str]],
        topic: DebateTopic,
        position: DebatePosition,
    ) -> Reward:
        """Evaluate complete debate using LLM."""
        # Similar to evaluate_argument but considers full debate
        # TODO: Implement actual LLM evaluation
        return Reward(value=0.5, components={"llm_score": 0.5})
