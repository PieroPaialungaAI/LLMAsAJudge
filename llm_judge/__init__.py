"""
LLM-as-a-Judge: A modular framework for using LLMs as classification judges.
"""

from .judge import LLMJudge
from .models import JudgmentResult, FewShotExample
from .prompts import PromptBuilder

__all__ = ["LLMJudge", "JudgmentResult", "FewShotExample", "PromptBuilder"]
__version__ = "0.1.0"

