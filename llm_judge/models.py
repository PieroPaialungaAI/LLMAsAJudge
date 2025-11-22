"""
Pydantic models for structured output from the LLM judge.
"""

from typing import Optional, List, Any
from pydantic import BaseModel, Field


class JudgmentResult(BaseModel):
    """
    Structured output for judging an AI model's output.
    
    This model enforces structured output from the LLM judge, ensuring:
    - A quality score evaluating the AI output
    - A binary verdict (correct/incorrect or pass/fail)
    - A confidence score (0-100) in the judgment
    - Chain-of-thought reasoning
    - Optional notes for additional context
    """
    score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Quality score from 0-100 evaluating how good the AI output is"
    )
    
    verdict: str = Field(
        ...,
        description="Binary judgment: 'Correct', 'Incorrect', 'Partially Correct', 'Pass', 'Fail', etc."
    )
    
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence score from 0 to 100 indicating the judge's certainty in this evaluation"
    )
    
    reasoning: str = Field(
        ...,
        description="Chain-of-thought reasoning explaining why the AI output received this judgment"
    )
    
    notes: Optional[str] = Field(
        None,
        description="Additional observations, edge cases, or suggestions for improvement"
    )


class FewShotExample(BaseModel):
    """
    A few-shot example to guide the judge's evaluation behavior.
    
    Provides the judge with concrete examples of:
    - What the original input looks like
    - What the AI model predicted/output
    - How to evaluate whether that output is good or not
    - What judgment to give
    
    NOTE: Ground truth is deliberately NOT included here - the judge must
    evaluate outputs based on quality and appropriateness, not by comparing
    to a known answer.
    """
    input_text: str = Field(
        ...,
        description="The original input given to the AI model"
    )
    
    model_output: str = Field(
        ...,
        description="The AI model's prediction or output for this input"
    )
    
    expected_verdict: str = Field(
        ...,
        description="The correct judgment (e.g., 'Correct', 'Incorrect', 'Pass', 'Fail')"
    )
    
    expected_score: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Optional quality score for this example"
    )
    
    reasoning: str = Field(
        ...,
        description="The reasoning explaining why this verdict was given"
    )

