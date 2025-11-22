"""
Prompt building utilities for the LLM judge.
"""

from typing import List, Optional
from .models import FewShotExample


class PromptBuilder:
    """
    Builds prompts for the LLM judge with support for:
    1. Custom role definitions
    2. Few-shot examples
    3. Chain-of-thought reasoning
    4. Structured output requirements
    """
    
    DEFAULT_ROLE = """You are an expert judge tasked with evaluating AI model outputs.
Your role is to carefully assess whether an AI system's predictions or classifications are correct and high-quality.
You must be objective, consistent, and thorough in your evaluations, identifying both strengths and weaknesses."""
    
    DEFAULT_TASK = """For each AI model output, you must:
1. Carefully read the original input and the AI model's output
2. Evaluate whether the output is correct, appropriate, and high-quality
3. Provide a quality score (0-100) and verdict (Correct/Incorrect/Partially Correct)
4. Explain your reasoning using chain-of-thought
5. Add any relevant notes, suggestions for improvement, or edge case observations"""
    
    def __init__(
        self,
        role: Optional[str] = None,
        task_description: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        valid_verdicts: Optional[List[str]] = None,
        few_shot_examples: Optional[List[FewShotExample]] = None
    ):
        """
        Initialize the prompt builder.
        
        Args:
            role: Custom role definition for the judge (defaults to generic judge)
            task_description: Specific task the judge should perform
            evaluation_criteria: Specific criteria for evaluating AI outputs
            valid_verdicts: List of valid verdict options (e.g., ['Correct', 'Incorrect'])
            few_shot_examples: Few-shot examples to guide the judge
        """
        self.role = role or self.DEFAULT_ROLE
        self.task_description = task_description or self.DEFAULT_TASK
        self.evaluation_criteria = evaluation_criteria
        self.valid_verdicts = valid_verdicts
        self.few_shot_examples = few_shot_examples or []
    
    def build_system_prompt(self) -> str:
        """
        Build the system prompt that defines the judge's role and task.
        
        Returns:
            A comprehensive system prompt
        """
        prompt_parts = [
            "# ROLE",
            self.role,
            "",
            "# TASK",
            self.task_description,
        ]
        
        if self.evaluation_criteria:
            prompt_parts.extend([
                "",
                "# EVALUATION CRITERIA",
                self.evaluation_criteria,
            ])
        
        if self.valid_verdicts:
            prompt_parts.extend([
                "",
                "# VALID VERDICTS",
                "You must choose one of the following verdicts:",
                *[f"- {verdict}" for verdict in self.valid_verdicts],
            ])
        
        prompt_parts.extend([
            "",
            "# OUTPUT FORMAT",
            "You must provide your judgment in the following structured format:",
            "- score: A quality score from 0-100 evaluating the AI output",
            "- verdict: Your judgment (e.g., Correct, Incorrect, Partially Correct)",
            "- confidence: A score from 0-100 indicating your certainty in this judgment",
            "- reasoning: Your chain-of-thought explanation for why you gave this verdict",
            "- notes: Any additional observations or suggestions for improvement (optional)",
        ])
        
        return "\n".join(prompt_parts)
    
    def build_few_shot_section(self) -> str:
        """
        Build the few-shot examples section.
        
        Returns:
            Formatted few-shot examples or empty string if none provided
        """
        if not self.few_shot_examples:
            return ""
        
        sections = ["# EXAMPLES\n", "Here are some examples of good judgments:\n"]
        
        for i, example in enumerate(self.few_shot_examples, 1):
            sections.append(f"## Example {i}")
            sections.append(f"**Original Input:** {example.input_text}")
            sections.append(f"**AI Model Output:** {example.model_output}")
            sections.append(f"**Verdict:** {example.expected_verdict}")
            if example.expected_score is not None:
                sections.append(f"**Score:** {example.expected_score}")
            sections.append(f"**Reasoning:** {example.reasoning}")
            sections.append("")
        
        return "\n".join(sections)
    
    def build_user_prompt(self, input_text: str, model_output: str) -> str:
        """
        Build the user prompt for judging a specific AI model output.
        
        Args:
            input_text: The original input given to the AI model
            model_output: The AI model's prediction/output
            
        Returns:
            Formatted user prompt
            
        Note:
            Ground truth is deliberately NOT included in the prompt. The judge must
            evaluate quality based on the input and output alone, not by comparing
            to a known correct answer.
        """
        few_shot_section = self.build_few_shot_section()
        
        prompt_parts = []
        if few_shot_section:
            prompt_parts.append(few_shot_section)
        
        prompt_parts.append("# AI OUTPUT TO EVALUATE\n")
        prompt_parts.append(f"**Original Input:** {input_text}\n")
        prompt_parts.append(f"**AI Model Output:** {model_output}\n")
        prompt_parts.append("\nEvaluate the AI model's output. Is it correct? Is it high-quality? Provide your judgment.")
        
        return "\n".join(prompt_parts)

