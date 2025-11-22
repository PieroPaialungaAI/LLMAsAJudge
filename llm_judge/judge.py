"""
Main LLM Judge class that orchestrates the classification process.
"""

from typing import List, Optional, Dict, Any, Callable, Tuple
import json
from .models import JudgmentResult, FewShotExample
from .prompts import PromptBuilder


class LLMJudge:
    """
    A modular LLM-as-a-Judge system for evaluating AI model outputs.
    
    This class provides a framework-agnostic approach to using LLMs as judges.
    It evaluates whether existing AI predictions/classifications are correct and high-quality.
    
    It supports:
    - Custom role and task definitions
    - Few-shot prompting with evaluation examples
    - Chain-of-thought reasoning with confidence scores
    - Structured output via Pydantic
    - Multiple LLM backends
    """
    
    def __init__(
        self,
        llm_client: Any,
        role: Optional[str] = None,
        task_description: Optional[str] = None,
        evaluation_criteria: Optional[str] = None,
        valid_verdicts: Optional[List[str]] = None,
        few_shot_examples: Optional[List[FewShotExample]] = None,
        model_name: str = "gpt-4o",
        temperature: float = 0.3,
    ):
        """
        Initialize the LLM Judge.
        
        Args:
            llm_client: The LLM client object (OpenAI, Anthropic, etc.)
            role: Custom role definition for the judge
            task_description: Specific evaluation task the judge should perform
            evaluation_criteria: Criteria for judging output quality
            valid_verdicts: List of valid verdict options (e.g., ['Correct', 'Incorrect'])
            few_shot_examples: Few-shot examples showing how to judge outputs
            model_name: Name of the model to use
            temperature: Temperature for generation (lower = more consistent)
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.temperature = temperature
        
        # Build prompts
        self.prompt_builder = PromptBuilder(
            role=role,
            task_description=task_description,
            evaluation_criteria=evaluation_criteria,
            valid_verdicts=valid_verdicts,
            few_shot_examples=few_shot_examples
        )
        
        self.system_prompt = self.prompt_builder.build_system_prompt()
    
    def judge_single(
        self,
        input_text: str,
        model_output: str,
        use_structured_output: bool = True
    ) -> JudgmentResult:
        """
        Judge a single AI model output.
        
        Args:
            input_text: The original input given to the AI model
            model_output: The AI model's prediction/output to evaluate
            use_structured_output: Whether to enforce structured output
            
        Returns:
            JudgmentResult with score, verdict, confidence, reasoning, and notes
            
        Note:
            Ground truth is NOT passed to the LLM. The judge evaluates outputs
            based purely on the input and output quality, as it would in real
            production scenarios where ground truth is not available.
        """
        user_prompt = self.prompt_builder.build_user_prompt(input_text, model_output)
        
        if use_structured_output:
            return self._judge_with_structured_output(user_prompt)
        else:
            return self._judge_with_parsing(user_prompt)
    
    def judge_batch(
        self,
        inputs: List[Tuple[str, str]],
        use_structured_output: bool = True
    ) -> List[JudgmentResult]:
        """
        Judge multiple AI model outputs.
        
        Args:
            inputs: List of tuples (input_text, model_output)
            use_structured_output: Whether to enforce structured output
            
        Returns:
            List of JudgmentResult objects
        """
        results = []
        for input_text, model_output in inputs:
            result = self.judge_single(input_text, model_output, use_structured_output)
            results.append(result)
        return results
    
    def _judge_with_structured_output(self, user_prompt: str) -> JudgmentResult:
        """
        Use structured output (beta feature in OpenAI, similar in other providers).
        
        This ensures the LLM output conforms to our Pydantic schema.
        """
        try:
            # Try OpenAI-style structured output
            if hasattr(self.llm_client, 'beta') and hasattr(self.llm_client.beta, 'chat'):
                completion = self.llm_client.beta.chat.completions.parse(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format=JudgmentResult,
                    temperature=self.temperature
                )
                return completion.choices[0].message.parsed
            
            # Fallback to standard completion with JSON mode
            elif hasattr(self.llm_client, 'chat'):
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt + "\n\nYou MUST respond with valid JSON matching this schema: {\"classification\": str, \"confidence\": float, \"reasoning\": str, \"notes\": str}"},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=self.temperature
                )
                content = completion.choices[0].message.content
                data = json.loads(content)
                return JudgmentResult(**data)
            
            else:
                raise ValueError("Unsupported LLM client. Please implement a custom adapter.")
                
        except Exception as e:
            raise RuntimeError(f"Failed to get structured output from LLM: {e}")
    
    def _judge_with_parsing(self, user_prompt: str) -> JudgmentResult:
        """
        Fallback method that parses the LLM output manually.
        
        Use this if structured output is not available for your LLM.
        """
        try:
            if hasattr(self.llm_client, 'chat'):
                completion = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=self.temperature
                )
                content = completion.choices[0].message.content
            else:
                raise ValueError("Unsupported LLM client")
            
            # Try to parse as JSON first
            try:
                data = json.loads(content)
                return JudgmentResult(**data)
            except json.JSONDecodeError:
                # Manual parsing fallback
                return self._parse_text_response(content)
                
        except Exception as e:
            raise RuntimeError(f"Failed to get judgment from LLM: {e}")
    
    def _parse_text_response(self, response: str) -> JudgmentResult:
        """
        Parse a text response into a JudgmentResult.
        
        This is a basic parser and may need to be customized.
        """
        # Simple heuristic parsing (can be improved)
        lines = response.strip().split('\n')
        data = {
            "score": 50.0,
            "verdict": "unknown",
            "confidence": 50.0,
            "reasoning": response,
            "notes": None
        }
        
        import re
        for line in lines:
            line_lower = line.lower()
            if 'score:' in line_lower:
                score_str = line.split(':', 1)[1].strip()
                match = re.search(r'(\d+(?:\.\d+)?)', score_str)
                if match:
                    data['score'] = float(match.group(1))
            elif 'verdict:' in line_lower:
                data['verdict'] = line.split(':', 1)[1].strip()
            elif 'confidence:' in line_lower:
                conf_str = line.split(':', 1)[1].strip()
                match = re.search(r'(\d+(?:\.\d+)?)', conf_str)
                if match:
                    data['confidence'] = float(match.group(1))
            elif 'reasoning:' in line_lower:
                data['reasoning'] = line.split(':', 1)[1].strip()
            elif 'notes:' in line_lower:
                data['notes'] = line.split(':', 1)[1].strip()
        
        return JudgmentResult(**data)
    
    def get_system_prompt(self) -> str:
        """Get the current system prompt being used."""
        return self.system_prompt
    
    def add_few_shot_example(self, example: FewShotExample):
        """
        Add a few-shot example to the judge.
        
        This will rebuild the prompts with the new example.
        """
        self.prompt_builder.few_shot_examples.append(example)
        self.system_prompt = self.prompt_builder.build_system_prompt()

