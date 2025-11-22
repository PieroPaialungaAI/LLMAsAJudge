# LLM-as-a-Judge: A Modular Framework for Classification

A comprehensive, production-ready framework for using Large Language Models as judges in classification tasks. This project demonstrates best practices in prompt engineering, including structured outputs, few-shot learning, and chain-of-thought reasoning.

## Features

### 1. **Customizable Role & Task Definition**
Define exactly what your judge should do and how it should behave. No generic prompts – tailor the judge to your specific use case.

### 2. **Few-Shot Prompt Engineering**
Provide examples of good judgments to guide the LLM's behavior. This dramatically improves consistency and accuracy.

### 3. **Chain-of-Thought Reasoning with ReAct**
The judge provides:
- **Confidence scores** (0-100) for each judgment
- **Detailed reasoning** explaining the decision
- **Optional notes** for edge cases or observations

### 4. **Structured Output with Pydantic**
Production-ready code using Pydantic models ensures:
- Type safety
- Validation
- Easy serialization
- Clear schema definitions

### 5. **Framework Agnostic**
Works with multiple LLM providers:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Any provider with a compatible API

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from llm_judge import LLMJudge, FewShotExample
from openai import OpenAI

# Initialize your LLM client
client = OpenAI(api_key="your-api-key")

# Create a judge
judge = LLMJudge(
    llm_client=client,
    role="You are an expert sentiment analyzer.",
    task_description="Classify the sentiment of customer reviews.",
    classification_categories=["Positive", "Negative", "Neutral"]
)

# Judge a single input
result = judge.judge_single("This product exceeded my expectations!")
print(f"Classification: {result.classification}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")
```

## Project Structure

```
llm_judge/
├── __init__.py          # Package initialization
├── models.py            # Pydantic models for structured output
├── prompts.py           # Prompt building utilities
└── judge.py             # Main LLMJudge class

requirements.txt         # Dependencies
README.md               # This file
tutorial.ipynb          # Comprehensive tutorial notebook
```

## Tutorial Notebook

The included `tutorial.ipynb` provides a step-by-step guide covering:

1. **Prompt Engineering Fundamentals** - How to define effective judge roles
2. **Few-Shot Learning** - Adding examples to improve performance
3. **ReAct Pattern** - Using confidence and reasoning for better outputs
4. **Structured Output** - Ensuring production-ready, validated results
5. **Real-World Examples** - Complete classification tasks

## Best Practices

### 1. Be Specific in Role Definition
```python
# Bad
role = "You are a judge."

# Good
role = """You are an expert medical records classifier with 10 years of experience.
You specialize in identifying patient safety incidents from clinical notes."""
```

### 2. Provide Clear Classification Categories
```python
classification_categories = [
    "High Priority - Immediate Action Required",
    "Medium Priority - Review Within 24 Hours",
    "Low Priority - Routine Processing"
]
```

### 3. Use Few-Shot Examples
```python
examples = [
    FewShotExample(
        input_text="Patient fell in bathroom, no injuries reported.",
        expected_classification="Medium Priority",
        reasoning="While no injuries occurred, falls are safety incidents requiring review."
    )
]
```

### 4. Lower Temperature for Consistency
```python
judge = LLMJudge(
    llm_client=client,
    temperature=0.3  # More deterministic for classification
)
```

## Use Cases

- **Content Moderation** - Classify user-generated content
- **Customer Support** - Triage support tickets
- **Document Classification** - Organize and categorize documents
- **Quality Assurance** - Evaluate outputs from other systems
- **Sentiment Analysis** - Analyze customer feedback
- **Medical Records** - Classify clinical notes
- **Legal Documents** - Categorize legal filings

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this in your projects!

## Citation

If you use this framework in your research or articles, please cite:

```
LLM-as-a-Judge: A Modular Framework for Classification
https://github.com/yourusername/LLMAsAJudge
```

## Author

Built for the data science community to demonstrate production-ready LLM applications.

