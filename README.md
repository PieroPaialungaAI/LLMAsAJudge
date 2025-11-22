# LLM-as-a-Judge: A Modular Framework for Evaluating AI Model Outputs

A comprehensive, production-ready framework for using Large Language Models to **evaluate existing AI model outputs**. This is not about using LLMs to classify - it's about using LLMs to judge whether other AI systems' predictions are correct and high-quality. This project demonstrates best practices in prompt engineering, including structured outputs, few-shot learning, and chain-of-thought reasoning.

## What is LLM-as-a-Judge?

**Important:** LLM-as-a-Judge is NOT about using an LLM to classify text. That's just an LLM classifier.

**LLM-as-a-Judge** is about using an LLM to **evaluate whether an existing AI model's output is correct, accurate, or high-quality**.

### The Real Use Case

You have an AI system (sentiment classifier, chatbot, content generator, etc.) producing outputs. How do you know if those outputs are good? 

- ❌ Manual human evaluation (expensive, slow)
- ❌ Ground truth comparison (requires labeled data you often don't have)
- ✅ **LLM-as-a-Judge** - Use a powerful LLM to evaluate each prediction at scale

## Features

### 1. **Evaluates Existing AI Outputs**
Takes an input + model output pair and judges whether the output is correct/high-quality. Does not create classifications itself.

### 2. **Customizable Evaluation Criteria**
Define exactly what makes an output "good" or "bad" for your specific use case. No generic prompts.

### 3. **Few-Shot Prompt Engineering**
Provide examples of correct and incorrect AI outputs with explanations. This teaches the judge what to look for.

### 4. **Chain-of-Thought Reasoning with ReAct**
The judge provides:
- **Quality scores** (0-100) evaluating the AI output
- **Verdicts** (Correct/Incorrect/Partially Correct)
- **Confidence scores** (0-100) for each judgment
- **Detailed reasoning** explaining why
- **Optional notes** for edge cases or improvements

### 5. **Structured Output with Pydantic**
Production-ready code using Pydantic models ensures:
- Type safety and validation
- Easy serialization to JSON/CSV
- Clear schema definitions
- No parsing errors

### 6. **Framework Agnostic**
Works with multiple LLM providers:
- OpenAI (GPT-4, GPT-4o, GPT-3.5)
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

# Create a judge to evaluate sentiment classifier outputs
judge = LLMJudge(
    llm_client=client,
    role="You are an expert evaluator of sentiment classification models.",
    task_description="Evaluate whether AI sentiment predictions are accurate.",
    evaluation_criteria="Consider sentiment intensity, context, and mixed sentiments.",
    valid_verdicts=["Correct", "Incorrect", "Partially Correct"]
)

# You have an AI model that made a prediction - let's judge it
input_text = "This product exceeded my expectations!"
model_prediction = "Positive"  # This is what YOUR model predicted

# Judge whether the model's output is correct
result = judge.judge_single(
    input_text=input_text,
    model_output=model_prediction
)

print(f"Verdict: {result.verdict}")           # Correct/Incorrect
print(f"Score: {result.score}/100")           # Quality score
print(f"Confidence: {result.confidence}%")    # How certain
print(f"Reasoning: {result.reasoning}")       # Why this verdict
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

The included `tutorial.ipynb` provides a comprehensive step-by-step guide covering:

1. **Understanding LLM-as-a-Judge** - The difference between judging and classifying
2. **Role and Task Definition** - How to define effective evaluation criteria
3. **Few-Shot Examples** - Teaching the judge what good vs. bad outputs look like
4. **ReAct Pattern** - Using confidence scores and reasoning for transparency
5. **Structured Output** - Production-ready Pydantic models
6. **Real-World Examples** - Evaluating sentiment classifiers and chatbot responses
7. **Batch Processing** - Analyzing model performance at scale

## Best Practices

### 1. Be Specific in Role Definition
```python
# Bad
role = "You are a judge."

# Good
role = """You are an expert evaluator of sentiment classification models with 10 years 
of NLP experience. You assess whether AI sentiment predictions accurately capture the 
tone and intensity of customer reviews."""
```

### 2. Define Clear Evaluation Criteria
```python
evaluation_criteria = """
A sentiment prediction is correct if:
- It captures the dominant sentiment (positive/negative/neutral)
- It accounts for sentiment intensity ('amazing' vs 'okay')
- It handles mixed sentiments appropriately
- It recognizes sarcasm and context
"""
```

### 3. Use Few-Shot Examples (Without Ground Truth!)
```python
examples = [
    FewShotExample(
        input_text="It's fine, does what it says on the box.",
        model_output="Positive",  # What the AI predicted
        expected_verdict="Incorrect",
        expected_score=20.0,
        reasoning="The model incorrectly classified this as Positive when it should be Neutral. 'Fine' indicates lukewarm sentiment, not satisfaction."
    )
]
```

**Critical:** Don't include ground truth in examples - the judge must learn to evaluate intrinsically.

### 4. Lower Temperature for Consistency
```python
judge = LLMJudge(
    llm_client=client,
    temperature=0.3  # More deterministic judgments
)
```

### 5. Never Pass Ground Truth to the Judge
```python
# The judge only sees input + model output
# Ground truth (if available) is for YOUR evaluation of the judge, not for the judge itself
result = judge.judge_single(
    input_text=text,
    model_output=your_models_prediction
    # NO ground_truth parameter!
)
```

## Use Cases

- **Model Evaluation** - Compare model versions without manual labeling
- **Quality Control** - Flag problematic AI outputs before they reach users
- **Active Learning** - Identify predictions that need human review (low confidence)
- **A/B Testing** - Evaluate which model variant performs better
- **Continuous Monitoring** - Track model quality in production over time
- **Model Debugging** - Understand patterns in when/why your model fails
- **Chatbot QA** - Evaluate if chatbot responses are helpful and accurate
- **Content Generation** - Judge quality of AI-generated text
- **Classification Audit** - Verify sentiment analyzers, content moderators, etc.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this in your projects!

## Key Advantages

1. **No Manual Labeling Required** - Evaluate at scale without human annotators
2. **Works Without Ground Truth** - Judge quality even when you don't have labels
3. **Transparent Reasoning** - See exactly why each judgment was made
4. **Confidence-Based Routing** - Automate high-confidence cases, review low-confidence
5. **Flexible and Modular** - Adapt to any AI system evaluation task

## Citation

If you use this framework in your research or articles, please cite:

```
LLM-as-a-Judge: A Modular Framework for Evaluating AI Model Outputs
https://github.com/yourusername/LLMAsAJudge
```

## Author

Built for the data science community to demonstrate production-ready LLM-as-a-Judge applications.

