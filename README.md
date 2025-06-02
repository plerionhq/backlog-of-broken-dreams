# AI-Powered Security Issue Prioritization

A collection of tools that use AWS Bedrock's models to intelligently prioritize security issues through different approaches: naive scoring, Elo ranking, and bubble sort comparison.

For a detailed explanation of how these tools work and when to use each approach, check out our [blog post](https://www.plerion.com/blog/automatically-prioritize-security-issues-from-different-tools-with-an-llm).

## Overview

These tools help security teams prioritize issues by using AI to analyze the context and impact of each issue, rather than relying solely on scanner-provided severity ratings. Each tool takes a different approach to prioritization:

1. **Naive Scoring** (`score_sort.py`): Assigns a 1-100 score to each issue independently
2. **Elo Ranking** (`elo_sort.py`): Uses pairwise comparisons with Elo scoring for relative ranking
3. **Bubble Sort** (`bubble_sort.py`): Uses pairwise comparisons with bubble sort for deterministic ordering

## Features

- Supports [any AWS Bedrock model](https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html)
- Natural language reasoning for prioritization decisions
- Detailed comparison/scoring rationale for each issue
- Progress tracking with tqdm
- Runtime performance metrics
- Customizable prompts via template files

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Configure AWS credentials for Bedrock access

## Usage

### Naive Scoring (`score_sort.py`)

Fastest method that scores each issue independently on a 1-100 scale.

```bash
./score_sort.py [--issues input.json] [--output prioritized.json] [--model <model_id>] [--prompt-file prompt.txt]
```

Pros:
- Much faster than pairwise methods (linear vs quadratic)
- Easier to explain and interpret
- Allows consistent re-evaluation as issues change

Cons:
- Less sensitive to subtle relative differences
- Requires a strong prompt to ensure score accuracy and consistency

### Elo Ranking (`elo_sort.py`)

Uses Elo scoring system for pairwise comparisons, allowing partial comparison sets.

```bash
./elo_sort.py [--issues input.json] [--output prioritized.json] [--model <model_id>] [--max-comparisons <multiplier>] [--prompt-file prompt.txt]
```

Pros:
- More efficient than bubble sort (fewer comparisons for large n)
- Can gracefully handle partial comparison sets
- Maintains ranking stability as more comparisons are added
- Still deterministic with fixed seeds and models

Cons:
- Order can be affected by comparison sequence
- May need tuning (K-factor) for best behavior
- Harder to explain intuitively than bubble sort

### Bubble Sort (`bubble_sort.py`)

Uses bubble sort algorithm for deterministic pairwise comparisons.

```bash
./bubble_sort.py [--issues input.json] [--output prioritized.json] [--model <model_id>] [--prompt-file prompt.txt]
```

Pros:
- Easy to implement and debug
- Deterministic: same input = same output
- Ensures transitive ordering based purely on model comparisons
- Makes it easy to store and explain the reasoning behind each swap

Cons:
- Poor performance at scale (O(n²) comparisons)
- Expensive if each comparison requires an LLM call
- Not suitable for very large input sets without optimization or batching

## Common Options

All tools support these common options:
- `--issues`: Path to input file containing unprioritized issues (default: input-issues.json)
- `--output`: Output file for sorted and annotated issues (default: prioritized-issues.json)
- `--model`: AWS Bedrock model ID (default: anthropic.claude-3-haiku-20240307-v1:0)
- `--summary-only`: Show prioritization summary from existing output file without rerunning
- `--prompt-file`: Path to the prompt template file (default: [tool]_prompt.txt)

## Input Format

The input JSON file should have this structure:
```json
{
  "issues": [
    {
      "id": "unique-id",
      "severityLevel": "CRITICAL|HIGH|MEDIUM|LOW",
      "message": "Issue description",
      "type": "issue-type"
    }
  ]
}
```

The LLM can process any additional fields you include in the issues. Feel free to add any fields that might help the AI make better prioritization or include anything fields you get from whatever tools generate security issues.

The more context you provide, the better the AI can understand and prioritize your issues.

## Output Format

The output JSON file will contain the same issues, sorted by priority, with additional metadata:
- For score_sort.py: `score` and `reasoning` fields
- For elo_sort.py: `elo` score and `comparison_reasoning` array
- For bubble_sort.py: `comparison_reasoning` array

## Prompt Customization

Each tool uses a template file for its prompts:
- `score_prompt.txt`: For naive scoring
- `elo_prompt.txt`: For Elo ranking comparisons
- `bubble_prompt.txt`: For bubble sort comparisons

You can customize these prompts by:
1. Modifying the default prompt files
2. Creating your own prompt files and using `--prompt-file`

## Author

Daniel Grzelak (@dagrz on X, daniel.grzelak@plerion.com)

And if you don't want to mess around with all that, or you just want something now, let our [AI cloud security teammate Pleri](https://www.plerion.com/pleri-ai "https://www.plerion.com/pleri-ai") do the prioritization for you, in Slack, in Jira, or wherever else you might work.