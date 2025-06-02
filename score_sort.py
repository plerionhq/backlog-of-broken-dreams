#!/usr/bin/env python3

"""
LLM-Based Security Issue Scorer: Direct 1–100 Severity Scoring
This script uses a large language model (LLM) via AWS Bedrock to assign a 1–100 severity score
for each security issue based on natural language analysis.

Instead of relying on inconsistent scanner severity labels or pairwise comparisons (e.g., bubble sort, Elo),
it directly scores each issue using contextual reasoning across impact, reachability, exploitability,
and business context.

Pros:
- Much faster than pairwise methods (linear vs quadratic)
- Easier to explain and interpret
- Allows consistent re-evaluation as issues change

Cons:
- Less sensitive to subtle relative differences
- Requires a strong prompt to ensure score accuracy and consistency

Features:
- Supports Claude models via AWS Bedrock (Haiku, Sonnet, Opus)
- Scores each issue independently from 1 (least severe) to 100 (most urgent)
- Annotates each issue with rationale

Usage:
    ./score_sort.py [--issues input.json] [--output prioritized.json] [--model <model_id>] [--prompt-file prompt.txt]

Options:
    --issues         Path to input file containing unprioritized issues (default: input-issues.json)
    --output         Output file for sorted and annotated issues (default: prioritized-issues.json)
    --model          AWS Bedrock model ID (e.g., Claude Haiku, Sonnet, or Opus)
    --summary-only   Show prioritization summary from existing output file without rerunning the model
    --prompt-file    Path to the prompt template file (default: score_prompt.txt)

Output:
    - `prioritized-issues.json`: Sorted issues with scoring metadata and reasoning
    - Terminal summary: Ranked list with score, severity, and truncated title/message

Written by: Daniel Grzelak (@dagrz on X, daniel.grzelak@plerion.com)  
For more tools and security automation, visit: https://www.plerion.com
"""

import argparse
import json
import boto3
from typing import List, Dict, Any
from tqdm import tqdm
import re
import time

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM scoring for security issues")
    parser.add_argument("--issues", type=str, default="input-issues.json")
    parser.add_argument("--output", type=str, default="prioritized-issues.json")
    parser.add_argument("--model", type=str, default="anthropic.claude-3-haiku-20240307-v1:0")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--prompt-file", type=str, default="score_prompt.txt",
                      help="Path to the prompt template file (default: score_prompt.txt)")
    return parser.parse_args()

def load_issues(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r') as f:
        return json.load(f).get('issues', [])

def save_issues(issues: List[Dict[str, Any]], file_path: str) -> None:
    with open(file_path, 'w') as f:
        json.dump({'issues': issues}, f, indent=2)

def sanitize_json_string(s: str) -> str:
    s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)
    return ''.join(c for c in s if c.isprintable() or c.isspace())

def load_prompt(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, 'r') as f:
        return f.read()

def create_scoring_prompt(issue: Dict[str, Any], prompt_template: str) -> str:
    return prompt_template.format(issue=json.dumps(issue, indent=2))

def score_issue(issue: Dict[str, Any], bedrock_client, model_id: str, prompt_template: str) -> Dict[str, Any]:
    prompt = create_scoring_prompt(issue, prompt_template)
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            })
        )
        response_text = json.loads(response['body'].read())['content'][0]['text']
        result = json.loads(sanitize_json_string(response_text))
        issue['score'] = result['score']
        issue['reasoning'] = result['reasoning']
    except Exception as e:
        print(f"Error scoring issue: {e}")
        issue['score'] = 50
        issue['reasoning'] = f"Fallback score due to error: {e}"
    return issue

def print_summary(issues: List[Dict[str, Any]], runtime: float) -> None:
    width = 140
    title_width = width - 45  # Reduced to make room for severity
    print("\nScored Issues Summary:")
    print("=" * width)
    print(f"{'Rank':<6} {'Score':<6} {'Severity':<10} {'Type':<15} {'Title/Message':<{title_width}}")
    print("-" * width)
    for i, issue in enumerate(issues, 1):
        title = issue.get('message', '') or issue.get('title', '')
        title = (title[:title_width - 6] + "...") if len(title) > title_width - 3 else title
        print(f"{i:<6} {issue['score']:<6} {issue.get('severityLevel', 'UNKNOWN'):<10} {issue.get('type', '').upper():<15} {title:<{title_width}}")
    print("=" * width)
    print(f"Total issues scored: {len(issues)}")
    print(f"Total runtime: {runtime:.2f} seconds")

def main():
    args = parse_arguments()
    start_time = time.time()

    if args.summary_only:
        try:
            with open(args.output, 'r') as f:
                issues = json.load(f).get('issues', [])
            print_summary(sorted(issues, key=lambda x: x['score'], reverse=True), time.time() - start_time)
            return
        except Exception as e:
            print(f"Error reading summary: {e}")
            return

    bedrock_client = boto3.client('bedrock-runtime')

    issues = load_issues(args.issues)
    print(f"Loaded {len(issues)} issues")

    prompt_template = load_prompt(args.prompt_file)
    print(f"Loaded prompt template from {args.prompt_file}")

    print(f"Scoring issues using model {args.model}...")
    scored = []
    for issue in tqdm(issues, desc="Scoring issues"):
        scored.append(score_issue(issue, bedrock_client, args.model, prompt_template))

    scored.sort(key=lambda x: x['score'], reverse=True)

    print(f"Saving to {args.output}...")
    save_issues(scored, args.output)

    runtime = time.time() - start_time
    print_summary(scored, runtime)

if __name__ == "__main__":
    main()
