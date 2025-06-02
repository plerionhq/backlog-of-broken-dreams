#!/usr/bin/env python3

"""
LLM-Based Security Issue Prioritizer: Intelligent Triage Using AWS Bedrock
This script uses a large language model (LLM) via AWS Bedrock to prioritize a list
of security issues by comparing them in pairs and reasoning about which is more urgent.

Instead of relying on inconsistent severity ratings from scanners, it performs
pairwise comparisons using natural language reasoning to determine which issues
deserve more immediate attention. The results are sorted and annotated with
rationale for each comparison.

Elo scoring is used to determine relative ranking through repeated pairwise 
comparisons. It has the advantage of being more scalable and statistically grounded
compared to bubble sort.

Pros of Elo scoring in this context:
- More efficient than bubble sort (fewer comparisons for large n)
- Can gracefully handle partial comparison sets
- Maintains ranking stability as more comparisons are added
- Still deterministic with fixed seeds and models

Cons:
- Order can be affected by comparison sequence
- May need tuning (K-factor) for best behavior
- Harder to explain intuitively than bubble sort

Features:
- Supports multiple Anthropic Claude models via AWS Bedrock (Haiku, Sonnet, Opus)
- Uses natural language prompts to drive prioritization logic
- Annotates each issue with reasoning and comparison metadata
- CLI options for reviewing prior outputs without rerunning the model

Usage:
    ./prioritize.py [--issues input.json] [--output prioritized.json] [--model <model_id>]

Options:
    --issues         Path to input file containing unprioritized issues (default: input-issues.json)
    --output         Output file for sorted and annotated issues (default: prioritized-issues.json)
    --model          AWS Bedrock model ID (e.g., Claude Haiku, Sonnet, or Opus)
    --summary-only   Show prioritization summary from existing output file
    --max-comparisons Maximum number of comparisons as a multiple of total pairs (default: 1.0 = all pairs)
    --prompt-file    Path to the prompt template file (default: elo_prompt.txt)

Output:
    - `prioritized-issues.json`: Sorted issues with comparison metadata and reasoning
    - Terminal summary: Ranked list with severity and truncated title/message

Written by: Daniel Grzelak (@dagrz on X, daniel.grzelak@plerion.com)  
For more tools and security automation, visit: https://www.plerion.com
"""

import argparse
import json
import boto3
from typing import List, Dict, Any, Tuple
import re
from tqdm import tqdm
import time

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered security issue prioritization tool using AWS Bedrock"
    )
    parser.add_argument("--issues", type=str, default="input-issues.json")
    parser.add_argument("--output", type=str, default="prioritized-issues.json")
    parser.add_argument("--model", type=str, default="anthropic.claude-3-haiku-20240307-v1:0")
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--max-comparisons", type=float, default=1.0,
                      help="Maximum number of comparisons as a multiple of total pairs (default: 1.0 = all pairs)")
    parser.add_argument("--prompt-file", type=str, default="elo_prompt.txt",
                      help="Path to the prompt template file (default: elo_prompt.txt)")
    return parser.parse_args()

def load_issues(file_path: str) -> List[Dict[str, Any]]:
    """Load security issues from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f).get('issues', [])

def save_issues(issues: List[Dict[str, Any]], file_path: str) -> None:
    """Save prioritized issues to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump({'issues': issues}, f, indent=2)

def sanitize_json_string(s: str) -> str:
    """Remove control characters and sanitize JSON string."""
    s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)
    return ''.join(c for c in s if c.isprintable() or c.isspace())

def load_prompt(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, 'r') as f:
        return f.read()

def create_comparison_prompt(issue1: Dict[str, Any], issue2: Dict[str, Any], prompt_template: str) -> str:
    """Generate the LLM prompt for comparing two issues."""
    return prompt_template.format(
        issue1=json.dumps(issue1, indent=2),
        issue2=json.dumps(issue2, indent=2)
    )

def compare_issues(issue1: Dict[str, Any], issue2: Dict[str, Any], bedrock_client, model_id: str, prompt_template: str) -> Tuple[bool, str]:
    """Compare two issues via LLM and return whether issue1 is higher priority and the reasoning."""
    prompt = create_comparison_prompt(issue1, issue2, prompt_template)
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
        if result['higher_priority_issue'] not in [1, 2]:
            raise ValueError("Invalid `higher_priority_issue` value")
        return result['higher_priority_issue'] == 1, result['reasoning']
    except Exception as e:
        print(f"Error during comparison: {e}")
        sev_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        sev1 = sev_order.get(issue1.get('severityLevel', 'LOW'), 0)
        sev2 = sev_order.get(issue2.get('severityLevel', 'LOW'), 0)
        return sev1 >= sev2, f"Fallback comparison based on severity: {issue1.get('severityLevel')} vs {issue2.get('severityLevel')}"

def elo_rank_issues(issues: List[Dict[str, Any]], bedrock_client, model_id: str, prompt_template: str, max_comparisons_multiplier: float = 1.0) -> List[Dict[str, Any]]:
    """Rank issues using Elo scoring and pairwise LLM comparisons."""
    K = 32
    for issue in issues:
        issue['elo'] = 1200
        issue['comparison_reasoning'] = []

    # Calculate total possible pairs and max comparisons
    total_pairs = len(issues) * (len(issues) - 1) // 2
    max_comparisons = int(total_pairs * max_comparisons_multiplier)
    
    # Generate all possible pairs
    pairs = [(i, j) for i in range(len(issues)) for j in range(i + 1, len(issues))]
    
    # If we're limiting comparisons, randomly sample the pairs
    if max_comparisons < total_pairs:
        import random
        random.seed(42)  # For reproducibility
        pairs = random.sample(pairs, max_comparisons)

    with tqdm(total=len(pairs), desc="Ranking with Elo") as pbar:
        for i, j in pairs:
            issue1, issue2 = issues[i], issues[j]
            id1 = issue1.get('id', f'Issue {i}')
            id2 = issue2.get('id', f'Issue {j}')
            pbar.set_description(f"Comparing {id1} vs {id2}")

            r1, r2 = issue1['elo'], issue2['elo']
            expected1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
            expected2 = 1 - expected1

            is_higher, reasoning = compare_issues(issue1, issue2, bedrock_client, model_id, prompt_template)
            score1, score2 = (1, 0) if is_higher else (0, 1)

            issue1['elo'] += K * (score1 - expected1)
            issue2['elo'] += K * (score2 - expected2)

            issue1['comparison_reasoning'].append({
                'compared_with': id2,
                'reasoning': reasoning,
                'was_higher_priority': is_higher
            })
            issue2['comparison_reasoning'].append({
                'compared_with': id1,
                'reasoning': reasoning,
                'was_higher_priority': not is_higher
            })

            pbar.update(1)

    return sorted(issues, key=lambda x: x['elo'], reverse=True)

def print_prioritization_summary(issues: List[Dict[str, Any]], runtime: float) -> None:
    """Print the ranked summary of prioritized issues."""
    width = 140
    title_width = width - 45  # Reduced to make room for Elo score
    print("\nElo Issues Summary:")
    print("=" * width)
    print(f"{'Rank':<6} {'Elo':<8} {'Severity':<10} {'Type':<15} {'Title/Message':<{title_width}}")
    print("-" * width)
    for i, issue in enumerate(issues, 1):
        title = issue.get('message', '') or issue.get('title', '')
        title = (title[:title_width - 6] + "...") if len(title) > title_width - 3 else title
        print(f"{i:<6} {issue['elo']:<8.0f} {issue.get('severityLevel', 'UNKNOWN'):<10} {issue.get('type', '').upper():<15} {title:<{title_width}}")
    print("=" * width)
    print(f"Total issues prioritized: {len(issues)}")
    print(f"Total runtime: {runtime:.2f} seconds")

def main():
    args = parse_arguments()
    start_time = time.time()

    if args.summary_only:
        try:
            with open(args.output, 'r') as f:
                issues = json.load(f).get('issues', [])
            print_prioritization_summary(issues, time.time() - start_time)
            return
        except Exception as e:
            print(f"Error reading summary: {e}")
            return

    bedrock_client = boto3.client('bedrock-runtime')

    issues = load_issues(args.issues)
    print(f"Loaded {len(issues)} issues")

    prompt_template = load_prompt(args.prompt_file)
    print(f"Loaded prompt template from {args.prompt_file}")

    print(f"Prioritizing with model {args.model} using Elo scoring...")
    prioritized_issues = elo_rank_issues(issues, bedrock_client, args.model, prompt_template, args.max_comparisons)

    print(f"Saving to {args.output}...")
    save_issues(prioritized_issues, args.output)

    runtime = time.time() - start_time
    print_prioritization_summary(prioritized_issues, runtime)

if __name__ == "__main__":
    main()
