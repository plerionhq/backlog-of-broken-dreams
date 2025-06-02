#!/usr/bin/env python3

"""
LLM-Based Security Issue Prioritizer: Intelligent Triage Using AWS Bedrock
This script uses a large language model (LLM) via AWS Bedrock to prioritize a list
of security issues by comparing them in pairs and reasoning about which is more urgent.

It uses a basic bubble sort algorithm to order issues based on the results of these
LLM-driven pairwise comparisons. Bubble sort is intentionally chosen here for its
simplicity and determinism — it guarantees that the final ranking is entirely
derived from individual model judgments.

Pros of bubble sort in this context:
- Easy to implement and debug
- Deterministic: same input = same output
- Ensures transitive ordering based purely on model comparisons
- Makes it easy to store and explain the reasoning behind each swap

Cons:
- Poor performance at scale (O(n²) comparisons)
- Expensive if each comparison requires an LLM call
- Not suitable for very large input sets without optimization or batching

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
    --prompt-file    Path to the prompt template file (default: bubble_prompt.txt)

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
import base64
from tqdm import tqdm
import re
import time

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-powered security issue prioritization tool using AWS Bedrock"
    )
    parser.add_argument(
        "--issues",
        type=str,
        default="input-issues.json",
        help="JSON file containing security issues to prioritize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="prioritized-issues.json",
        help="Output file for prioritized issues (default: prioritized-issues.json)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic.claude-3-haiku-20240307-v1:0",
        help="AWS Bedrock model to use. Options:\n"
             "- anthropic.claude-3-haiku-20240307-v1:0 (fastest, cheapest)\n"
             "- anthropic.claude-3-sonnet-20240229-v1:0 (balanced)\n"
             "- anthropic.claude-3-opus-20240229-v1:0 (most capable)\n"
             "Default: anthropic.claude-3-haiku-20240307-v1:0"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print summary from existing output file without running prioritization"
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="bubble_prompt.txt",
        help="Path to the prompt template file (default: bubble_prompt.txt)"
    )
    return parser.parse_args()

def load_issues(file_path: str) -> List[Dict[str, Any]]:
    """Load security issues from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        return data.get('issues', [])

def save_issues(issues: List[Dict[str, Any]], file_path: str) -> None:
    """Save prioritized issues to a JSON file."""
    with open(file_path, 'w') as f:
        json.dump({'issues': issues}, f, indent=2)

def sanitize_json_string(s: str) -> str:
    """Remove control characters and sanitize JSON string."""
    # Remove control characters
    s = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', s)
    # Remove any non-printable characters
    s = ''.join(char for char in s if char.isprintable() or char.isspace())
    return s

def load_prompt(file_path: str) -> str:
    """Load prompt template from file."""
    with open(file_path, 'r') as f:
        return f.read()

def create_comparison_prompt(issue1: Dict[str, Any], issue2: Dict[str, Any], prompt_template: str) -> str:
    """Create a prompt for comparing two security issues."""
    return prompt_template.format(
        issue1=json.dumps(issue1, indent=2),
        issue2=json.dumps(issue2, indent=2)
    )

def compare_issues(issue1: Dict[str, Any], issue2: Dict[str, Any], bedrock_client, model_id: str, prompt_template: str) -> Tuple[bool, str]:
    """Compare two security issues using AWS Bedrock."""
    prompt = create_comparison_prompt(issue1, issue2, prompt_template)
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [{
                    "role": "user",
                    "content": prompt
                }]
            })
        )
        
        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']
        
        # Sanitize the response text
        sanitized_text = sanitize_json_string(response_text)
        
        try:
            result = json.loads(sanitized_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {str(e)}")
            print(f"Raw response: {response_text}")
            print(f"Sanitized response: {sanitized_text}")
            raise
        
        if not isinstance(result, dict) or 'higher_priority_issue' not in result or 'reasoning' not in result:
            raise ValueError("Invalid response format from model")
            
        if result['higher_priority_issue'] not in [1, 2]:
            raise ValueError(f"Invalid higher_priority_issue value: {result['higher_priority_issue']}")
        
        return result['higher_priority_issue'] == 1, result['reasoning']
        
    except Exception as e:
        print(f"Error comparing issues: {str(e)}")
        # Fallback to basic severity comparison if Bedrock fails
        severity_order = {'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}
        sev1 = severity_order.get(issue1.get('severityLevel', 'LOW'), 0)
        sev2 = severity_order.get(issue2.get('severityLevel', 'LOW'), 0)
        return sev1 >= sev2, f"Fallback comparison based on severity levels: {issue1.get('severityLevel')} vs {issue2.get('severityLevel')}"

def bubble_sort_issues(issues: List[Dict[str, Any]], bedrock_client, model_id: str, prompt_template: str) -> List[Dict[str, Any]]:
    """Sort security issues using bubble sort based on AI pairwise comparisons."""
    n = len(issues)
    total_comparisons = (n * (n - 1)) // 2  # Total number of comparisons in bubble sort
    
    with tqdm(total=total_comparisons, desc="Comparing issues") as pbar:
        for i in range(n):
            for j in range(0, n - i - 1):
                # Get issue identifiers for progress display
                issue1_id = issues[j].get('id', issues[j].get('vulnerabilityId', f'Issue {j}'))
                issue2_id = issues[j + 1].get('id', issues[j + 1].get('vulnerabilityId', f'Issue {j + 1}'))
                
                # Update progress description
                pbar.set_description(f"Comparing {issue1_id} vs {issue2_id}")
                
                is_higher, reasoning = compare_issues(issues[j], issues[j + 1], bedrock_client, model_id, prompt_template)
                
                # Store the comparison reasoning
                if 'comparison_reasoning' not in issues[j]:
                    issues[j]['comparison_reasoning'] = []
                if 'comparison_reasoning' not in issues[j + 1]:
                    issues[j + 1]['comparison_reasoning'] = []
                
                # Add the reasoning to both issues
                comparison = {
                    'compared_with': issues[j + 1].get('id', issues[j + 1].get('vulnerabilityId', str(j + 1))),
                    'reasoning': reasoning,
                    'was_higher_priority': is_higher
                }
                issues[j]['comparison_reasoning'].append(comparison)
                
                comparison = {
                    'compared_with': issues[j].get('id', issues[j].get('vulnerabilityId', str(j))),
                    'reasoning': reasoning,
                    'was_higher_priority': not is_higher
                }
                issues[j + 1]['comparison_reasoning'].append(comparison)
                
                # Swap if needed
                if not is_higher:
                    issues[j], issues[j + 1] = issues[j + 1], issues[j]
                
                pbar.update(1)
    
    return issues

def print_prioritization_summary(issues: List[Dict[str, Any]], runtime: float) -> None:
    """Print a summary of the prioritized issues."""
    # Fixed width of 140 characters
    terminal_width = 140
    
    # Calculate title width (leave space for other columns and padding)
    title_width = terminal_width - 30  # 6 for rank + 10 for severity + 15 for type + spacing
    
    print("\nBubble Sort Issues Summary:")
    print("=" * terminal_width)
    print(f"{'Rank':<6} {'Severity':<10} {'Type':<15} {'Title/Message':<{title_width}}")
    print("-" * terminal_width)
    
    for i, issue in enumerate(issues, 1):
        # Get the title/message
        title = issue.get('message', '') or issue.get('title', '')
        if len(title) > title_width - 3:
            title = title[:title_width - 6] + "..."
            
        # Get the type
        issue_type = issue.get('type', '').upper()
        
        # Get the severity
        severity = issue.get('severityLevel', 'UNKNOWN')
        
        print(f"{i:<6} {severity:<10} {issue_type:<15} {title:<{title_width}}")
    
    print("=" * terminal_width)
    print(f"Total issues prioritized: {len(issues)}")
    print(f"Total runtime: {runtime:.2f} seconds")

def main():
    args = parse_arguments()
    start_time = time.time()
    
    if args.summary_only:
        try:
            print(f"Loading prioritized issues from {args.output}...")
            with open(args.output, 'r') as f:
                data = json.load(f)
                issues = data.get('issues', [])
            print(f"Loaded {len(issues)} issues")
            print_prioritization_summary(issues, time.time() - start_time)
            return
        except FileNotFoundError:
            print(f"Error: Output file {args.output} not found. Run prioritization first.")
            return
        except json.JSONDecodeError:
            print(f"Error: {args.output} is not a valid JSON file.")
            return
    
    # Initialize AWS Bedrock client
    bedrock_client = boto3.client('bedrock-runtime')
    
    # Load issues
    issues = load_issues(args.issues)
    print(f"Loaded {len(issues)} issues")

    # Load prompt template
    prompt_template = load_prompt(args.prompt_file)
    print(f"Loaded prompt template from {args.prompt_file}")
    
    # Sort issues
    print(f"Starting prioritization using {args.model}...")
    prioritized_issues = bubble_sort_issues(issues, bedrock_client, args.model, prompt_template)
    
    # Save results
    print(f"Saving prioritized issues to {args.output}...")
    save_issues(prioritized_issues, args.output)
    print(f"Prioritized issues saved to {args.output}")
    
    # Print summary
    runtime = time.time() - start_time
    print_prioritization_summary(prioritized_issues, runtime)

if __name__ == "__main__":
    main() 