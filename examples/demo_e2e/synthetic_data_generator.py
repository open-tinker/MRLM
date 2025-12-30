"""
Synthetic Data Generator for MRLM Demo

This script generates synthetic datasets for demonstrating MRLM capabilities:
- Math problems with solutions
- Code problems with test cases
- Debate topics with arguments
- Tool usage scenarios

Usage:
    python synthetic_data_generator.py
"""

import json
import random
from pathlib import Path
from typing import Dict, List


def generate_math_problems(num_problems: int = 20) -> List[Dict]:
    """Generate synthetic math problems."""
    problems = []
    problem_templates = [
        # Addition
        {
            "template": "{name} has {n1} {item}. {pronoun} gets {n2} more {item}. How many {item} does {name} have now?",
            "operation": lambda n1, n2: n1 + n2,
            "difficulty": "easy",
        },
        # Subtraction
        {
            "template": "{name} has {n1} {item}. {pronoun} gives away {n2} {item}. How many {item} does {name} have left?",
            "operation": lambda n1, n2: n1 - n2,
            "difficulty": "easy",
        },
        # Multiplication
        {
            "template": "A box contains {n1} rows of {item} with {n2} {item} in each row. How many {item} are in the box?",
            "operation": lambda n1, n2: n1 * n2,
            "difficulty": "medium",
        },
        # Multi-step (buy and change)
        {
            "template": "{name} has ${n1}. {pronoun} buys {n2} books for ${n3} each. How much money does {name} have left?",
            "operation": lambda n1, n2, n3: n1 - (n2 * n3),
            "difficulty": "medium",
            "num_vars": 3,
        },
    ]

    names = ["Alice", "Bob", "Charlie", "Diana", "Emma", "Frank"]
    items = ["apples", "pencils", "candies", "marbles", "stickers", "cookies"]
    pronouns = {"Alice": "She", "Bob": "He", "Charlie": "He", "Diana": "She", "Emma": "She", "Frank": "He"}

    for i in range(num_problems):
        template_data = random.choice(problem_templates)
        template = template_data["template"]
        operation = template_data["operation"]
        num_vars = template_data.get("num_vars", 2)

        name = random.choice(names)
        item = random.choice(items)
        pronoun = pronouns[name]

        if num_vars == 2:
            n1 = random.randint(10, 50)
            n2 = random.randint(1, min(20, n1))
            question = template.format(name=name, item=item, pronoun=pronoun, n1=n1, n2=n2)
            answer = operation(n1, n2)
        else:  # num_vars == 3
            n1 = random.randint(30, 100)
            n2 = random.randint(2, 5)
            n3 = random.randint(3, 15)
            question = template.format(name=name, item=item, pronoun=pronoun, n1=n1, n2=n2, n3=n3)
            answer = operation(n1, n2, n3)

        problems.append({
            "id": f"math_{i+1:03d}",
            "question": question,
            "answer": int(answer) if answer == int(answer) else answer,
            "difficulty": template_data["difficulty"],
        })

    return problems


def generate_code_problems(num_problems: int = 15) -> List[Dict]:
    """Generate synthetic code problems."""
    problems = []

    # Problem templates
    templates = [
        {
            "name": "sum_list",
            "description": "Write a function that returns the sum of all numbers in a list.",
            "function_name": "sum_list",
            "test_cases": [
                {"input": "[1, 2, 3, 4, 5]", "output": "15"},
                {"input": "[10, 20, 30]", "output": "60"},
                {"input": "[]", "output": "0"},
            ],
        },
        {
            "name": "is_even",
            "description": "Write a function that returns True if a number is even, False otherwise.",
            "function_name": "is_even",
            "test_cases": [
                {"input": "4", "output": "True"},
                {"input": "7", "output": "False"},
                {"input": "0", "output": "True"},
            ],
        },
        {
            "name": "reverse_string",
            "description": "Write a function that reverses a string.",
            "function_name": "reverse_string",
            "test_cases": [
                {"input": "'hello'", "output": "'olleh'"},
                {"input": "'python'", "output": "'nohtyp'"},
                {"input": "'a'", "output": "'a'"},
            ],
        },
        {
            "name": "max_in_list",
            "description": "Write a function that returns the maximum number in a list.",
            "function_name": "max_in_list",
            "test_cases": [
                {"input": "[1, 5, 3, 9, 2]", "output": "9"},
                {"input": "[-1, -5, -3]", "output": "-1"},
                {"input": "[42]", "output": "42"},
            ],
        },
        {
            "name": "count_vowels",
            "description": "Write a function that counts the number of vowels (a,e,i,o,u) in a string.",
            "function_name": "count_vowels",
            "test_cases": [
                {"input": "'hello'", "output": "2"},
                {"input": "'python'", "output": "1"},
                {"input": "'aeiou'", "output": "5"},
            ],
        },
    ]

    for i, template in enumerate(templates):
        problems.append({
            "id": f"code_{i+1:03d}",
            "description": template["description"],
            "function_name": template["function_name"],
            "test_cases": template["test_cases"],
            "difficulty": "easy",
        })

    # Add more varied problems
    additional = [
        {
            "id": "code_006",
            "description": "Write a function that returns the factorial of a number n.",
            "function_name": "factorial",
            "test_cases": [
                {"input": "5", "output": "120"},
                {"input": "3", "output": "6"},
                {"input": "0", "output": "1"},
            ],
            "difficulty": "medium",
        },
        {
            "id": "code_007",
            "description": "Write a function that checks if a string is a palindrome.",
            "function_name": "is_palindrome",
            "test_cases": [
                {"input": "'racecar'", "output": "True"},
                {"input": "'hello'", "output": "False"},
                {"input": "'a'", "output": "True"},
            ],
            "difficulty": "medium",
        },
    ]

    problems.extend(additional)
    return problems[:num_problems]


def generate_debate_topics(num_topics: int = 10) -> List[Dict]:
    """Generate synthetic debate topics."""
    topics = [
        {
            "topic": "Remote work should be the default for all office jobs",
            "context": "With advances in technology and changing work culture, remote work has become increasingly viable.",
        },
        {
            "topic": "Social media does more harm than good to society",
            "context": "Social media platforms have transformed communication but also raised concerns about mental health and misinformation.",
        },
        {
            "topic": "Electric vehicles should replace all gas-powered cars by 2035",
            "context": "Climate change concerns have led to calls for transitioning away from fossil fuel vehicles.",
        },
        {
            "topic": "AI should be regulated like pharmaceuticals",
            "context": "The rapid advancement of AI technology has raised questions about safety and oversight.",
        },
        {
            "topic": "Universal basic income should be implemented",
            "context": "As automation increases, some propose guaranteed income for all citizens.",
        },
        {
            "topic": "Space exploration funding should be redirected to Earth problems",
            "context": "Billions are spent on space programs while many issues remain on Earth.",
        },
        {
            "topic": "Online education is as effective as traditional classroom learning",
            "context": "The COVID-19 pandemic accelerated the adoption of online learning platforms.",
        },
        {
            "topic": "Privacy is more important than security",
            "context": "Government surveillance and data collection raise concerns about individual privacy.",
        },
    ]

    debate_data = []
    for i, topic_data in enumerate(topics[:num_topics]):
        debate_data.append({
            "id": f"debate_{i+1:03d}",
            "topic": topic_data["topic"],
            "context": topic_data["context"],
            "difficulty": "medium",
        })

    return debate_data


def generate_tool_scenarios(num_scenarios: int = 8) -> List[Dict]:
    """Generate synthetic tool usage scenarios."""
    scenarios = [
        {
            "task": "Calculate the compound interest on $1000 at 5% annual rate for 3 years",
            "required_tools": ["calculator"],
            "difficulty": "easy",
        },
        {
            "task": "Search for the latest Python release version and calculate how many days since it was released",
            "required_tools": ["web_search", "calculator"],
            "difficulty": "medium",
        },
        {
            "task": "Create a file called 'test.txt' with the content 'Hello World'",
            "required_tools": ["file_system"],
            "difficulty": "easy",
        },
        {
            "task": "Use Python to generate a random number between 1 and 100",
            "required_tools": ["python_repl"],
            "difficulty": "easy",
        },
        {
            "task": "Calculate the sum of squares from 1 to 10 using Python",
            "required_tools": ["python_repl"],
            "difficulty": "medium",
        },
    ]

    tool_data = []
    for i, scenario in enumerate(scenarios[:num_scenarios]):
        tool_data.append({
            "id": f"tool_{i+1:03d}",
            "task": scenario["task"],
            "required_tools": scenario["required_tools"],
            "difficulty": scenario["difficulty"],
        })

    return tool_data


def main():
    """Generate all synthetic datasets."""
    print("=" * 70)
    print("MRLM: Synthetic Data Generator")
    print("=" * 70)

    # Create output directory
    output_dir = Path(__file__).parent / "data"
    output_dir.mkdir(exist_ok=True)

    print(f"\nOutput directory: {output_dir}")

    # Generate math problems
    print("\n[1/4] Generating math problems...")
    math_problems = generate_math_problems(num_problems=20)
    math_file = output_dir / "math_problems.json"
    with open(math_file, "w") as f:
        json.dump(math_problems, f, indent=2)
    print(f"   ✓ Generated {len(math_problems)} math problems -> {math_file}")

    # Generate code problems
    print("\n[2/4] Generating code problems...")
    code_problems = generate_code_problems(num_problems=15)
    code_file = output_dir / "code_problems.json"
    with open(code_file, "w") as f:
        json.dump(code_problems, f, indent=2)
    print(f"   ✓ Generated {len(code_problems)} code problems -> {code_file}")

    # Generate debate topics
    print("\n[3/4] Generating debate topics...")
    debate_topics = generate_debate_topics(num_topics=10)
    debate_file = output_dir / "debate_topics.json"
    with open(debate_file, "w") as f:
        json.dump(debate_topics, f, indent=2)
    print(f"   ✓ Generated {len(debate_topics)} debate topics -> {debate_file}")

    # Generate tool scenarios
    print("\n[4/4] Generating tool usage scenarios...")
    tool_scenarios = generate_tool_scenarios(num_scenarios=8)
    tool_file = output_dir / "tool_scenarios.json"
    with open(tool_file, "w") as f:
        json.dump(tool_scenarios, f, indent=2)
    print(f"   ✓ Generated {len(tool_scenarios)} tool scenarios -> {tool_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Data Generation Complete!")
    print("=" * 70)
    print(f"\nGenerated datasets:")
    print(f"  - Math problems: {len(math_problems)}")
    print(f"  - Code problems: {len(code_problems)}")
    print(f"  - Debate topics: {len(debate_topics)}")
    print(f"  - Tool scenarios: {len(tool_scenarios)}")
    print(f"\nAll data saved to: {output_dir}")

    # Show sample data
    print("\n" + "-" * 70)
    print("Sample Math Problem:")
    print("-" * 70)
    print(json.dumps(math_problems[0], indent=2))

    print("\n" + "-" * 70)
    print("Sample Code Problem:")
    print("-" * 70)
    print(json.dumps(code_problems[0], indent=2))

    print("\n✓ Synthetic data generation completed successfully!")


if __name__ == "__main__":
    main()
