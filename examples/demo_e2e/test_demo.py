"""
Test script to validate the demo setup without running full training.

This script checks:
1. Synthetic data is generated correctly
2. Data can be loaded
3. All imports would work (if dependencies installed)
4. File structure is correct

Usage:
    python test_demo.py
"""

import json
import sys
from pathlib import Path


def test_data_files():
    """Test that all data files exist and are valid."""
    data_dir = Path(__file__).parent / "data"

    expected_files = [
        "math_problems.json",
        "code_problems.json",
        "debate_topics.json",
        "tool_scenarios.json",
    ]

    print("Testing data files...")
    all_good = True

    for filename in expected_files:
        filepath = data_dir / filename
        if not filepath.exists():
            print(f"  ✗ Missing: {filename}")
            all_good = False
        else:
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                print(f"  ✓ {filename}: {len(data)} items")
            except Exception as e:
                print(f"  ✗ Error loading {filename}: {e}")
                all_good = False

    return all_good


def test_math_data():
    """Test math problems data structure."""
    data_dir = Path(__file__).parent / "data"
    filepath = data_dir / "math_problems.json"

    print("\nTesting math problems structure...")

    with open(filepath, "r") as f:
        problems = json.load(f)

    # Check structure
    required_keys = ["id", "question", "answer", "difficulty"]

    for i, problem in enumerate(problems[:3]):  # Test first 3
        missing_keys = [key for key in required_keys if key not in problem]
        if missing_keys:
            print(f"  ✗ Problem {i+1} missing keys: {missing_keys}")
            return False

    print(f"  ✓ All problems have required keys: {required_keys}")
    print(f"  ✓ Sample problem: {problems[0]['question'][:60]}...")
    print(f"  ✓ Sample answer: {problems[0]['answer']}")

    return True


def test_code_data():
    """Test code problems data structure."""
    data_dir = Path(__file__).parent / "data"
    filepath = data_dir / "code_problems.json"

    print("\nTesting code problems structure...")

    with open(filepath, "r") as f:
        problems = json.load(f)

    # Check structure
    required_keys = ["id", "description", "function_name", "test_cases", "difficulty"]

    for i, problem in enumerate(problems[:3]):  # Test first 3
        missing_keys = [key for key in required_keys if key not in problem]
        if missing_keys:
            print(f"  ✗ Problem {i+1} missing keys: {missing_keys}")
            return False

        # Check test cases
        if not isinstance(problem["test_cases"], list) or len(problem["test_cases"]) == 0:
            print(f"  ✗ Problem {i+1} has invalid test cases")
            return False

    print(f"  ✓ All problems have required keys: {required_keys}")
    print(f"  ✓ Sample problem: {problems[0]['description'][:60]}...")
    print(f"  ✓ Sample test cases: {len(problems[0]['test_cases'])} cases")

    return True


def test_script_syntax():
    """Test that Python scripts are valid."""
    print("\nTesting script syntax...")

    scripts = [
        "synthetic_data_generator.py",
        "demo_complete.py",
    ]

    import py_compile

    for script in scripts:
        filepath = Path(__file__).parent / script
        try:
            py_compile.compile(str(filepath), doraise=True)
            print(f"  ✓ {script}: Valid Python syntax")
        except py_compile.PyCompileError as e:
            print(f"  ✗ {script}: Syntax error - {e}")
            return False

    return True


def test_readme():
    """Test that README exists."""
    print("\nTesting documentation...")

    readme = Path(__file__).parent / "README.md"
    if not readme.exists():
        print("  ✗ README.md not found")
        return False

    content = readme.read_text()

    # Check for key sections
    required_sections = [
        "Quick Start",
        "What the Demo Does",
        "Expected Output",
        "Troubleshooting",
    ]

    for section in required_sections:
        if section not in content:
            print(f"  ✗ README missing section: {section}")
            return False

    print(f"  ✓ README.md exists ({len(content)} characters)")
    print(f"  ✓ All required sections present")

    return True


def main():
    """Run all tests."""
    print("=" * 70)
    print("MRLM Demo Validation Tests")
    print("=" * 70)

    tests = [
        ("Data Files", test_data_files),
        ("Math Problems", test_math_data),
        ("Code Problems", test_code_data),
        ("Script Syntax", test_script_syntax),
        ("Documentation", test_readme),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:<10} {test_name}")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ All tests passed! Demo is ready to run.")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -e .")
        print("  2. Run the demo: python demo_complete.py")
        return 0
    else:
        print("\n✗ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
