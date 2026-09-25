"""
Adversarial Syntax Stress Test Suite for Fed-LUNAR LaTeX Paper Package
Independent checker script verifying:
1. Curly bracket balance, nesting depth >= 0, and depth == 0 at EOF
2. Math delimiter balance ($ and $$) and unclosed math modes
3. Unescaped special characters ('&', '_', '%', '#', '^') in text mode and accidental Markdown artifacts (e.g. '**text**')
4. LIFO environment nesting stack
5. Resolution and non-emptiness of all \\input{} statements in main.tex
"""

import os
import re
import sys

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def strip_comments(line):
    """Strip LaTeX comments (% not preceded by odd number of backslashes)."""
    parts = re.split(r'(?<!\\)%', line)
    return parts[0]

def check_brace_balance_and_depth(filename, text):
    r"""
    Verify curly brackets balance and depth >= 0 throughout file.
    Must ignore escaped braces: \{ and \}
    """
    errors = []
    depth = 0
    lines = text.splitlines()

    for line_num, line in enumerate(lines, 1):
        clean = strip_comments(line)
        i = 0
        while i < len(clean):
            char = clean[i]
            if char == '\\':
                i += 2
                continue
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth < 0:
                    errors.append(f"Line {line_num}, col {i+1}: Closing brace '}}' without matching '{{' (depth dropped to {depth})")
                    depth = 0
            i += 1

    if depth != 0:
        errors.append(f"End of file: Unmatched open braces (final depth = {depth})")

    return errors

def check_math_delimiters(filename, text):
    r"""
    Verify math delimiter balance ($ and $$) across the file.
    Must ignore escaped \$.
    """
    errors = []
    lines = text.splitlines()
    clean_lines = [strip_comments(l) for l in lines]
    full_clean = "\n".join(clean_lines)

    in_single_math = False
    in_double_math = False
    single_start = None
    double_start = None

    tokens = []
    for m in re.finditer(r'(?<!\\)(\$\$|\$)', full_clean):
        delim = m.group(1)
        tokens.append((delim, m.start()))

    for delim, idx in tokens:
        if delim == '$$':
            if in_single_math:
                errors.append(f"Found '$$' inside single '$' math mode at index {idx}")
            else:
                in_double_math = not in_double_math
                if in_double_math:
                    double_start = idx
        elif delim == '$':
            if in_double_math:
                errors.append(f"Found '$' inside double '$$' math mode at index {idx}")
            else:
                in_single_math = not in_single_math
                if in_single_math:
                    single_start = idx

    if in_single_math:
        errors.append(f"Unclosed single '$' math mode starting at character index {single_start}")
    if in_double_math:
        errors.append(f"Unclosed double '$$' math mode starting at character index {double_start}")

    return errors

def check_lifo_environments(filename, text):
    """
    Verify LIFO environment nesting stack: every \\begin{X} is closed by \\end{X} in LIFO order.
    """
    errors = []
    lines = text.splitlines()
    clean_lines = [strip_comments(l) for l in lines]
    full_clean = "\n".join(clean_lines)

    stack = []
    env_matches = list(re.finditer(r'\\(begin|end)\{([^}]+)\}', full_clean))

    for m in env_matches:
        action = m.group(1)
        env = m.group(2).strip()
        pos = m.start()

        if action == 'begin':
            stack.append((env, pos))
        elif action == 'end':
            if not stack:
                errors.append(f"\\end{{{env}}} encountered with empty stack at char {pos}")
            else:
                top_env, top_pos = stack.pop()
                if top_env != env:
                    errors.append(f"Mismatched environment: expected \\end{{{top_env}}} (opened at {top_pos}), got \\end{{{env}}} at {pos}")

    if stack:
        for env, pos in stack:
            errors.append(f"Unclosed environment \\begin{{{env}}} opened at char {pos}")

    return errors

def check_unescaped_special_chars_and_markdown(filename, text):
    """
    Check for:
    1. Unescaped percent signs with preceding digits (accidental comment cutting off text, e.g. '0.15%')
    2. Markdown bold/italic artifacts (e.g., '**text**')
    3. Unescaped ampersand '&' outside tabular/matrix/align/cases environments.
    4. Unescaped underscore outside math/label/cite/url/texttt.
    """
    issues = []
    lines = text.splitlines()

    for line_num, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('%'):
            continue

        clean = strip_comments(line)

        # 1. Check for unescaped percent sign preceded by digits or numbers
        m_percent = re.search(r'(\d+)\s*(?<!\\)%', line)
        if m_percent:
            issues.append({
                "type": "UNESCAPED_PERCENT",
                "severity": "CRITICAL",
                "line": line_num,
                "message": f"Potential accidental comment from unescaped % after digit: '{m_percent.group(0)}' in: {line.strip()[:80]}"
            })

        # 2. Check for markdown bold syntax: **word**
        m_md_bold = re.findall(r'\*\*([^*]+)\*\*', line)
        if m_md_bold:
            issues.append({
                "type": "MARKDOWN_SYNTAX_IN_LATEX",
                "severity": "CRITICAL",
                "line": line_num,
                "message": f"Markdown bold syntax '**...**' found in LaTeX source (must be \\textbf{{...}}): {m_md_bold}"
            })

        # 3. Check for unescaped & outside alignment environments
        # First, ignore escaped \&
        for m in re.finditer(r'(?<!\\)&', clean):
            # Check if this line is in an alignment environment
            issues.append({
                "type": "UNESCAPED_AMPERSAND",
                "severity": "CRITICAL",
                "line": line_num,
                "message": f"Unescaped ampersand '&' found in text mode at column {m.start()+1} in line: {line.strip()}"
            })

    # 4. Token-level check for unescaped underscore outside math mode
    clean_lines = [strip_comments(l) for l in lines]
    full_clean = "\n".join(clean_lines)

    def mask_pattern(pattern, text):
        return re.sub(pattern, lambda m: ' ' * len(m.group(0)), text, flags=re.DOTALL)

    masked = full_clean
    # Mask allowable macro arguments with optional brackets: [ ... ]
    masked = mask_pattern(r'\\(label|cite|ref|input|includegraphics|url|lstinline|usepackage|bibliographystyle|bibliography)(\[[^\]]*\])?\{[^}]*\}', masked)
    masked = mask_pattern(r'\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}.*?\\end\{\1\}', masked)
    masked = mask_pattern(r'\$\$.*?\$\$', masked)
    masked = mask_pattern(r'\$.*?\$', masked)

    for m in re.finditer(r'(?<!\\)_', masked):
        char_idx = m.start()
        line_no = full_clean[:char_idx].count('\n') + 1
        start = max(0, char_idx - 30)
        end = min(len(full_clean), char_idx + 30)
        snippet = full_clean[start:end].replace('\n', ' ')
        issues.append({
            "type": "UNESCAPED_UNDERSCORE",
            "severity": "CRITICAL",
            "line": line_no,
            "message": f"Unescaped underscore '_' outside math/label/cite/url at line {line_no}: '...{snippet}...'"
        })

    return issues

def check_section_stubs(main_path):
    """
    Verify every \\input{sec_...} in main.tex resolves to an existing, non-empty, syntactically valid .tex file.
    """
    errors = []
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()

    inputs = re.findall(r'\\input\{([^}]+)\}', content)

    results = []
    for inp in inputs:
        fname = inp if inp.endswith('.tex') else f"{inp}.tex"
        fpath = os.path.join(PAPER_DIR, fname)

        if not os.path.exists(fpath):
            errors.append(f"Missing input file: {fname} (path: {fpath})")
            results.append((fname, False, 0, ["File does not exist"]))
            continue

        size = os.path.getsize(fpath)
        if size == 0:
            errors.append(f"Empty input file: {fname}")
            results.append((fname, False, 0, ["File is empty (0 bytes)"]))
            continue

        with open(fpath, 'r', encoding='utf-8') as f:
            fcontent = f.read()

        file_errors = []
        file_errors.extend(check_brace_balance_and_depth(fname, fcontent))
        file_errors.extend(check_math_delimiters(fname, fcontent))
        file_errors.extend(check_lifo_environments(fname, fcontent))

        # Also check special characters inside section stub
        spec_issues = check_unescaped_special_chars_and_markdown(fname, fcontent)
        for issue in spec_issues:
            file_errors.append(f"Line {issue['line']}: [{issue['type']}] {issue['message']}")

        status = len(file_errors) == 0
        results.append((fname, status, size, file_errors))

    return results, errors

def run_adversarial_suite():
    print("=" * 75)
    print("  FED-LUNAR ADVERSARIAL SYNTAX & STUB VERIFICATION HARNESS")
    print("=" * 75)

    tex_files = sorted([f for f in os.listdir(PAPER_DIR) if f.endswith(".tex")])
    total_files = len(tex_files)
    print(f"Total .tex files to inspect: {total_files}\n")

    all_syntax_errors = {}
    all_special_issues = {}

    for tf in tex_files:
        path = os.path.join(PAPER_DIR, tf)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        file_errs = []
        file_errs.extend(check_brace_balance_and_depth(tf, content))
        file_errs.extend(check_math_delimiters(tf, content))
        file_errs.extend(check_lifo_environments(tf, content))

        special_issues = check_unescaped_special_chars_and_markdown(tf, content)

        if file_errs:
            all_syntax_errors[tf] = file_errs
        if special_issues:
            all_special_issues[tf] = special_issues

    print("--- SUITE 1: BRACKET DEPTH, MATH DELIMITERS, AND LIFO ENVIRONMENTS ---")
    if not all_syntax_errors:
        print("  [OK] All .tex files passed strict balance, depth, and LIFO nesting checks.")
    else:
        print("  [FAIL] Structural balance errors detected:")
        for tf, errs in all_syntax_errors.items():
            print(f"    File: {tf}")
            for e in errs:
                print(f"      - {e}")

    print("\n--- SUITE 2: UNESCAPED SPECIAL CHARACTERS & ACCIDENTAL MARKDOWN SYNTAX ---")
    if not all_special_issues:
        print("  [OK] Zero unescaped special characters or Markdown artifacts found.")
    else:
        print("  [FAIL] Unescaped special characters or Markdown artifacts detected:")
        for tf, items in all_special_issues.items():
            print(f"    File: {tf} ({len(items)} defects):")
            for item in items:
                print(f"      - Line {item['line']}: [{item['type']}] {item['message']}")

    print("\n--- SUITE 3: SECTION STUB RESOLUTION & COMPLIANCE ---")
    main_path = os.path.join(PAPER_DIR, "main.tex")
    stub_results, stub_errors = check_section_stubs(main_path)

    for fname, ok, size, errs in stub_results:
        status_str = "[OK]" if ok else "[FAIL]"
        print(f"  {status_str} {fname:<25} ({size:,} bytes) - Defects: {len(errs)}")
        for e in errs:
            print(f"      * {e}")

    print("\n" + "=" * 75)
    total_defects = sum(len(v) for v in all_syntax_errors.values()) + sum(len(v) for v in all_special_issues.values()) + len(stub_errors)
    print(f"  TOTAL DETECTED ADVERSARIAL DEFECTS: {total_defects}")
    print("=" * 75)

    return total_defects == 0

# Pytest integration
def test_adversarial_syntax_stress():
    passed = run_adversarial_suite()
    # In pytest we want true assertion
    assert passed, "Adversarial syntax stress test found defects in paper_latex!"

if __name__ == "__main__":
    if run_adversarial_suite():
        sys.exit(0)
    else:
        sys.exit(1)
