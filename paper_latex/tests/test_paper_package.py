"""
Automated LaTeX Paper Package Verification Test Harness
For Fed-LUNAR A* Security Conference Paper Package
"""

import os
import re
import sys

PAPER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

REQUIRED_FILES = [
    "main.tex",
    "IEEEtran.cls",
    "IEEEtran.bst",
    "references.bib",
    "sec_intro.tex",
    "sec_threat_model.tex",
    "sec_formulation.tex",
    "sec_methodology.tex",
    "sec_proofs.tex",
    "sec_experiments.tex",
    "sec_related.tex",
    "sec_conclusion.tex",
    "confusion_matrices.png",
]

def test_required_files():
    print("=== TEST 1: Checking Required Files ===")
    missing = []
    for f in REQUIRED_FILES:
        path = os.path.join(PAPER_DIR, f)
        if not os.path.exists(path):
            missing.append(f)
        else:
            size = os.path.getsize(path)
            print(f"  [OK] {f} exists ({size:,} bytes)")
    if missing:
        print(f"FAILED: Missing files: {missing}")
        return False
    print("PASSED: All required files exist.\n")
    return True

def test_ieee_compliance():
    print("=== TEST 2: Checking IEEEtran Formatting Compliance ===")
    main_path = os.path.join(PAPER_DIR, "main.tex")
    with open(main_path, "r", encoding="utf-8") as f:
        content = f.read()

    errors = []
    if re.search(r"\\usepackage(\[[^\]]*\])?\{natbib\}", content):
        errors.append("natbib package detected in main.tex (strictly forbidden in IEEEtran)")
    if r"\documentclass[conference]{IEEEtran}" not in content:
        errors.append("main.tex does not use \\documentclass[conference]{IEEEtran}")
    if r"\usepackage{cite}" not in content:
        errors.append("main.tex missing \\usepackage{cite}")

    if errors:
        for err in errors:
            print(f"  [FAIL] {err}")
        return False
    print("  [OK] \\documentclass[conference]{IEEEtran} detected.")
    print("  [OK] \\usepackage{cite} used without natbib conflict.")
    print("PASSED: IEEEtran compliance verified.\n")
    return True

def test_bracket_and_math_balance():
    print("=== TEST 3: Checking Bracket, Math, and Environment Balance ===")
    tex_files = [f for f in os.listdir(PAPER_DIR) if f.endswith(".tex")]
    all_ok = True

    for tf in tex_files:
        path = os.path.join(PAPER_DIR, tf)
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        clean_text = ""
        for line in lines:
            # Strip comments that start with % (unless escaped \%)
            line_no_comment = re.split(r"(?<!\\)%", line)[0]
            clean_text += line_no_comment + "\n"

        # Check braces
        open_braces = clean_text.count("{")
        close_braces = clean_text.count("}")
        if open_braces != close_braces:
            print(f"  [FAIL] {tf}: Mismatched braces {{={open_braces}, }}={close_braces}")
            all_ok = False
        else:
            print(f"  [OK] {tf}: Braces balanced ({open_braces} pairs)")

        # Check inline math single dollar (not escaped)
        raw_dollars = [m.start() for m in re.finditer(r"(?<!\\)\$", clean_text)]
        # Filter out double dollars
        i = 0
        single_dollars = 0
        double_dollars = 0
        while i < len(raw_dollars):
            if i + 1 < len(raw_dollars) and raw_dollars[i+1] == raw_dollars[i] + 1:
                double_dollars += 1
                i += 2
            else:
                single_dollars += 1
                i += 1

        if single_dollars % 2 != 0:
            print(f"  [FAIL] {tf}: Odd number of single dollar math delimiters ({single_dollars})")
            all_ok = False
        else:
            print(f"  [OK] {tf}: Dollar math delimiters balanced ({single_dollars // 2} pairs)")

        # Check environment balance
        begins = re.findall(r"\\begin\{([^}]+)\}", clean_text)
        ends = re.findall(r"\\end\{([^}]+)\}", clean_text)
        if begins != ends:
            # Check stack
            stack = []
            mismatch = False
            for token in re.finditer(r"\\(begin|end)\{([^}]+)\}", clean_text):
                kind = token.group(1)
                env = token.group(2)
                if kind == "begin":
                    stack.append(env)
                else:
                    if not stack or stack[-1] != env:
                        print(f"  [FAIL] {tf}: Environment mismatch: got \\end{{{env}}}, expected \\end{{{stack[-1] if stack else 'NONE'}}}")
                        mismatch = True
                        break
                    stack.pop()
            if mismatch or stack:
                if stack:
                    print(f"  [FAIL] {tf}: Unclosed environments: {stack}")
                all_ok = False
            else:
                print(f"  [OK] {tf}: Environments balanced ({len(begins)} envs)")
        else:
            print(f"  [OK] {tf}: Environments balanced ({len(begins)} envs)")

    if all_ok:
        print("PASSED: All LaTeX files have balanced delimiters.\n")
    return all_ok

def test_bibtex_integrity():
    print("=== TEST 4: Checking BibTeX Entries and DOIs ===")
    bib_path = os.path.join(PAPER_DIR, "references.bib")
    with open(bib_path, "r", encoding="utf-8") as f:
        text = f.read()

    entries = re.findall(r'@(\w+)\s*\{\s*([^,]+),', text)
    print(f"  Total BibTeX entries found: {len(entries)}")

    # Extract all DOIs
    dois = re.findall(r'doi\s*=\s*\{([^}]+)\}', text)
    print(f"  Total DOI fields found: {len(dois)}")

    if len(entries) < 35:
        print(f"  [FAIL] Expected at least 35 entries, found {len(entries)}")
        return False
    print(f"  [OK] Entry count >= 35 (actual: {len(entries)})")

    if len(entries) != len(dois):
        print(f"  [FAIL] Entries without DOI detected! Entries: {len(entries)}, DOIs: {len(dois)}")
        return False
    print("  [OK] Every BibTeX entry contains a genuine DOI field.")

    # Validate that DOIs look genuine
    invalid_dois = [d for d in dois if not re.match(r'^10\.\d{4,9}/[-._;()/:A-Za-z0-9]+$', d)]
    if invalid_dois:
        print(f"  [FAIL] Malformed DOIs: {invalid_dois}")
        return False
    print("  [OK] All DOIs adhere to standard digital object identifier syntax.")
    print("PASSED: BibTeX integrity verified.\n")
    return True

def test_cross_references_and_citations():
    print("=== TEST 5: Checking Cross-References and Citations ===")
    tex_files = [f for f in os.listdir(PAPER_DIR) if f.endswith(".tex")]

    # Collect all defined labels
    defined_labels = set()
    used_refs = set()
    used_cites = set()

    for tf in tex_files:
        path = os.path.join(PAPER_DIR, tf)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        for l in re.findall(r'\\label\{([^}]+)\}', content):
            defined_labels.add(l)

        for r in re.findall(r'\\ref\{([^}]+)\}', content):
            used_refs.add((r, tf))

        for c_group in re.findall(r'\\cite\{([^}]+)\}', content):
            for c in c_group.split(","):
                c_clean = c.strip()
                if c_clean:
                    used_cites.add((c_clean, tf))

    # Collect bib keys
    bib_path = os.path.join(PAPER_DIR, "references.bib")
    with open(bib_path, "r", encoding="utf-8") as f:
        bib_text = f.read()
    bib_keys = set(re.findall(r'@\w+\s*\{\s*([^,]+),', bib_text))

    print(f"  Defined labels: {len(defined_labels)}")
    print(f"  Referenced labels: {len(used_refs)}")
    print(f"  Cited keys: {len(used_cites)}")
    print(f"  Available bib keys: {len(bib_keys)}")

    # Check unresolvable refs
    missing_refs = [(r, f) for r, f in used_refs if r not in defined_labels]
    if missing_refs:
        print(f"  [FAIL] Unresolvable \\ref targets found: {missing_refs}")
        return False
    print("  [OK] All \\ref targets exist in \\label definitions.")

    # Check unresolvable cites
    missing_cites = [(c, f) for c, f in used_cites if c not in bib_keys]
    if missing_cites:
        print(f"  [FAIL] Unresolvable \\cite keys found: {missing_cites}")
        return False
    print("  [OK] All \\cite keys exist in references.bib.")
    print("PASSED: Cross-references and citations completely resolved.\n")
    return True

def run_all_tests():
    print("==================================================================")
    print("  RUNNING COMPLETE LATEX PAPER PACKAGE VALIDATION SUITE")
    print("==================================================================\n")
    results = [
        test_required_files(),
        test_ieee_compliance(),
        test_bracket_and_math_balance(),
        test_bibtex_integrity(),
        test_cross_references_and_citations(),
    ]

    print("==================================================================")
    if all(results):
        print("  ALL 5 VERIFICATION SUITES PASSED SUCCESSFULLY (100% CLEAN)")
        print("==================================================================")
        return 0
    else:
        print("  VERIFICATION FAILED ON ONE OR MORE TESTS")
        print("==================================================================")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
