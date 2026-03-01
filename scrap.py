# CHANGES:
# - Added TODO_FILENAME constant for "todo.txt"
# - Added skip condition in the file loop to exclude todo.txt (case-insensitive match)

# scrape_files.py
# Recursively scans the script's directory for .py, .yaml, and .txt files,
# then compiles their contents into a single _compiled_files.txt output.


import os


EXTENSIONS = (".py", ".yaml", ".txt")
OUTPUT_FILENAME = "_compiled_files.txt"
TODO_FILENAME = "todo.txt"
SEPARATOR = "-----------"


def scrape_directory():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, OUTPUT_FILENAME)
    this_script = os.path.abspath(__file__)


    matched_files = []


    for root, dirs, files in os.walk(script_dir):
        # Skip hidden directories (e.g. .git, __pycache__)
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]


        for filename in sorted(files):
            if not filename.endswith(EXTENSIONS):
                continue
            filepath = os.path.abspath(os.path.join(root, filename))
            # Skip this script and the output file
            if filepath == this_script:
                continue
            if filepath == output_path:
                continue
            # Skip todo.txt
            if filename.lower() == TODO_FILENAME:
                continue
            matched_files.append(filepath)


    matched_files.sort()


    if not matched_files:
        print("No matching files found.")
        return


    with open(output_path, "w", encoding="utf-8") as outfile:
        for filepath in matched_files:
            # Show path relative to script dir for readability
            relative_path = os.path.relpath(filepath, script_dir)
            try:
                with open(filepath, "r", encoding="utf-8") as infile:
                    content = infile.read()
            except Exception as e:
                content = f"[ERROR reading file: {e}]"


            outfile.write(f"{SEPARATOR}\n")
            outfile.write(f"{relative_path}\n")
            outfile.write(f"{content}\n")
            outfile.write(f"{SEPARATOR}\n\n")


    print(f"Done. {len(matched_files)} file(s) compiled into: {output_path}")
    for fp in matched_files:
        print(f"  - {os.path.relpath(fp, script_dir)}")


if __name__ == "__main__":
    scrape_directory()
