# Curated collaborative research code; see docs/PROVENANCE.md.
import re
from pathlib import Path
from file_utils.file_utils import load_json_file, save_json_file


def check_files_in_directory(directory, latex_file_path):
    # create a new directory if it does not exist
    Path(directory).mkdir(parents=True, exist_ok=True)
    # Read all files in the directory
    dir_files = [f.name for f in Path(directory).rglob("*") if f.is_file()]
    # only keep json files
    dir_files = [f for f in dir_files if f.endswith(".json")]
    # Parse the chapters from the LaTeX file
    chapters_all = parse_chapters(latex_file_path)
    chapter_titles = set(chapters_all.keys())  # Set of chapter titles from the LaTeX file

    # Check if each file in the directory corresponds to a chapter in the LaTeX file
    for file in dir_files:
        # Check if the file name (without extension) matches a chapter title
        file_name_without_extension = Path(file).stem

        if file_name_without_extension not in chapter_titles:
            print(f"File '{file}' does not exist in dataset.tex chapters")


def parse_chapters(latex_file_path):
    # 1. Read the LaTeX file
    with open(latex_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 2. Use a regular expression to match:
    #    - \\chapter\{(.*?)\} for the chapter title
    #    - (.*?)((?=\\chapter\{)|\Z) for the content until the next \chapter or end of file
    #    - re.DOTALL ensures '.' matches newlines as well
    pattern = r'\\chapter\{(.*?)\}(.*?)((?=\\chapter\{)|\Z)'
    matches = re.findall(pattern, text, flags=re.DOTALL)

    # 3. Create a dictionary of chapter title -> chapter content
    chapters_dict = {}
    for match in matches:
        # match[0] is the chapter title, match[1] is the chapter content
        chapter_title = match[0].strip().replace(".", "_")
        chapter_content = match[1].strip()

        chapters_dict[chapter_title] = chapter_content

    return chapters_dict


def chapter_to_json(target_folder, chapter_title, chapter_content, update=True, clear_solutions_on_change=True):
    # Ensure the directory exists before any file operations
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    metadata_dict = {}

    # Regex to find the entire itemize block in the "Metadata" section
    metadata_block_pattern = r'\\section\{Metadata\}.*?\\begin\{itemize\}(.*?)\\end\{itemize\}'
    metadata_block_match = re.search(metadata_block_pattern, chapter_content, flags=re.DOTALL)

    if metadata_block_match:
        metadata_block = metadata_block_match.group(1)

        item_pattern = r'\\item\s+\\textbf\{(.*?)\}:\s*(.*?)(?=(\\item|$))'
        item_matches = re.findall(item_pattern, metadata_block, re.DOTALL)

        metadata_dict = {}
        for key, value, _ in item_matches:
            metadata_dict[key.strip()] = value.strip()

    section_pattern = r'(\\sub(?:section|subsection)\{(.*?)\})(.*?)(?=(\\sub(?:section|subsection)\{|$))'
    matches = re.findall(section_pattern, chapter_content, re.DOTALL)

    # Create a dictionary with the section name as the key and content as the value
    section_dict = {}
    for _, section_name, content, _ in matches:
        section_dict[section_name.strip()] = content.strip()

    # Recursively search for the json file in all subdirectories
    target_path = Path(target_folder)
    json_file_path = None

    # Search for the file in all subdirectories
    for path in target_path.rglob(f"{chapter_title}.json"):
        json_file_path = path
        break

    # If file not found, create it in the target folder
    if json_file_path is None:
        json_file_path = target_path / f"{chapter_title}.json"
        update = False

    # Ensure parent directory exists for the json file (safety check)
    json_file_path.parent.mkdir(parents=True, exist_ok=True)

    if update:
        problem_dict = load_json_file(json_file_path)
        # print(f"Updating {json_file_path.name}")
        problem_metadata = problem_dict.get("problem_metadata", {})
        metadata_difference = {k: (v, metadata_dict.get(k, "Not in metadata_dict")) for k, v in
                               problem_metadata.items() if metadata_dict.get(k) != v}

        problem_details = problem_dict.get("problem_details", {})
        details_difference = {k: (v, section_dict.get(k, "Not in section_dict")) for k, v in problem_details.items()
                              if section_dict.get(k) != v}
        details_difference.pop("test", None )

        # Print the unmatched entries
        if metadata_difference:
            print(f"\n[{json_file_path.name}] Updating metadata:")
            for key, (old_val, new_val) in metadata_difference.items():
                print(f"  • {key}:")
                print(f"    - Old: {old_val}")
                print(f"    - New: {new_val}")
            problem_dict["problem_metadata"].update(metadata_dict)

        if details_difference:
            changed_sections = list(details_difference.keys())
            # if new_val does not haave the changed section, print it does not have instead
            print(f"\n[{json_file_path.name}] have changes in the following sections:")
            for section in changed_sections:
                old_val, new_val = details_difference[section]

                # Check if the new content does not contain the section (new_val is "Not in section_dict")
                if new_val == "Not in section_dict":
                    print(f"  • {section}: This section is missing in the tex document.")
                else:
                    # Otherwise, print the change between the old and new value
                    print(f"  • {section}:")
                    print(f"    - Old: {old_val}")
                    print(f"    - New: {new_val}")

            problem_dict["problem_details"].update(section_dict)
            if clear_solutions_on_change and (
                    "Problem Statement" in details_difference or "Answer Requirements" in details_difference):
                print(f"  ⚠️  Problem core content changed - clearing all solutions")
                # Print the specific changes in core content
                for key in ["Problem Statement", "Answer Requirements"]:
                    if key in details_difference:
                        old_val, new_val = details_difference[key]
                        print(f"\n  Changes in {key}:")
                        print(f"    - Old: {old_val}")
                        print(f"    - New: {new_val}")
                problem_dict["model_solutions"] = []

        if problem_dict.get("problem_id", "") != chapter_title:
            print(f"\n[{json_file_path.name}] Updating problem_id:")
            print(f"  • {problem_dict.get('problem_id', '')} → {chapter_title}")
            problem_dict["problem_id"] = chapter_title

    else:
        problem_dict = {
            "problem_id": chapter_title,
            "problem_metadata": metadata_dict,
            "problem_details": section_dict,
            "model_solutions": []
        }
    print(json_file_path)
    save_json_file(json_file_path, problem_dict)
