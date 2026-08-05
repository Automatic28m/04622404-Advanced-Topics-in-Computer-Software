"""
Load data from qa text files and convert it into a list of dictionaries.
The original file format is:
    [Category: category_name] or [หมวดหมู่: category_name]
    Q: question_text
    A: answer_text (can be multi-line)
    (blank line separates each Q&A pair)

Since the actual data is in .txt format, we use "line_no" instead of page numbers
to reference the original location, so that we can still refer to the source text.
"""

import os

def load_qa_file(file_path):
    """
    read the .txt file and split it into individual question-answer pairs

    returns a list of dictionaries, each with the following keys:
        - id: the index of the question-answer pair (starting from 0)
        - category: the category to which the pair belongs
        - question: the question text
        - answer: the answer text (supports multi-line content)
        - line_no: the line number of the "Q:" in the original file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    records = []
    current_category = "Unspecified Category"
    pending_question = None
    pending_answer_lines = []
    pending_line_no = None

    def save_record():
        # Helper function to save the accumulated Q&A pair before starting a new one
        if pending_question is not None and pending_answer_lines:
            records.append({
                "id": len(records),
                "category": current_category,
                "question": pending_question,
                "answer": "\n".join(pending_answer_lines).strip(),
                "line_no": pending_line_no,
            })

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()

        # Skip comments
        if line.startswith("#"):
            continue

        # Blank lines mark the end of a Q&A block in this format
        if line == "":
            save_record()
            pending_question = None
            pending_answer_lines = []
            continue

        # Handle both Thai and English category tags
        if line.startswith("[หมวด") or line.startswith("[Category"):
            save_record() # Save any pending record just in case a blank line was missed
            pending_question = None
            pending_answer_lines = []
            
            # Extract the category name (e.g., from "[Category: Internships]")
            clean_tag = line.strip("[]")
            if ":" in clean_tag:
                current_category = clean_tag.split(":", 1)[1].strip()
            else:
                current_category = clean_tag
            continue

        if line.startswith("Q:"):
            save_record() # Save any pending record just in case
            pending_question = line[len("Q:"):].strip()
            pending_answer_lines = []
            pending_line_no = line_no
            continue

        if line.startswith("A:"):
            # Start capturing the answer
            pending_answer_lines.append(line[len("A:"):].strip())
            continue
            
        # If we are currently capturing an answer, anything that isn't a blank line or tag belongs to the answer
        if pending_question is not None:
            # Use raw_line.rstrip() to preserve potential indentations like bullet points or tabs
            pending_answer_lines.append(raw_line.rstrip('\r\n'))

    # Save the very last record if the file ends without a blank line
    save_record()

    return records