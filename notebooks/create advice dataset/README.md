# Advice CSV Generator

This project generates a CSV file containing advice, study plans, and book recommendations for various subjects based on different score ranges. The generated entries are randomized using subject-specific templates.

## Overview

The script performs the following tasks:
1. **Template Definition:**
   - For each subject (Math, Science, History, Literature, Geography, Art), the script defines multiple templates for advice, study plans, and recommended books.
   - It also defines a list of topics related to each subject.

2. **Content Generation:**
   - The function `generate_subject_content` selects random templates and topics to create customized text for advice, study plans, and book recommendations.
   - A realistic daily study time is randomly chosen between 1 and 5 hours and incorporated into the study plan.

3. **Entry Generation:**
   - The function `generate_random_entries` generates a specified number of entries for each score category and subject.
   - Each entry includes a minimum and maximum score, the subject name, and the generated advice content.

4. **CSV File Creation:**
   - The function `generate_advice_csv` defines several score categories (e.g., 0-49, 50-74, 75-89, 90-100).
   - It generates 1000 entries per subject for each score category.
   - All generated entries are written into a CSV file with the fields: `min_score`, `max_score`, `subject`, `advice`, `study_plan`, and `recommended_books`.

## Requirements

- Python 3.x
- No external libraries are required beyond the Python Standard Library (uses `csv` and `random`).

## Setup and Usage

1. **Download the Script:**
   - Save the provided Python script to a file (e.g., `generate_advice.py`).

2. **Run the Script:**
   ```bash
   python generate_advice.py

3. **CSV Output:**
   - The CSV file will contain a large number of unique entries with advice, study plans, and book recommendations for each subject and score range.
   - Each row represents a unique advice entry based on the defined templates and topics.

## Customization

1. **Templates and Topics:**
  You can modify or extend the templates and topics for each subject within the generate_subject_content function.

2. **Score Categories and Entry Count:**
 - Adjust the categories list in generate_advice_csv to change score ranges.
 - Change the entries_per_category variable to generate a different number of entries per subject per score category.

## Conclusion

This tool is useful for educators, students, or any application that requires customized study advice based on academic performance.



