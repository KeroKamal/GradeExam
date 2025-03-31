# MCQ Bubble Sheet Generator

This project generates images of multiple-choice question (MCQ) bubble sheets with pre-filled answer bubbles. It creates two types of bubble sheet images:
- **Correct Answers Sheet:** Displays the correct answer for each question in a cyclic pattern.
- **Student Answers Sheet:** Simulates a student's responses with a chance of making mistakes.

## How It Works

1. **Image Layout Setup:**
   - The output image has a fixed width (default is 600 pixels) and dynamically calculated height based on the number of questions and margins.
   - The image includes a header row displaying option labels ("A", "B", "C", "D") and a left margin for question numbers.

2. **Bubble Sheet Creation:**
   - For each question, a row is created where the question number is placed on the left.
   - Four bubbles (one for each option A, B, C, D) are drawn horizontally in each row.
   - Each bubble is represented by an outer circle. If an answer is provided in the input dictionary (mapping question numbers to selected options), the corresponding bubble is filled with a smaller inner circle.

3. **Answer Generation:**
   - **Correct Answers Generation:**  
     The correct answers for each question are generated in a cyclic order (A, B, C, D).
   - **Student Answers Simulation:**  
     The student's responses are simulated with a 30% probability of being incorrect. If an answer is to be wrong, a random incorrect option is chosen.

4. **Output:**
   - The generated bubble sheets are saved as images (e.g., `correct_mcq_sheet.png` and `student_mcq_sheet.png`).
   - A message is printed to confirm that the image has been saved.

## Requirements

- Python 3.x
- OpenCV
- NumPy

## Setup and Usage

1. **Install Dependencies:**
   ```bash
   pip install opencv-python numpy

2. **Run the Script:**

- Save the script to a file (e.g., mcq_generator.py).
- python mcq_generator.py

3. **Output Files:**

- correct_mcq_sheet.png: Contains the correct answers in a cyclic pattern.
- student_mcq_sheet.png: Contains simulated student answers with possible mistakes.

## Conclusion

This tool is useful for creating practice materials or simulating exam answer sheets for automated grading systems.