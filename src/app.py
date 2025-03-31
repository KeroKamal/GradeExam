import re
import cv2
import spacy
import numpy as np
import string
import csv
import random
import json
from collections import OrderedDict
from flask import Flask, request, Response
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Initialize models
ocr = PaddleOCR(use_angle_cls=True, lang='en')
nlp = spacy.load("en_core_web_md")
sbert_model = SentenceTransformer("all-mpnet-base-v2")
entailment_classifier = pipeline("text-classification", model="roberta-large-mnli", return_all_scores=True)

app = Flask(__name__)

# --- Subject classification function ---
def classify_subject(question, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["Math", "Science", "History", "Literature", "Geography", "Art"]
    subject_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = subject_classifier(question, candidate_labels)
    return result["labels"][0]

# --- Load advice CSV with subject field ---
def load_advice(filename):
    advice_list = []
    try:
        with open(filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                advice_list.append({
                    "min_score": float(row["min_score"]),
                    "max_score": float(row["max_score"]),
                    "subject": row["subject"],
                    "advice": row["advice"],
                    "study_plan": row["study_plan"],
                    "recommended_books": row["recommended_books"]
                })
    except Exception as e:
        print("Advice file error:", e)
    return advice_list

# --- Get dynamic advice based on score and subject ---
def get_advice(score, subject, advice_list):
    filtered_advices = [
        advice for advice in advice_list 
        if advice["subject"].lower() == subject.lower() and advice["min_score"] <= score <= advice["max_score"]
    ]
    if filtered_advices:
        return random.choice(filtered_advices)
    else:
        return {
            "advice": "No advice available.",
            "study_plan": "No study plan available.",
            "recommended_books": "No books available."
        }

# --- OCR helper for image array (handwritten exam) ---
def ocr_from_array(image):
    # Ensure the image has a contiguous memory layout
    image = np.ascontiguousarray(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result = ocr.ocr(gray, cls=True)
    return "\n".join([line[1][0] for line in result[0]])

def preprocess_text(text):
    return " ".join(token.lemma_ for token in nlp(text.lower()) if not token.is_stop and not token.is_punct)

def text_to_vector_sbert(text):
    return sbert_model.encode(text, convert_to_tensor=True)

def compute_similarity(text1, text2):
    return util.pytorch_cos_sim(text_to_vector_sbert(text1), text_to_vector_sbert(text2)).item()

def contains_keyword(reference, student):
    translator = str.maketrans('', '', string.punctuation)
    return bool(set(reference.lower().translate(translator).split()) &
                set(student.lower().translate(translator).split()))

def check_entailment(student, reference):
    result = entailment_classifier(f"{student} </s></s> {reference}", truncation=True)
    return next((item["score"] for item in result[0] if item["label"] == "ENTAILMENT"), 0.0)

def entity_match(ref_ans, stud_ans):
    return bool({ent.text.lower() for ent in nlp(ref_ans).ents} &
                {ent.text.lower() for ent in nlp(stud_ans).ents})

def extract_numbers(text):
    nums = set(re.findall(r'\d+', text))
    number_words = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10"}
    for word in text.lower().split():
        if (w := word.strip(string.punctuation)) in number_words:
            nums.add(number_words[w])
    return nums

def is_year(text):
    text_clean = text.strip().replace(".", "")
    numbers = re.findall(r'\d{4}', text_clean)
    return len(numbers) == 1 and re.sub(r'\d{4}', '', text_clean).strip(string.punctuation + " ") == ""

def advanced_grade(ref_ans, stud_ans, similarity, threshold=0.8, max_grade=100):
    min_correct_score = 50
    min_incorrect_score = 30
    translator = str.maketrans('', '', string.punctuation)
    ref_clean = ref_ans.lower().translate(translator).strip()
    stud_clean = stud_ans.lower().translate(translator).strip()
    base_grade = similarity * max_grade
    grade = None
    mark = None
    if is_year(ref_ans):
        ref_years, stud_years = re.findall(r'\d{4}', ref_ans), re.findall(r'\d{4}', stud_ans)
        if not stud_years or ref_years[0] != stud_years[0]:
            grade = 0
            if contains_keyword(ref_ans, stud_ans):
                grade = max(grade, min_incorrect_score)
            mark = "Incorrect"
        else:
            grade = max_grade
            mark = "Correct"
    elif ref_clean == stud_clean:
        grade = max_grade
        mark = "Correct"
    elif len(stud_clean.split()) <= 3 and contains_keyword(ref_ans, stud_ans):
        grade = max_grade
        mark = "Correct"
    elif extract_numbers(stud_ans) and (extract_numbers(stud_ans) & extract_numbers(ref_ans)):
        grade = max_grade
        mark = "Correct"
    elif check_entailment(stud_ans, ref_ans) > 0.9:
        grade = max_grade
        mark = "Correct"
    elif entity_match(ref_ans, stud_ans):
        grade = max(base_grade, threshold * max_grade)
        if grade < min_correct_score:
            grade = min_correct_score
        mark = "Correct"
    elif contains_keyword(ref_ans, stud_ans):
        if similarity < threshold:
            grade = max(base_grade, threshold * max_grade)
            if grade < min_correct_score:
                grade = min_correct_score
            mark = "Correct"
        else:
            grade = min(base_grade + 10, max_grade)
            if grade < min_correct_score:
                grade = min_correct_score
            mark = "Correct"
    elif similarity >= threshold:
        grade = min(base_grade + 10, max_grade)
        if grade < min_correct_score:
            grade = min_correct_score
        mark = "Correct"
    else:
        grade = base_grade
        if contains_keyword(ref_ans, stud_ans) and grade < min_incorrect_score:
            grade = min_incorrect_score
        mark = "Incorrect"
    if mark == "Correct":
        ref_word_count = len(ref_ans.split())
        stud_word_count = len(stud_ans.split())
        if ref_word_count > 0 and stud_word_count < ref_word_count:
            ratio = stud_word_count / ref_word_count
            grade = grade * ratio
            if grade < min_correct_score:
                grade = min_correct_score
    return (grade, mark)

def correct_token(token):
    replacements = {
        'o': '0', 'O': '0',
        'l': '1', 'I': '1', '|': '1',
        'z': '2', 'Z': '2',
        'e': '3', 'E': '3',
        'a': '4', 'A': '4',
        'y': '4', 'Y': '4',
        's': '5', 'S': '5',
        'g': '6', 'G': '6',
        't': '7', 'T': '7',
        'b': '8', 'B': '8',
        'q': '9', 'Q': '9',
        'L': '2',
        'i': '1'
    }
    return "".join(replacements.get(char, char) for char in token)

# --- Parsing functions for handwritten answers ---
def parse_reference_answers(text):
    ref_dict = {}
    lines = text.splitlines()
    current_qnum = None
    current_question = ""
    current_answer = ""
    state = "question"
    question_pat = re.compile(r'^(?:Question\s*)?[\(\[]?\s*([A-Za-z0-9]+)[\)\]\-\.]\s*(.*)', re.IGNORECASE)
    answer_pat = re.compile(r'^(?i)Answer[:\s]*(.*)')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        q_match = question_pat.match(line)
        if q_match:
            if current_qnum is not None:
                if not current_question.strip().endswith('?'):
                    current_question = current_question.strip() + '?'
                ref_dict[current_qnum] = {"question": current_question.strip(), "answer": current_answer.strip()}
            try:
                token = q_match.group(1)
                current_qnum = int(token)
            except ValueError:
                try:
                    current_qnum = int(correct_token(q_match.group(1)))
                except ValueError:
                    current_qnum = None
            current_question = q_match.group(2).strip()
            current_answer = ""
            if re.search(r'(?i)\bAnswer[:\s]*', current_question):
                parts = re.split(r'(?i)\bAnswer[:\s]*', current_question, maxsplit=1)
                current_question = parts[0].strip()
                current_answer = parts[1].strip()
                state = "answer"
            else:
                state = "question"
            continue
        a_match = answer_pat.match(line)
        if a_match:
            state = "answer"
            current_answer += " " + a_match.group(1).strip()
        else:
            if state == "question":
                current_question += " " + line
            elif state == "answer":
                current_answer += " " + line
    if current_qnum is not None:
        if not current_question.strip().endswith('?'):
            current_question = current_question.strip() + '?'
        ref_dict[current_qnum] = {"question": current_question.strip(), "answer": current_answer.strip()}
    return ref_dict

def parse_student_answers(text):
    stud_dict = {}
    lines = text.splitlines()
    stud_pat = re.compile(r'^(?:Question\s*)?[\(\[]?\s*([A-Za-z0-9]+)[\)\]\-\.]\s*(.+)', re.IGNORECASE)
    for line in lines:
        line = line.strip()
        if not line:
            continue
        m = stud_pat.match(line)
        if m:
            try:
                token = m.group(1)
                q_num = int(token)
            except ValueError:
                try:
                    q_num = int(correct_token(m.group(1)))
                except ValueError:
                    continue
            stud_dict[q_num] = m.group(2).strip()
    return stud_dict

def grade_answers(ref_dict, stud_dict, advice_list, threshold=0.8, max_grade=100):
    results = []
    for q_num in sorted(ref_dict.keys()):
        ref_entry = ref_dict[q_num]
        ref_question = ref_entry.get('question', f"Question {q_num}")
        ref_ans = ref_entry.get('answer', '')
        subject = classify_subject(ref_question)
        if q_num in stud_dict:
            stud_ans = stud_dict[q_num]
            ref_proc = preprocess_text(ref_ans)
            stud_proc = preprocess_text(stud_ans)
            similarity = compute_similarity(ref_proc, stud_proc)
            grade, mark = advanced_grade(ref_ans, stud_ans, similarity, threshold, max_grade)
            advice_data = get_advice(grade, subject, advice_list)
            results.append({
                "question_number": q_num,
                "question": ref_question,
                "subject": subject,
                "reference": ref_ans,
                "student": stud_ans,
                "similarity": similarity,
                "grade": grade,
                "mark": mark,
                "advice": advice_data["advice"],
                "study_plan": advice_data["study_plan"],
                "recommended_books": advice_data["recommended_books"]
            })
        else:
            advice_data = get_advice(0, subject, advice_list)
            results.append({
                "question_number": q_num,
                "question": ref_question,
                "subject": subject,
                "reference": ref_ans,
                "student": "No answer provided",
                "similarity": 0,
                "grade": 0,
                "mark": "Incorrect",
                "advice": advice_data["advice"],
                "study_plan": advice_data["study_plan"],
                "recommended_books": advice_data["recommended_books"]
            })
    return results

# --- New functions for MCQ exam correction using bubble sheet layout ---
def extract_mcq_answers_from_image(image, num_questions=None):
    # Layout parameters (should match those used in the bubble sheet generator)
    margin = 50
    vertical_gap = 60
    header_gap = vertical_gap  # account for the header row at the top
    start_x = margin + 50       # same as in bubble sheet generator
    horizontal_gap = 100
    bubble_radius = 20
    inner_radius = bubble_radius - 5  # inner filled circle radius

    # Compute num_questions if not provided, subtracting header gap.
    if num_questions is None:
        image_height = image.shape[0]
        num_questions = (image_height - 2 * margin - header_gap) // vertical_gap

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    answers = {}
    for q in range(1, num_questions + 1):
        y = margin + header_gap + (q - 1) * vertical_gap  # adjust y for header
        marked_option = None
        for idx, option in enumerate(["A", "B", "C", "D"]):
            x = start_x + idx * horizontal_gap

            # Define region around the bubble center using inner_radius.
            x1 = max(0, x - inner_radius)
            y1 = max(0, y - inner_radius)
            x2 = min(thresh.shape[1], x + inner_radius)
            y2 = min(thresh.shape[0], y + inner_radius)
            region = thresh[y1:y2, x1:x2]

            # Skip if region is empty.
            if region.size == 0:
                continue

            # Create a circular mask for the region.
            mask = np.zeros(region.shape, dtype=np.uint8)
            # Wrap mask in a UMat to satisfy OpenCV's layout requirements.
            mask_umat = cv2.UMat(mask)
            center_mask = (region.shape[1] // 2, region.shape[0] // 2)
            radius_mask = min(center_mask[0], center_mask[1])
            cv2.circle(mask_umat, center_mask, radius_mask, 255, -1)
            mask = mask_umat.get()

            # Compute the average intensity within the masked circle.
            if np.count_nonzero(mask) == 0:
                continue
            avg = np.mean(region[mask == 255])
            # A filled bubble should be nearly black (low avg). Adjust threshold as needed.
            if avg < 150:
                marked_option = option
                break
        if marked_option is not None:
            answers[q] = marked_option
    return answers

def grade_mcq_answers(correct_dict, student_dict, points_per_question=1):
    correct_questions = []
    incorrect_questions = []
    total_questions = len(correct_dict)
    score = 0
    for q in sorted(correct_dict.keys()):
        correct_ans = correct_dict[q]
        student_ans = student_dict.get(q, None)
        if student_ans is None:
            incorrect_questions.append(q)
        elif student_ans == correct_ans:
            correct_questions.append(q)
            score += points_per_question
        else:
            incorrect_questions.append(q)
    total_grade = (score / (total_questions * points_per_question)) * 100 if total_questions > 0 else 0
    return {
        "Correct Questions": correct_questions,
        "Incorrect Questions": incorrect_questions,
        "Total Grade": total_grade
    }

# --- Flask API endpoint ---
@app.route('/grade_exam', methods=['POST'])
def grade_exam():
    if 'ref_image' not in request.files or 'stud_image' not in request.files:
        return Response(json.dumps({"Error": "Missing one or both image files."}),
                        status=400, mimetype='application/json')

    ref_file = request.files['ref_image']
    stud_file = request.files['stud_image']

    ref_bytes = np.frombuffer(ref_file.read(), np.uint8)
    stud_bytes = np.frombuffer(stud_file.read(), np.uint8)
    ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
    stud_img = cv2.imdecode(stud_bytes, cv2.IMREAD_COLOR)

    if ref_img is None or stud_img is None:
        return Response(json.dumps({"Error": "One or both images could not be processed."}),
                        status=400, mimetype='application/json')

    # For MCQ extraction, compute num_questions dynamically.
    margin = 50
    vertical_gap = 60
    header_gap = vertical_gap  # same header gap used in the generator
    computed_questions = (ref_img.shape[0] - 2 * margin - header_gap) // vertical_gap

    mcq_ref = extract_mcq_answers_from_image(ref_img, num_questions=computed_questions)
    mcq_stud = extract_mcq_answers_from_image(stud_img, num_questions=computed_questions)

    if len(mcq_ref) >= computed_questions // 2 and len(mcq_stud) >= computed_questions // 2:
        result = grade_mcq_answers(mcq_ref, mcq_stud)
        response = {
            "Exam Type": "MCQ",
            "Results": result
        }
    else:
        advice_file = 'data/advice.csv'
        ref_text = ocr_from_array(ref_img)
        stud_text = ocr_from_array(stud_img)
        ref_answers = parse_reference_answers(ref_text)
        stud_answers = parse_student_answers(stud_text)
        advice_list = load_advice(advice_file)
        results = grade_answers(ref_answers, stud_answers, advice_list, threshold=0.8, max_grade=100)
        ordered_results = []
        for res in results:
            od = OrderedDict([
                ("Question Number", res.get("question_number")),
                ("Question", res.get("question")),
                ("Subject", res.get("subject")),
                ("Reference", res.get("reference")),
                ("Student", res.get("student")),
                ("Similarity", res.get("similarity")),
                ("Grade", res.get("grade")),
                ("Mark", res.get("mark")),
                ("Advice", res.get("advice")),
                ("Study Plan", res.get("study_plan")),
                ("Recommended Books", res.get("recommended_books"))
            ])
            ordered_results.append(od)
        if results:
            overall_grade = sum(res["grade"] for res in results) / len(results)
        else:
            overall_grade = 0
        response = {
            "Exam Type": "Handwritten",
            "Results": ordered_results,
            "Overall Grade": overall_grade
        }
    return Response(json.dumps(response, indent=4, ensure_ascii=False), mimetype='application/json')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
