import re
import cv2
import spacy
import numpy as np
import os
import string
import csv
import random
import json
import requests
from collections import OrderedDict
from flask import Flask, request, Response
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Ensure the language model is available
try:
    import en_core_web_md
except ImportError:
    print("en_core_web_md not found. Downloading now...")
    import spacy.cli
    spacy.cli.download("en_core_web_md")
    import en_core_web_md

# Load the model using one method.
nlp = en_core_web_md.load()

# Initialize other components
ochr = PaddleOCR(use_angle_cls=True, lang='en')
sbert_model = SentenceTransformer("all-mpnet-base-v2")
entailment_classifier = pipeline(
    "text-classification",
    model="roberta-large-mnli",
    return_all_scores=True
)

app = Flask(__name__)

def classify_subject(question, candidate_labels=None):
    if candidate_labels is None:
        candidate_labels = ["Math", "Science", "History", "Literature", "Geography", "Art"]
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    result = classifier(question, candidate_labels)
    return result["labels"][0]

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
                    "advice_parent": row["advice_parent"],
                    "advice_teacher": row["advice_teacher"],
                    "study_plan": row["study_plan"],
                    "recommended_books": row["recommended_books"]
                })
    except Exception as e:
        print("Advice file error:", e)
    return advice_list

def get_advice(score, subject, advice_list):
    filtered = [a for a in advice_list
                if a["subject"].lower() == subject.lower()
                and a["min_score"] <= score <= a["max_score"]]
    if filtered:
        return random.choice(filtered)
    return {
        "advice_parent": "No parent advice available.",
        "advice_teacher": "No teacher advice available.",
        "study_plan": "No study plan available.",
        "recommended_books": "No books available."
    }
    
def ocr_from_array(image):
    image = np.ascontiguousarray(image)
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print("Error converting image to grayscale:", e)
        return ""
    result = ochr.ocr(gray, cls=True)
    
    # If result is None or empty, log and return an empty string.
    if not result or not result[0]:
        print("PaddleOCR returned no results for this image.")
        return ""
    
    # Join the detected text parts.
    try:
        # This assumes result[0] contains the OCR detections for the image.
        return "\n".join([line[1][0] for line in result[0]])
    except Exception as e:
        print("Error processing OCR result:", e)
        return ""


def preprocess_text(text):
    return " ".join(
        token.lemma_ for token in nlp(text.lower())
        if not token.is_stop and not token.is_punct
    )

def text_to_vector_sbert(text):
    return sbert_model.encode(text, convert_to_tensor=True)

def compute_similarity(text1, text2):
    return util.pytorch_cos_sim(
        text_to_vector_sbert(text1),
        text_to_vector_sbert(text2)
    ).item()

def contains_keyword(reference, student):
    tr = str.maketrans('', '', string.punctuation)
    return bool(
        set(reference.lower().translate(tr).split()) &
        set(student.lower().translate(tr).split())
    )

def check_entailment(student, reference):
    scores = entailment_classifier(f"{student} </s></s> {reference}", truncation=True)
    for item in scores[0]:
        if item["label"] == "ENTAILMENT":
            return item["score"]
    return 0.0

def entity_match(ref_ans, stud_ans):
    return bool({ent.text.lower() for ent in nlp(ref_ans).ents} &
                {ent.text.lower() for ent in nlp(stud_ans).ents})

def extract_numbers(text):
    nums = set(re.findall(r'\d+', text))
    words = {"zero": "0", "one": "1", "two": "2", "three": "3",
             "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9", "ten": "10"}
    for w in text.lower().split():
        tok = w.strip(string.punctuation)
        if tok in words:
            nums.add(words[tok])
    return nums

def is_year(text):
    clean = text.strip().replace(".", "")
    years = re.findall(r'\d{4}', clean)
    return len(years) == 1 and re.sub(r'\d{4}', '', clean).strip(string.punctuation + " ") == ""

def advanced_grade(ref_ans, stud_ans, similarity, threshold=0.8, max_grade=100):
    min_corr, min_inc = 50, 30
    tr = str.maketrans('', '', string.punctuation)
    r = ref_ans.lower().translate(tr).strip()
    s = stud_ans.lower().translate(tr).strip()
    base = similarity * max_grade
    if is_year(ref_ans):
        ref_years = re.findall(r'\d{4}', ref_ans)
        stud_years = re.findall(r'\d{4}', stud_ans)
        if not stud_years or ref_years[0] != stud_years[0]:
            grade = min_inc if contains_keyword(ref_ans, stud_ans) else 0
            mark = "Incorrect"
        else:
            grade, mark = max_grade, "Correct"
    elif r == s or (len(s.split()) <= 3 and contains_keyword(ref_ans, stud_ans)) or \
         (extract_numbers(stud_ans) & extract_numbers(ref_ans)) or \
         check_entailment(stud_ans, ref_ans) > 0.9:
        grade, mark = max_grade, "Correct"
    elif entity_match(ref_ans, stud_ans) or (contains_keyword(ref_ans, stud_ans) and similarity < threshold):
        grade = max(base, threshold * max_grade)
        mark = "Correct"
    elif contains_keyword(ref_ans, stud_ans) or similarity >= threshold:
        grade = min(base + 10, max_grade)
        mark = "Correct"
    else:
        grade = max(base, min_inc) if contains_keyword(ref_ans, stud_ans) else base
        mark = "Incorrect"
    if mark == "Correct":
        rw, sw = len(ref_ans.split()), len(stud_ans.split())
        if rw > 0 and sw < rw:
            grade = max(min_corr, grade * (sw / rw))
    return grade, mark

def correct_token(token):
    rep = {'o':'0','O':'0','l':'1','I':'1','|':'1','z':'2','Z':'2',
           'e':'3','E':'3','a':'4','A':'4','y':'4','Y':'4','s':'5','S':'5',
           'g':'6','G':'6','t':'7','T':'7','b':'8','B':'8','q':'9','Q':'9'}
    return ''.join(rep.get(c, c) for c in token)

def fix_question_prefix(line):
    if not line:
        return line
    first, rest = line[0], line[1:]
    mapping = {'I': '1', 'l': '1', '|': '1', 'S': '5', 's': '5'}
    if first in mapping and rest and rest[0] in ".- )":
        return mapping[first] + rest
    return line

def parse_reference_answers(text):
    ref_dict = {}
    lines = text.splitlines()
    current_question = None
    question_text = ""
    answer_text = ""
    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = fix_question_prefix(raw_line).strip()
        if not line:
            i += 1
            continue
        q_match = re.match(r'^(\d+)[\s\.\-]+(.+)', line)
        if q_match:
            if current_question is not None and question_text:
                ref_dict[current_question] = {"question": question_text.strip(), "answer": answer_text.strip()}
            current_question = int(q_match.group(1))
            question_text = q_match.group(2).strip()
            answer_text = ""
            if i + 1 < len(lines) and "answer" in lines[i+1].lower():
                answer_line = fix_question_prefix(lines[i+1]).strip()
                answer_match = re.match(r'^answer:?[ \t]*(.+)', answer_line, re.IGNORECASE)
                if answer_match:
                    answer_text = answer_match.group(1).strip()
                    i += 2
                    continue
            i += 1
            continue
        a_match = re.match(r'^answer:?[ \t]*(.+)', line, re.IGNORECASE)
        if a_match and current_question is not None:
            answer_text = a_match.group(1).strip()
            i += 1
            continue
        if current_question is not None:
            if not answer_text:
                question_text += " " + line
            else:
                answer_text += " " + line
        i += 1
    if current_question is not None and question_text:
        ref_dict[current_question] = {"question": question_text.strip(), "answer": answer_text.strip()}
    for q in ref_dict:
        if not ref_dict[q]["question"].strip().endswith('?'):
            ref_dict[q]["question"] += '?'
    return ref_dict

def parse_student_answers(text):
    stud_dict = {}
    lines = text.splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r'^(\d+)[\s\.\-]+(.+)', line)
        if match:
            stud_dict[int(match.group(1))] = match.group(2).strip()
            continue
        match = re.match(r'^(\d+)[\.|\)][\s]*(.+)', line)
        if match:
            stud_dict[int(match.group(1))] = match.group(2).strip()
    return stud_dict

def print_parsed_answers(ref_dict, stud_dict):
    print("\n" + "="*80)
    print("PARSED QUESTIONS AND ANSWERS".center(80))
    print("="*80)
    for q in sorted(ref_dict.keys()):
        print(f"\nQuestion {q}:")
        print(f"  Question text: {ref_dict[q]['question']}")
        print(f"  Reference answer: {ref_dict[q]['answer']}")
        print(f"  Student answer: {stud_dict.get(q, 'No answer provided')}")
    print("\nMissing reference questions:", set(stud_dict.keys()) - set(ref_dict.keys()))
    print("Missing student answers:", set(ref_dict.keys()) - set(stud_dict.keys()))
    print("="*80 + "\n")

def display_results_in_terminal(results, mcq_results=None):
    print("\n" + "="*80)
    print("GRADING DETAILS".center(80))
    print("="*80)
    if results:
        print("\nFREE-TEXT ANSWERS GRADING:\n")
        for r in results:
            print(f"Question {r['Question Number']}:")
            print(f"  Subject: {r['Subject']}")
            print(f"  Similarity Score: {r['Similarity']:.2f}")
            print(f"  Grade: {r['Grade']:.1f}")
            print(f"  Mark: {r['Mark']}")
            print("-"*70)
    if mcq_results:
        print("\nMCQ ANSWERS GRADING:\n")
        print(f"Correct Questions: {mcq_results['Correct Questions']}")
        print(f"Incorrect Questions: {mcq_results['Incorrect Questions']}")
        print(f"Total Grade: {mcq_results['Total Grade']:.1f}")
        print(f"Letter Grade: {mcq_results['Letter Grade']}")
    print("="*80 + "\n")

def grade_answers(ref_dict, stud_dict, advice_list, threshold=0.8, max_grade=100):
    results, total, p_adv, t_adv = [], 0, "", ""
    for q in sorted(ref_dict):
        entry = ref_dict[q]
        sim = compute_similarity(
            preprocess_text(entry['answer']),
            preprocess_text(stud_dict.get(q, ''))
        )
        grade, mark = advanced_grade(entry['answer'], stud_dict.get(q, ''), sim, threshold, max_grade)
        total += grade
        adv = get_advice(grade, classify_subject(entry['question']), advice_list)
        if not p_adv and adv['advice_parent']:
            p_adv = adv['advice_parent']
        if not t_adv and adv['advice_teacher']:
            t_adv = adv['advice_teacher']
        results.append(OrderedDict([
            ("Question Number", q),
            ("Question", entry['question']),
            ("Subject", classify_subject(entry['question'])),
            ("Reference", entry['answer']),
            ("Student", stud_dict.get(q, 'No answer provided')),
            ("Similarity", sim),
            ("Grade", grade),
            ("Mark", mark),
            ("Advice for Parents", adv['advice_parent']),
            ("Advice for Teachers", adv['advice_teacher']),
            ("Study Plan", adv['study_plan']),
            ("Recommended Books", adv['recommended_books'])
        ]))
    overall = total / len(ref_dict) if ref_dict else 0
    display_results_in_terminal(results)
    return results, overall, numeric_to_letter_grade(overall), \
           (p_adv or "Encourage your child to review areas where they struggled."), \
           (t_adv or "Consider focusing additional instruction on areas where the student showed weakness.")

def extract_mcq_answers_from_image(image, num_questions=None):
    margin, vgap, header = 50, 60, 60
    if num_questions is None:
        num_questions = (image.shape[0] - 2*margin - header) // vgap
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    answers = {}
    for i in range(1, num_questions + 1):
        y = margin + header + (i - 1) * vgap
        for idx, opt in enumerate(["A", "B", "C", "D"]):
            x = margin + 50 + idx * 100
            r = 15
            reg = thresh[y-r:y+r, x-r:x+r]
            if reg.size and np.mean(reg) < 150:
                answers[i] = opt
                break
    return answers

def numeric_to_letter_grade(grade):
    if grade >= 90: return "A+"
    if grade >= 85: return "A"
    if grade >= 80: return "A-"
    if grade >= 75: return "B+"
    if grade >= 70: return "B"
    if grade >= 65: return "B-"
    if grade >= 60: return "C+"
    if grade >= 50: return "C"
    if grade >= 40: return "D+"
    if grade >= 30: return "D"
    return "F"

def grade_mcq_answers(correct_dict, student_dict, points_per_question=1):
    correct, incorrect = [], []
    score = 0
    for q in sorted(correct_dict):
        if student_dict.get(q) == correct_dict[q]:
            correct.append(q)
            score += points_per_question
        else:
            incorrect.append(q)
    total = (score / (len(correct_dict) * points_per_question)) * 100 if correct_dict else 0
    return {"Correct Questions": correct,
            "Incorrect Questions": incorrect,
            "Total Grade": total,
            "Letter Grade": numeric_to_letter_grade(total)}

def generate_random_id():
    return random.randint(10000, 99999)

@app.route('/grade_exam', methods=['POST'])
def grade_exam():
    if 'ref_image' not in request.files or 'stud_image' not in request.files:
        return Response(json.dumps({"Error": "Missing one or both image files."}), status=400, mimetype='application/json')
    ref_file = request.files['ref_image']
    stud_file = request.files['stud_image']
    ref_bytes = np.frombuffer(ref_file.read(), np.uint8)
    stud_bytes = np.frombuffer(stud_file.read(), np.uint8)
    ref_img = cv2.imdecode(ref_bytes, cv2.IMREAD_COLOR)
    stud_img = cv2.imdecode(stud_bytes, cv2.IMREAD_COLOR)
    if ref_img is None or stud_img is None:
        return Response(json.dumps({"Error": "One or both images could not be processed."}), status=400, mimetype='application/json')
    margin, vgap, header = 50, 60, 60
    computed_questions = (ref_img.shape[0] - 2*margin - header) // vgap
    mcq_ref = extract_mcq_answers_from_image(ref_img, num_questions=computed_questions)
    mcq_stud = extract_mcq_answers_from_image(stud_img, num_questions=computed_questions)
    if len(mcq_ref) >= computed_questions // 2 and len(mcq_stud) >= computed_questions // 2:
        mcq_result = grade_mcq_answers(mcq_ref, mcq_stud)
        total_grade = mcq_result["Total Grade"]
        letter_grade = mcq_result["Letter Grade"]
        parent_advice = "Review incorrect answers with your child and focus on identified knowledge gaps."
        teacher_advice = "Consider revisiting topics with high error rates in upcoming lessons."
        display_results_in_terminal(None, mcq_result)
    else:
        advice_file = 'data/advice.csv'
        ref_text = ocr_from_array(ref_img)
        stud_text = ocr_from_array(stud_img)
        ref_answers = parse_reference_answers(ref_text)
        stud_answers = parse_student_answers(stud_text)
        print_parsed_answers(ref_answers, stud_answers)
        advice_list = load_advice(advice_file)
        results, total_grade, letter_grade, parent_advice, teacher_advice = grade_answers(
            ref_answers, stud_answers, advice_list, threshold=0.8, max_grade=100
        )
    exam_id = request.form.get("examId")
    student_idg = request.form.get("StudentIDg")
    parent_id = request.form.get("parentId")
    teacher_id = request.form.get("teacherId")
    grade_payload = {
        "id": str(generate_random_id()),
        "examId": exam_id,
        "obtainedMarks": str(total_grade),
        "grade": letter_grade,
        "StudentIDg": student_idg
    }
    advice_payload = {
        "id": str(generate_random_id()),
        "parentAdvice": parent_advice,
        "teacherAdvice": teacher_advice,
        "parentId": parent_id,
        "teacherId": teacher_id
    }
    try:
        grade_resp = requests.post("http://54.242.19.19:3000/api/grades/", json=grade_payload)
        advice_resp = requests.post("http://54.242.19.19:3000/api/advices/create/", json=advice_payload)
        print("→ Posted grade payload:", json.dumps(grade_payload, indent=2))
        print("→ Grade API response:", grade_resp.status_code, grade_resp.text)
        print("→ Posted advice payload:", json.dumps(advice_payload, indent=2))
        print("→ Advice API response:", advice_resp.status_code, advice_resp.text)
    except Exception as e:
        print("Error sending to external APIs:", e)
    return Response(
        json.dumps({"status": "ok", "message": "Grade and advice sent to external services."}),
        status=200, mimetype="application/json"
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port, debug=False)
