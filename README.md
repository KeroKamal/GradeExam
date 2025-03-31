# 🎓 Exam Grading and Advice API

Welcome to the **Exam Grading and Advice API** – a smart, Flask-based application designed to simplify exam grading. Using advanced OCR and NLP techniques, it not only grades exams (handwritten and multiple-choice) but also provides personalized study advice!

---

## 📖 Overview

This application automates the entire grading process by:

- **Extracting & Preprocessing Text:**  
  🖼️ Uses [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and OpenCV to extract text from exam images.

- **Grading Handwritten Answers:**  
  🧠 Utilizes semantic similarity, entity matching, and entailment analysis to evaluate handwritten responses.

- **Grading Multiple-Choice Questions:**  
  ✅ Automatically reads bubble sheets and grades accordingly.

- **Providing Personalized Feedback:**  
  🎯 Offers tailored advice, study plans, and book recommendations based on exam performance and subject classification.

---

## 🔧 Key Components

- **OCR & Image Processing:**  
  Integrates [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) and [OpenCV](https://opencv.org/) for accurate text extraction.

- **Natural Language Processing:**  
  Implements spaCy for text preprocessing and [SentenceTransformer](https://www.sbert.net/) for semantic similarity assessments.

- **Advanced Analysis:**  
  Uses Hugging Face Transformers (RoBERTa for entailment analysis and BART for zero-shot classification) to ensure thorough evaluation.

- **Exam Grading Logic:**  
  Combines grading algorithms for both handwritten and multiple-choice exams, with dynamic advice based on performance.

---

## 🛠️ Requirements

Ensure you have the following installed:

- **Python 3.x**  
- **Flask**  
  [Flask Documentation](https://palletsprojects.com/p/flask/)  
- **PaddleOCR**  
  [PaddleOCR GitHub](https://github.com/PaddlePaddle/PaddleOCR)  
- **OpenCV**  
  [OpenCV Website](https://opencv.org/) (`opencv-python`)  
- **spaCy**  
  [spaCy Website](https://spacy.io/) *(with model `en_core_web_md`)*  
- **SentenceTransformers**  
  [SentenceTransformer Website](https://www.sbert.net/)  
- **Transformers**  
  [Transformers Documentation](https://huggingface.co/transformers/)  
- **numpy**  
  [NumPy Website](https://numpy.org/)  
- **pandas**  
  [Pandas Documentation](https://pandas.pydata.org/)  
- **scipy**  
  [SciPy Website](https://www.scipy.org/)  
- **csv** *(Standard library)*  
  [Python csv module](https://docs.python.org/3/library/csv.html)  
- **json** *(Standard library)*  
  [Python json module](https://docs.python.org/3/library/json.html)

> **Tip:** Make sure to download the necessary models (e.g., spaCy's `en_core_web_md`) and that transformer models will automatically download on first use.

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/KeroKamal/GradeExam.git
cd GradeExam.git
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Necessary Models

```bash
python -m spacy download en_core_web_md
```

*Note:* Transformer models used by SentenceTransformer and Hugging Face pipelines will be automatically downloaded when first used.

### 5. Run the Application

```bash
python "src/app.py"
```

---

## 🔌 API Usage

### Endpoint Details

- **URL:** `POST /grade_exam`
- **Description:** Accepts two image files – one for the reference exam (answer key) and one for the student's exam. Supports both handwritten and multiple-choice formats.

### Example cURL Command

```bash
curl -X POST http://localhost:5000/grade_exam \
  -F "ref_image=@/path/to/reference_image.jpg" \
  -F "stud_image=@/path/to/student_image.jpg"
```

---

## 📊 Sample Response

```json
{
  "Exam Type": "Handwritten",
  "Results": [
    {
      "Question Number": 1,
      "Question": "What is the capital of France?",
      "Subject": "Geography",
      "Reference": "Paris",
      "Student": "Paris",
      "Similarity": 0.95,
      "Grade": 100,
      "Mark": "Correct",
      "Advice": "Keep up the good work!",
      "Study Plan": "Review geography chapters 2 and 3.",
      "Recommended Books": "Geography 101"
    },
    ...
  ],
  "Overall Grade": 87.5
}
```

---

## 📌 Final Notes

- Ensure your environment is properly set up with all dependencies.
- For any issues, refer to the documentation links provided.

Happy coding! 😎🎉

---
