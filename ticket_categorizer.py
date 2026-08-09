"""
Support Ticket Auto-Categorizer
--------------------------------
Reads support ticket text (subject + body) and predicts one of:
Billing / Technical / HR / General

Pipeline: clean text -> TF-IDF vectorize -> Multinomial Naive Bayes classifier
-> evaluate (accuracy, precision/recall, confusion matrix) -> predict on new
unseen tickets with confidence score + human-review fallback + priority tag.
"""

import re
import string
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
DATA_PATH = "tickets_dataset.csv"
df = pd.read_csv(DATA_PATH)

# Combine subject + body into a single text field — subject often carries
# strong signal ("Refund request", "App crashes") so we don't throw it away.
df["text"] = df["subject"].fillna("") + " " + df["body"].fillna("")

# ----------------------------------------------------------------------
# 2. TEXT CLEANING / PREPROCESSING
# ----------------------------------------------------------------------
# Minimal, fast stopword list (kept local so no extra downloads like nltk
# corpora are needed — important for a tool that has to run instantly on a
# live queue).
STOPWORDS = set("""
a an the is are was were be been being to of in on for with and or but if
this that these those i you he she it we they my your his her its our their
me him her us them at as by from up down out about into over after before
do does did doing have has had having will would shall should can could may
might must not no nor so than too very just also
""".split())

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # strip URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # strip emails
    text = text.translate(str.maketrans("", "", string.punctuation))  # punctuation
    text = re.sub(r"\d+", " ", text)                      # strip standalone numbers
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(clean_text)

# ----------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["category"],
    test_size=0.2, random_state=42, stratify=df["category"]
)

# ----------------------------------------------------------------------
# 4. FEATURE REPRESENTATION — TF-IDF
# ----------------------------------------------------------------------
# TF-IDF over Bag-of-Words because raw word counts let common-but-uninformative
# words (e.g. "please", "account") dominate. TF-IDF down-weights terms that
# appear across many tickets and up-weights terms distinctive to a category
# (e.g. "invoice", "crash", "leave", "partnership").
vectorizer = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ----------------------------------------------------------------------
# 5. MODEL — Logistic Regression (Multinomial Naive Bayes also tried)
# ----------------------------------------------------------------------
# Both MultinomialNB and LogisticRegression were tried on this dataset.
# NB is the classic fast baseline for text, but on a small dummy dataset its
# independence assumption between words makes predict_proba badly
# calibrated (everything hovers around 30-45% confidence, even correct
# predictions) — that's unusable for a confidence-based triage system.
# LogisticRegression directly optimizes class probabilities, so its
# confidence scores are much more trustworthy and separate "the model is
# sure" from "the model is guessing" far more cleanly. Since this tool's
# whole bonus feature set (confidence %, review threshold) depends on
# well-calibrated probabilities, Logistic Regression is the better real
# choice here — with the honest caveat that with more real ticket data,
# MultinomialNB would likely close the gap and is cheaper to retrain often.
model = LogisticRegression(max_iter=1000, C=5)
model.fit(X_train_vec, y_train)

# (Kept for comparison / reference — uncomment to see the NB numbers)
# nb_model = MultinomialNB()
# nb_model.fit(X_train_vec, y_train)

# ----------------------------------------------------------------------
# 6. EVALUATION
# ----------------------------------------------------------------------
y_pred = model.predict(X_test_vec)

print("=" * 60)
print("EVALUATION ON HELD-OUT TEST SET")
print("=" * 60)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}\n")
print("Classification report (precision / recall / f1):")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix (rows = actual, cols = predicted):")
labels = sorted(df["category"].unique())
cm = confusion_matrix(y_test, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)
print()

# ----------------------------------------------------------------------
# 7. PRIORITY TAGGING (rule-based, bonus)
# ----------------------------------------------------------------------
URGENT_KEYWORDS = [
    "down", "urgent", "not working", "asap", "immediately", "broken",
    "crash", "critical", "failed", "cannot", "can't", "blocked", "error 500",
]

def get_priority(raw_text: str) -> str:
    t = raw_text.lower()
    return "Urgent" if any(kw in t for kw in URGENT_KEYWORDS) else "Normal"

# ----------------------------------------------------------------------
# 8. PREDICTION FUNCTION — confidence score + human-review fallback
# ----------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60  # below this -> route to manual review queue

def classify_ticket(raw_text: str) -> dict:
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_

    best_idx = probs.argmax()
    predicted_category = classes[best_idx]
    confidence = probs[best_idx]

    needs_review = confidence < CONFIDENCE_THRESHOLD
    priority = get_priority(raw_text)

    return {
        "text": raw_text,
        "predicted_category": "Needs Human Review" if needs_review else predicted_category,
        "model_top_guess": predicted_category,
        "confidence": round(float(confidence) * 100, 2),
        "priority": priority,
        "needs_human_review": needs_review,
    }

# ----------------------------------------------------------------------
# 9. PREDICT ON 5 NEW, UNSEEN SAMPLE TICKETS (written fresh, not in dataset)
# ----------------------------------------------------------------------
new_tickets = [
    "My card got charged twice this month for the same subscription, please refund the extra payment.",
    "The server has been down since morning and the app keeps crashing, this is urgent!",
    "I applied for paternity leave last week, can you tell me the approval status?",
    "Just checking, what are your customer support working hours during weekends?",
    "The thing isn't working right and I'm not sure what's going on, can someone take a look?",
]

print("=" * 60)
print("PREDICTIONS ON 5 NEW UNSEEN TICKETS")
print("=" * 60)
for t in new_tickets:
    result = classify_ticket(t)
    print(f"\nTicket: {result['text']}")
    print(f"  -> Category      : {result['predicted_category']}")
    print(f"  -> Confidence    : {result['confidence']}%")
    print(f"  -> Priority      : {result['priority']}")
    print(f"  -> Needs review? : {result['needs_human_review']}")

# ----------------------------------------------------------------------
# 10. MINI LIVE DEMO (CLI) — bonus
# ----------------------------------------------------------------------
def run_cli_demo():
    print("\n" + "=" * 60)
    print("LIVE TICKET CLASSIFIER — type a ticket, or 'quit' to exit")
    print("=" * 60)
    while True:
        user_input = input("\nEnter ticket text: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Exiting demo.")
            break
        if not user_input:
            continue
        result = classify_ticket(user_input)
        print(f"  Category   : {result['predicted_category']}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Priority   : {result['priority']}")

if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv:
        run_cli_demo()
