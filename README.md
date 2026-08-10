# Support Ticket Auto-Categorizer

Predicts the category (Billing / Technical / HR / General) of an incoming
support ticket from its subject + body text, with a confidence score,
a human-review fallback for low-confidence tickets, and simple
urgent/normal priority tagging.

## How to run

```bash
pip install -r requirements.txt
python ticket_categorizer.py            # trains, evaluates, predicts 5 new tickets
python ticket_categorizer.py --demo     # also launches an interactive CLI classifier
```

## Files
- `tickets_dataset.csv` – 60 dummy labeled tickets (15 per category)
- `ticket_categorizer.py` – full pipeline: clean → TF-IDF → train → evaluate → predict → CLI demo
- `requirements.txt` – pandas + scikit-learn

## A note on the accuracy number

The 83% test accuracy is measured on just a 12-ticket held-out split (20% of
60 total), so it should be read as "the pipeline works end-to-end and gets
most cases right" rather than a statistically solid estimate — with a sample
that small, one or two misclassified tickets swing the number by ~8 points.
This is the same limitation called out below: more real data is the fix.

## Approach summary

Cleaned raw ticket text (lowercased, stripped URLs/emails/punctuation/numbers,
removed stopwords), then vectorized with TF-IDF (unigrams, sublinear TF) so
category-distinctive words like "invoice" or "crash" are weighted higher than
generic words like "please". Tried both Multinomial Naive Bayes and Logistic
Regression; went with Logistic Regression because its predicted probabilities
were far better calibrated on this small dataset, which matters since the
confidence-score and human-review-threshold features depend directly on
trustworthy probabilities. Tickets below 60% confidence are routed to
"Needs Human Review" instead of being auto-assigned, and a small keyword
rule layer tags tickets as Urgent/Normal independent of the category model.

## Reflection — what I'd improve with more data or time

- With more real ticket data (hundreds/thousands per category instead of 15),
  I'd expect both TF-IDF+LogReg and Naive Bayes to get meaningfully more
  accurate and confident, and I'd re-check whether NB actually becomes the
  better/cheaper option at that scale.
- I'd add proper stemming/lemmatization instead of a hand-written stopword
  list, and try character n-grams to handle typos and informal phrasing.
- The "General" category is really a catch-all right now — with more data
  I'd want to see if it's actually hiding sub-categories worth splitting out.
- I'd replace the fixed 60% confidence threshold with one tuned per class
  using a validation set, since some categories (e.g. HR vs General) are
  inherently harder to separate than others (e.g. Technical is very distinct).
- For production, I'd log every low-confidence/misrouted ticket and use it
  to retrain periodically — that feedback loop matters more long-term than
  the initial model choice.

## Github Link
https://github.com/VishnuNagarajanS/ticket_categorizer