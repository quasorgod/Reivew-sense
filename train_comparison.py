"""
ReviewSense - Head-to-head ML vs DL comparison on IMDB sentiment analysis.

Trains classical ML models (TF-IDF features) and evaluates a pretrained
Transformer (DistilBERT) on the EXACT SAME test data for a fair comparison.

Also runs "challenge examples" that test contextual understanding — negation,
sarcasm, contrast — where DL's self-attention should outperform bag-of-words ML.

Run:
    python -I train_comparison.py                   # default: 5000 train, 1000 test
    python -I train_comparison.py --train 10000     # more data (better DL accuracy)
    python -I train_comparison.py --quick            # 2000 train, 500 test (fast demo)
"""

import argparse, json, os, random, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── CLI ──────────────────────────────────────────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("--train", type=int, default=5000)
p.add_argument("--test", type=int, default=1000)
p.add_argument("--quick", action="store_true", help="Use 2000/500 for a quick demo")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
if args.quick:
    args.train, args.test = 2000, 500

random.seed(args.seed)
np.random.seed(args.seed)
os.makedirs("results", exist_ok=True)

# ── 1. Load IMDB dataset ────────────────────────────────────────────────────
print("=" * 60)
print("ReviewSense: ML vs DL Comparison Training")
print("=" * 60)
print(f"\nLoading IMDB dataset ({args.train} train / {args.test} test)...")

from datasets import load_dataset
ds = load_dataset("stanfordnlp/imdb")
train_ds = ds["train"].shuffle(seed=args.seed).select(range(args.train))
test_ds = ds["test"].shuffle(seed=args.seed).select(range(args.test))

X_train = train_ds["text"]
y_train = np.array(train_ds["label"])
X_test = test_ds["text"]
y_test = np.array(test_ds["label"])

print(f"  Train: {len(X_train)} reviews  |  Test: {len(X_test)} reviews")
print(f"  Labels: 0=Negative, 1=Positive")

# ── 2. ML Models with TF-IDF ────────────────────────────────────────────────
print("\n" + "─" * 60)
print("PHASE 1: Training Classical ML Models (TF-IDF features)")
print("─" * 60)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# TF-IDF vectorization (shared across all ML models)
print("\nFitting TF-IDF vectorizer (max 50,000 features, unigrams+bigrams)...")
t0 = time.time()
tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
tfidf_time = time.time() - t0
print(f"  TF-IDF shape: {X_train_tfidf.shape}  ({tfidf_time:.1f}s)")

ml_models = {
    "Naive Bayes": MultinomialNB(alpha=1.0),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=args.seed),
    "SVM (Linear)": LinearSVC(max_iter=2000, C=1.0, random_state=args.seed),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=args.seed, n_jobs=-1),
}

ml_results = {}
for name, model in ml_models.items():
    print(f"\n  Training {name}...", end=" ")
    t0 = time.time()
    model.fit(X_train_tfidf, y_train)
    train_t = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test_tfidf)
    if hasattr(model, "decision_function"):
        # For SVM, use decision_function values as confidence proxy
        conf_raw = model.decision_function(X_test_tfidf)
        # Normalize to 0-1 using sigmoid
        confidence = 1 / (1 + np.exp(-conf_raw))
    elif hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test_tfidf)
        confidence = np.max(proba, axis=1)
    else:
        confidence = np.ones(len(y_pred)) * 0.5
    pred_t = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    ml_results[name] = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "train_time": round(train_t, 2),
        "predict_time": round(pred_t, 2),
        "confusion_matrix": cm.tolist(),
        "type": "ML",
        "features": "TF-IDF (unigrams + bigrams)",
        "predictions": y_pred.tolist(),
        "confidence": [round(c, 4) for c in confidence.tolist()],
    }
    print(f"Accuracy: {acc*100:.1f}%  F1: {f1:.3f}  ({train_t:.1f}s train)")

best_ml = max(ml_results.items(), key=lambda x: x[1]["accuracy"])
print(f"\n  ★ Best ML model: {best_ml[0]} → {best_ml[1]['accuracy']*100:.1f}% accuracy")

# ── 3. Deep Learning: DistilBERT (Pretrained Transformer) ───────────────────
print("\n" + "─" * 60)
print("PHASE 2: Evaluating Deep Learning Model (DistilBERT Transformer)")
print("─" * 60)

import torch
from transformers import pipeline

device_name = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n  Device: {device_name}")
print("  Loading DistilBERT-SST2 (pretrained on sentiment data)...")

dl_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
clf = pipeline("sentiment-analysis", model=dl_model_name, device=-1, truncation=True)

print(f"  Running inference on {len(X_test)} test reviews...")
print("  (This takes a few minutes on CPU — Transformer attention is computing context for every word pair)")

t0 = time.time()
batch_size = 32
dl_preds = []
dl_confs = []
for i in range(0, len(X_test), batch_size):
    batch = X_test[i:i+batch_size]
    results = clf(list(batch), truncation=True, max_length=512)
    for r in results:
        pred = 1 if r["label"] == "POSITIVE" else 0
        dl_preds.append(pred)
        dl_confs.append(round(r["score"], 4))
    if (i // batch_size) % 5 == 0:
        print(f"    Processed {min(i+batch_size, len(X_test))}/{len(X_test)} reviews...")

dl_preds = np.array(dl_preds)
dl_confs = np.array(dl_confs)
dl_time = time.time() - t0

dl_acc = accuracy_score(y_test, dl_preds)
dl_prec = precision_score(y_test, dl_preds)
dl_rec = recall_score(y_test, dl_preds)
dl_f1 = f1_score(y_test, dl_preds)
dl_cm = confusion_matrix(y_test, dl_preds)

dl_result = {
    "accuracy": round(dl_acc, 4),
    "precision": round(dl_prec, 4),
    "recall": round(dl_rec, 4),
    "f1": round(dl_f1, 4),
    "predict_time": round(dl_time, 2),
    "confusion_matrix": dl_cm.tolist(),
    "type": "Deep Learning",
    "model": "DistilBERT (Transformer)",
    "features": "Contextual word embeddings (self-attention)",
    "predictions": dl_preds.tolist(),
    "confidence": dl_confs.tolist(),
}

print(f"\n  ★ DistilBERT: {dl_acc*100:.1f}% accuracy  F1: {dl_f1:.3f}  ({dl_time:.1f}s)")
print(f"  ★ Improvement over best ML: +{(dl_acc - best_ml[1]['accuracy'])*100:.1f}% accuracy")

# ── 4. Challenge Examples: Where DL beats ML ─────────────────────────────────
print("\n" + "─" * 60)
print("PHASE 3: Challenge Examples (Context Understanding)")
print("─" * 60)
print("  Testing examples that require contextual understanding...")

challenges = [
    {"text": "This movie is not good at all.", "true_label": 0, "category": "Negation",
     "explanation": "ML sees 'good' and may predict positive; DL understands 'not good' as negative through attention."},
    {"text": "Oh great, another predictable superhero movie. Just what the world needed.", "true_label": 0, "category": "Sarcasm",
     "explanation": "ML sees 'great' and 'needed' as positive words; DL captures the sarcastic tone from context."},
    {"text": "I can't say I didn't enjoy this film.", "true_label": 1, "category": "Double Negation",
     "explanation": "ML struggles with double negation; DL resolves 'can't + didn't enjoy' = 'did enjoy' through attention layers."},
    {"text": "Despite the stellar cast, the movie fails on every level.", "true_label": 0, "category": "Contrast/But",
     "explanation": "ML picks up 'stellar' as positive; DL understands 'despite X, Y fails' means overall negative."},
    {"text": "The movie was absolutely sick! Best thing I've seen all year!", "true_label": 1, "category": "Slang/Colloquial",
     "explanation": "ML may interpret 'sick' literally as negative; DL understands modern slang from pretraining context."},
    {"text": "It's not the worst movie I've ever seen, but it comes dangerously close.", "true_label": 0, "category": "Subtle Negative",
     "explanation": "ML might focus on 'not worst' as positive; DL captures the overall negative sentiment from the full phrase."},
    {"text": "The acting was phenomenal, the cinematography breathtaking, but the story was so incoherent that none of it mattered.", "true_label": 0, "category": "Mixed (Negative Overall)",
     "explanation": "ML averages positive + negative words; DL understands 'none of it mattered' overrides the positives."},
    {"text": "What started as a dull first act transformed into one of the most gripping thrillers I have ever watched.", "true_label": 1, "category": "Sentiment Shift",
     "explanation": "ML sees both 'dull' and 'gripping'; DL understands the narrative arc — it starts bad but ends great."},
    {"text": "If you enjoy watching paint dry, you'll love this film.", "true_label": 0, "category": "Irony",
     "explanation": "ML detects 'enjoy' and 'love'; DL understands the ironic comparison to paint drying."},
    {"text": "Not bad. Not great either. Just... there.", "true_label": 0, "category": "Understated Negative",
     "explanation": "ML may see 'not bad' and 'great' as mixed positive; DL captures the dismissive, negative tone."},
]

# Predict with best ML model
best_ml_name = best_ml[0]
best_ml_model = ml_models[best_ml_name]
challenge_texts = [c["text"] for c in challenges]
challenge_tfidf = tfidf.transform(challenge_texts)
ml_challenge_preds = best_ml_model.predict(challenge_tfidf)
if hasattr(best_ml_model, "predict_proba"):
    ml_challenge_proba = best_ml_model.predict_proba(challenge_tfidf)
    ml_challenge_conf = np.max(ml_challenge_proba, axis=1)
elif hasattr(best_ml_model, "decision_function"):
    conf_raw = best_ml_model.decision_function(challenge_tfidf)
    ml_challenge_conf = 1 / (1 + np.exp(-np.abs(conf_raw)))
else:
    ml_challenge_conf = np.ones(len(challenges)) * 0.5

# Predict with DL
dl_challenge_results = clf(challenge_texts, truncation=True, max_length=512)
dl_challenge_preds = np.array([1 if r["label"] == "POSITIVE" else 0 for r in dl_challenge_results])
dl_challenge_conf = np.array([r["score"] for r in dl_challenge_results])

challenge_results = []
ml_challenge_correct = 0
dl_challenge_correct = 0
for i, c in enumerate(challenges):
    ml_correct = int(ml_challenge_preds[i]) == c["true_label"]
    dl_correct = int(dl_challenge_preds[i]) == c["true_label"]
    if ml_correct: ml_challenge_correct += 1
    if dl_correct: dl_challenge_correct += 1

    result = {
        "text": c["text"],
        "true_label": c["true_label"],
        "true_sentiment": "Positive" if c["true_label"] == 1 else "Negative",
        "category": c["category"],
        "explanation": c["explanation"],
        "ml_prediction": int(ml_challenge_preds[i]),
        "ml_sentiment": "Positive" if ml_challenge_preds[i] == 1 else "Negative",
        "ml_confidence": round(float(ml_challenge_conf[i]), 4),
        "ml_correct": ml_correct,
        "dl_prediction": int(dl_challenge_preds[i]),
        "dl_sentiment": "Positive" if dl_challenge_preds[i] == 1 else "Negative",
        "dl_confidence": round(float(dl_challenge_conf[i]), 4),
        "dl_correct": dl_correct,
    }
    challenge_results.append(result)

    status_ml = "✓" if ml_correct else "✗"
    status_dl = "✓" if dl_correct else "✗"
    print(f"\n  [{c['category']}] \"{c['text'][:60]}...\"")
    print(f"    True: {'Pos' if c['true_label']==1 else 'Neg'}  |  ML: {'Pos' if ml_challenge_preds[i]==1 else 'Neg'} {status_ml}  |  DL: {'Pos' if dl_challenge_preds[i]==1 else 'Neg'} {status_dl}")

print(f"\n  ★ Challenge scores: ML {ml_challenge_correct}/{len(challenges)}  vs  DL {dl_challenge_correct}/{len(challenges)}")

# ── 5. Save all results ─────────────────────────────────────────────────────
print("\n" + "─" * 60)
print("Saving results...")
print("─" * 60)

# Remove predictions from saved metrics (too large)
ml_save = {}
for name, r in ml_results.items():
    ml_save[name] = {k: v for k, v in r.items() if k not in ("predictions", "confidence")}

dl_save = {k: v for k, v in dl_result.items() if k not in ("predictions", "confidence")}

comparison = {
    "train_samples": args.train,
    "test_samples": args.test,
    "seed": args.seed,
    "ml_models": ml_save,
    "dl_model": dl_save,
    "best_ml": {"name": best_ml_name, **ml_save[best_ml_name]},
    "dl_advantage": round((dl_acc - best_ml[1]["accuracy"]) * 100, 2),
    "challenge_results": challenge_results,
    "challenge_scores": {
        "ml": ml_challenge_correct,
        "dl": dl_challenge_correct,
        "total": len(challenges),
    }
}

json.dump(comparison, open("results/comparison.json", "w"), indent=2)
print("  ✓ results/comparison.json")

# ── 6. Generate comparison chart ─────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Accuracy comparison bar chart
models = list(ml_save.keys()) + ["DistilBERT\n(Transformer)"]
accs = [ml_save[n]["accuracy"] * 100 for n in ml_save] + [dl_acc * 100]
colors = ["#6366f1"] * len(ml_save) + ["#10b981"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, accs, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("Test Accuracy (%)", fontsize=12)
ax.set_title("ML vs Deep Learning: Accuracy on Same IMDB Test Set", fontsize=14, fontweight="bold")
ax.set_ylim(max(0, min(accs) - 10), 100)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/accuracy_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ results/accuracy_comparison.png")

# F1 comparison
f1s = [ml_save[n]["f1"] * 100 for n in ml_save] + [dl_f1 * 100]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(models, f1s, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("F1 Score (%)", fontsize=12)
ax.set_title("ML vs Deep Learning: F1 Score Comparison", fontsize=14, fontweight="bold")
ax.set_ylim(max(0, min(f1s) - 10), 100)
for bar, f1_val in zip(bars, f1s):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{f1_val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/f1_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ results/f1_comparison.png")

# Challenge results chart
fig, ax = plt.subplots(figsize=(8, 4))
categories = ["ML\n(TF-IDF + " + best_ml_name + ")", "Deep Learning\n(DistilBERT)"]
scores = [ml_challenge_correct, dl_challenge_correct]
bar_colors = ["#6366f1", "#10b981"]
bars = ax.bar(categories, scores, color=bar_colors, edgecolor="white", linewidth=1.5, width=0.5)
ax.set_ylabel("Correct Predictions", fontsize=12)
ax.set_title(f"Challenge Examples: Context Understanding ({len(challenges)} tricky reviews)", fontsize=13, fontweight="bold")
ax.set_ylim(0, len(challenges) + 1)
ax.axhline(y=len(challenges), color="gray", linestyle="--", alpha=0.5, label=f"Perfect score ({len(challenges)})")
for bar, s in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.2,
            f'{s}/{len(challenges)}', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("results/challenge_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ results/challenge_comparison.png")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"\n  {'Model':<25} {'Accuracy':>10} {'F1':>10} {'Type':>12}")
print(f"  {'─'*25} {'─'*10} {'─'*10} {'─'*12}")
for name, r in ml_save.items():
    print(f"  {name:<25} {r['accuracy']*100:>9.1f}% {r['f1']*100:>9.1f}% {'ML':>12}")
print(f"  {'DistilBERT (Transformer)':<25} {dl_acc*100:>9.1f}% {dl_f1*100:>9.1f}% {'DL':>12}")
print(f"\n  DL advantage: +{comparison['dl_advantage']:.1f}% accuracy over best ML")
print(f"  Challenge: ML {ml_challenge_correct}/{len(challenges)} vs DL {dl_challenge_correct}/{len(challenges)}")
print(f"\n  All results saved to results/comparison.json")
print(f"  Now run:  streamlit run app.py")
