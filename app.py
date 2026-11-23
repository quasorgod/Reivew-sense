"""
ReviewSense — Deep Learning Sentiment Analysis with ML vs DL Comparison.
Streamlit dashboard showcasing why Transformers beat classical ML for NLP.

Run:  streamlit run app.py
"""
import json, os, time
import numpy as np
import pandas as pd
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG + CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ReviewSense — DL Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* Global */
.stApp { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #1e1b4b 0%, #312e81 100%); }
[data-testid="stSidebar"] * { color: #e0e7ff !important; }
[data-testid="stSidebar"] .stMarkdown a { color: #a5b4fc !important; }

/* Hero section */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 25px rgba(79, 70, 229, 0.3);
}
.hero h1 { margin: 0; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.5px; }
.hero p { margin: 0.5rem 0 0; font-size: 1.05rem; opacity: 0.9; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #f8fafc, #f1f5f9);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.metric-card.green { border-left: 4px solid #10b981; }
.metric-card.purple { border-left: 4px solid #8b5cf6; }
.metric-card.blue { border-left: 4px solid #3b82f6; }
.metric-card.amber { border-left: 4px solid #f59e0b; }
.metric-label { font-size: 0.8rem; color: #64748b; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.metric-value { font-size: 2rem; font-weight: 800; color: #1e293b; margin: 0.3rem 0; }
.metric-sub { font-size: 0.8rem; color: #94a3b8; }

/* Comparison cards */
.vs-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}
.ml-tag { background: #ede9fe; color: #6d28d9; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }
.dl-tag { background: #d1fae5; color: #047857; padding: 3px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }

/* Challenge cards */
.challenge-card {
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 4px solid;
}
.challenge-card.dl-wins { background: #f0fdf4; border-color: #10b981; }
.challenge-card.ml-wins { background: #fef3f2; border-color: #ef4444; }
.challenge-card.both-right { background: #f0f9ff; border-color: #3b82f6; }
.challenge-card.both-wrong { background: #fefce8; border-color: #eab308; }

/* Info boxes */
.info-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border-left: 4px solid #3b82f6;
    border-radius: 0 10px 10px 0;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
.info-box.green { background: linear-gradient(135deg, #f0fdf4, #dcfce7); border-color: #10b981; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 20px;
    font-weight: 600;
}

/* Table improvements */
.dataframe { font-size: 0.9rem; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 ReviewSense")
    st.markdown("**Deep Learning Sentiment Analysis**")
    st.markdown("---")
    st.markdown("""
    **Project Highlights:**
    - 🏆 DL vs ML head-to-head comparison
    - 🔬 DistilBERT Transformer model
    - 🎯 Context-aware predictions
    - 📊 Real IMDB movie reviews
    """)
    st.markdown("---")
    st.markdown("**Syllabus Coverage:**")
    st.markdown("""
    ✅ AI, ML & DL Fundamentals
    ✅ Transformers & LLMs
    ✅ Generative AI Evolution
    ✅ Industry Applications
    ✅ Challenges & Limitations
    """)
    st.markdown("---")
    st.markdown("**Tech Stack:**")
    st.markdown("🐍 Python · 🤗 HuggingFace · 🔥 PyTorch · 📊 Streamlit · 🧮 scikit-learn")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
RES = "results"

def load_comparison():
    path = os.path.join(RES, "comparison.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None

@st.cache_resource(show_spinner="Loading DistilBERT model (first time only)...")
def load_model():
    try:
        from transformers import pipeline
        return pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english",
                        device=-1, truncation=True)
    except Exception as e:
        st.warning(f"Could not load DistilBERT model: {e}")
        return None

def predict(clf, text):
    if clf is None:
        return "Positive", 0.5  # fallback
    r = clf([text], truncation=True, max_length=512)[0]
    label = "Positive" if r["label"] == "POSITIVE" else "Negative"
    return label, float(r["score"])

comp = load_comparison()

# ═══════════════════════════════════════════════════════════════════════════════
# HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <h1>🧠 ReviewSense</h1>
    <p>Deep Learning Sentiment Analysis — proving why Transformers outperform classical ML on nuanced text understanding</p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_overview, tab_battle, tab_demo, tab_results, tab_theory, tab_genai = st.tabs([
    "🏠 Overview",
    "⚔️ ML vs DL Battle",
    "🔍 Live Demo",
    "📊 Training Results",
    "🧠 Why Deep Learning?",
    "🌍 GenAI & Industry",
])

# ═════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ═════════════════════════════════════════════════════════════════════════
with tab_overview:
    if comp:
        dl = comp["dl_model"]
        best_ml = comp["best_ml"]
        chall = comp["challenge_scores"]

        st.markdown("### Key Results at a Glance")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-label">DL Accuracy</div>
                <div class="metric-value" style="color:#10b981">{dl['accuracy']*100:.1f}%</div>
                <div class="metric-sub">DistilBERT Transformer</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card purple">
                <div class="metric-label">Best ML Accuracy</div>
                <div class="metric-value" style="color:#8b5cf6">{best_ml['accuracy']*100:.1f}%</div>
                <div class="metric-sub">{best_ml['name']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-label">DL Advantage</div>
                <div class="metric-value" style="color:#3b82f6">+{comp['dl_advantage']:.1f}%</div>
                <div class="metric-sub">Higher accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card amber">
                <div class="metric-label">Challenge Score</div>
                <div class="metric-value" style="color:#f59e0b">{chall['dl']}/{chall['total']}</div>
                <div class="metric-sub">DL context understanding</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Why This Project Uses Deep Learning")
        st.markdown("""
        <div class="info-box green">
            <strong>The core argument:</strong> For simple bag-of-words sentiment ("good movie" vs "bad movie"), ML works fine.
            But real reviews contain <strong>negation</strong> ("not good"), <strong>sarcasm</strong> ("oh great, another sequel"),
            <strong>contrast</strong> ("great cast, terrible script"), and <strong>sentiment shifts</strong>.
            Only Deep Learning with <strong>self-attention</strong> (Transformers) can understand these patterns —
            and that's exactly what real-world NLP demands.
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🤖 What ML Does")
            st.markdown("""
            - Converts text to **TF-IDF** (word frequency vectors)
            - Each word treated **independently** — no word order
            - "not good" → sees "not" and "good" separately
            - Cannot understand context, sarcasm, or irony
            - **Fast to train**, decent on simple text
            """)
        with col2:
            st.markdown("#### 🧠 What DL Does (Transformers)")
            st.markdown("""
            - Uses **self-attention** to relate every word to every other
            - Understands "not good" ≠ "good" through attention weights
            - **Pretrained on billions of words** (transfer learning)
            - Captures context, negation, sarcasm, tone shifts
            - **Foundation of modern NLP** (GPT, BERT, Claude, Gemini)
            """)
    else:
        st.warning("⚠️ Run `python -I train_comparison.py` first to generate the ML vs DL comparison metrics.")
        st.info("The app's Live Demo tab works without training — it uses the pretrained DistilBERT model.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 2: ML vs DL BATTLE
# ═════════════════════════════════════════════════════════════════════════
with tab_battle:
    if comp:
        st.markdown("### ⚔️ Head-to-Head: ML vs Deep Learning")
        st.markdown(f"*Tested on the same {comp['test_samples']} IMDB movie reviews (trained on {comp['train_samples']})*")

        # ── Accuracy comparison chart ──
        st.markdown("#### 📊 Accuracy Comparison")
        img_path = os.path.join(RES, "accuracy_comparison.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)

        # ── Detailed metrics table ──
        st.markdown("#### 📋 Full Metrics Table")
        rows = []
        for name, m in comp["ml_models"].items():
            rows.append({
                "Model": name,
                "Type": "🤖 ML",
                "Accuracy (%)": f"{m['accuracy']*100:.1f}",
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1 Score": f"{m['f1']:.3f}",
                "Features": m.get("features", "TF-IDF"),
            })
        dl = comp["dl_model"]
        rows.append({
            "Model": "DistilBERT",
            "Type": "🧠 DL",
            "Accuracy (%)": f"{dl['accuracy']*100:.1f}",
            "Precision": f"{dl['precision']:.3f}",
            "Recall": f"{dl['recall']:.3f}",
            "F1 Score": f"{dl['f1']:.3f}",
            "Features": "Self-attention embeddings",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # ── F1 comparison ──
        st.markdown("#### 📈 F1 Score Comparison")
        f1_path = os.path.join(RES, "f1_comparison.png")
        if os.path.exists(f1_path):
            st.image(f1_path, use_container_width=True)

        st.markdown("---")

        # ── Challenge examples ──
        st.markdown("### 🧪 Challenge Examples: Where Context Matters")
        st.markdown("""
        These are **tricky reviews** that require understanding context — negation, sarcasm, contrast, irony.
        This is where the difference between ML and DL becomes clear.
        """)

        chall = comp["challenge_scores"]
        c1, c2 = st.columns(2)
        with c1:
            st.metric("🤖 ML Score", f"{chall['ml']}/{chall['total']}",
                      delta=f"{chall['ml']/chall['total']*100:.0f}%")
        with c2:
            st.metric("🧠 DL Score", f"{chall['dl']}/{chall['total']}",
                      delta=f"{chall['dl']/chall['total']*100:.0f}%")

        chall_img = os.path.join(RES, "challenge_comparison.png")
        if os.path.exists(chall_img):
            st.image(chall_img, use_container_width=True)

        for cr in comp.get("challenge_results", []):
            ml_ok = cr["ml_correct"]
            dl_ok = cr["dl_correct"]
            if dl_ok and not ml_ok:
                cls = "dl-wins"
                badge = "🧠 DL Wins"
            elif ml_ok and not dl_ok:
                cls = "ml-wins"
                badge = "🤖 ML Wins"
            elif ml_ok and dl_ok:
                cls = "both-right"
                badge = "✅ Both Correct"
            else:
                cls = "both-wrong"
                badge = "⚠️ Both Wrong"

            st.markdown(f"""
            <div class="challenge-card {cls}">
                <strong>[{cr['category']}]</strong> {badge}<br>
                <em>"{cr['text']}"</em><br>
                <span style="color:#666">True: <strong>{cr['true_sentiment']}</strong> &nbsp;|&nbsp;
                ML: <strong>{cr['ml_sentiment']}</strong> ({cr['ml_confidence']*100:.0f}%) &nbsp;|&nbsp;
                DL: <strong>{cr['dl_sentiment']}</strong> ({cr['dl_confidence']*100:.0f}%)</span><br>
                <small style="color:#888">💡 {cr['explanation']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box green">
            <strong>Key Insight:</strong> ML models treat text as a "bag of words" — they lose word order and context.
            Transformers use <strong>self-attention</strong> to understand relationships between ALL words simultaneously,
            enabling them to handle negation, sarcasm, contrast, and sentiment shifts that ML fundamentally cannot.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Run `python -I train_comparison.py` first to see the comparison.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 3: LIVE DEMO
# ═════════════════════════════════════════════════════════════════════════
with tab_demo:
    st.markdown("### 🔍 Try It Yourself")
    st.markdown("Type any movie review and watch the Transformer analyze its sentiment in real time.")

    examples = {
        "(write your own)": "",
        "✅ Positive — simple": "An absolute masterpiece. The acting was brilliant and the story kept me hooked till the end.",
        "❌ Negative — simple": "Painfully boring. The plot made no sense and I nearly walked out of the theatre.",
        "🧪 Negation test": "This movie is not good at all. The acting was not convincing and the story is not engaging.",
        "🧪 Sarcasm test": "Oh wonderful, another superhero movie with the exact same plot. Just what cinema needed.",
        "🧪 Double negation": "I can't say I didn't enjoy this film. It wasn't unpleasant to watch.",
        "🧪 Contrast test": "Despite the stellar cast and beautiful cinematography, the movie fails on every level.",
        "🧪 Sentiment shift": "What started as a dull first act transformed into one of the most gripping thrillers I have ever watched.",
        "🧪 Irony test": "If you enjoy watching paint dry, you'll absolutely love this two-hour masterpiece.",
        "🧪 Mixed review": "The visuals were stunning, but the story was predictable and the ending felt rushed.",
    }

    choice = st.selectbox("Pick an example or write your own:", list(examples.keys()))
    text = st.text_area("Movie review:", value=examples.get(choice, ""), height=120,
                        placeholder="Type your movie review here...")

    if st.button("🧠 Analyse with Deep Learning", type="primary") and text.strip():
        with st.spinner("DistilBERT is processing... (self-attention across all word pairs)"):
            clf = load_model()
            start = time.time()
            label, conf = predict(clf, text)
            elapsed = time.time() - start

        emoji = "😀" if label == "Positive" else "😞"
        color = "#10b981" if label == "Positive" else "#ef4444"

        c1, c2, c3 = st.columns(3)
        c1.markdown(f"""
        <div class="metric-card {'green' if label=='Positive' else 'purple'}">
            <div class="metric-label">Prediction</div>
            <div class="metric-value" style="color:{color}">{emoji} {label}</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""
        <div class="metric-card blue">
            <div class="metric-label">Confidence</div>
            <div class="metric-value">{conf*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        c3.markdown(f"""
        <div class="metric-card amber">
            <div class="metric-label">Inference Time</div>
            <div class="metric-value">{elapsed*1000:.0f}ms</div>
        </div>""", unsafe_allow_html=True)

        # Confidence bar
        pos_score = conf if label == "Positive" else 1 - conf
        st.progress(pos_score, text="Negative ← model's leaning → Positive")

        st.markdown("""
        <div class="info-box">
            <strong>What just happened:</strong> The review was tokenized into subwords, each was converted to a 768-dimensional
            embedding, then passed through 6 Transformer layers where self-attention computed relevance between every pair of tokens.
            The final [CLS] token representation was fed to a classifier to produce the sentiment prediction.
        </div>
        """, unsafe_allow_html=True)

    # ── Batch analysis ──
    st.markdown("---")
    st.markdown("### 📁 Batch Analysis")
    st.markdown("Upload a CSV with a `review` column, or paste reviews (one per line).")

    up = st.file_uploader("Upload CSV:", type="csv")
    lines = st.text_area("...or paste reviews (one per line):", height=100)
    texts = None
    if up is not None:
        df = pd.read_csv(up)
        col = "review" if "review" in df.columns else df.columns[0]
        texts = df[col].astype(str).tolist()[:200]
    elif lines.strip():
        texts = [l for l in lines.splitlines() if l.strip()][:200]

    if texts and st.button("Analyse batch"):
        clf = load_model()
        with st.spinner(f"Analysing {len(texts)} reviews..."):
            results = clf(texts, truncation=True, max_length=512)
        out = pd.DataFrame({
            "Review": [t[:120] for t in texts],
            "Sentiment": ["Positive" if r["label"]=="POSITIVE" else "Negative" for r in results],
            "Confidence": [f"{r['score']*100:.1f}%" for r in results],
        })
        st.dataframe(out, use_container_width=True)
        pos_count = sum(1 for r in results if r["label"]=="POSITIVE")
        neg_count = len(results) - pos_count
        c1, c2 = st.columns(2)
        c1.metric("😀 Positive", pos_count)
        c2.metric("😞 Negative", neg_count)
        st.download_button("Download results", out.to_csv(index=False), "reviewsense_results.csv")

# ═════════════════════════════════════════════════════════════════════════
# TAB 4: TRAINING RESULTS
# ═════════════════════════════════════════════════════════════════════════
with tab_results:
    if comp:
        st.markdown("### 📊 Detailed Training & Evaluation Results")
        st.markdown(f"*{comp['train_samples']} training reviews, {comp['test_samples']} test reviews from IMDB*")

        # ML models detail
        st.markdown("#### 🤖 Machine Learning Models (TF-IDF Features)")
        for name, m in comp["ml_models"].items():
            with st.expander(f"{name} — Accuracy: {m['accuracy']*100:.1f}%"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
                c2.metric("Precision", f"{m['precision']:.3f}")
                c3.metric("Recall", f"{m['recall']:.3f}")
                c4.metric("F1 Score", f"{m['f1']:.3f}")
                if "confusion_matrix" in m:
                    cm = np.array(m["confusion_matrix"])
                    st.markdown(f"**Confusion Matrix:** TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
                st.caption(f"Features: {m.get('features', 'TF-IDF')} | Train time: {m.get('train_time', 'N/A')}s")

        # DL model detail
        st.markdown("#### 🧠 Deep Learning Model (DistilBERT Transformer)")
        dl = comp["dl_model"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{dl['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{dl['precision']:.3f}")
        c3.metric("Recall", f"{dl['recall']:.3f}")
        c4.metric("F1 Score", f"{dl['f1']:.3f}")

        if "confusion_matrix" in dl:
            cm = np.array(dl["confusion_matrix"])
            st.markdown(f"**Confusion Matrix:** TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
        st.caption(f"Model: {dl.get('model', 'DistilBERT')} | Features: {dl.get('features', 'Self-attention')} | Inference time: {dl.get('predict_time', 'N/A')}s")

        st.markdown("""
        <div class="info-box">
            <strong>Note on training:</strong> ML models were trained from scratch on the IMDB subset.
            DistilBERT uses <strong>transfer learning</strong> — it was pretrained on a massive text corpus (Wikipedia + BookCorpus)
            and then fine-tuned on the SST-2 sentiment dataset. This is why DL excels: it brings world knowledge into the task,
            rather than learning only from the small training set.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Run `python -I train_comparison.py` first to see detailed results.")

# ═════════════════════════════════════════════════════════════════════════
# TAB 5: WHY DEEP LEARNING?
# ═════════════════════════════════════════════════════════════════════════
with tab_theory:
    st.markdown("### 🧠 Why Deep Learning for Sentiment Analysis?")

    st.markdown("#### The Fundamental Problem with ML for NLP")
    st.markdown("""
    Classical ML models like Naive Bayes, Logistic Regression, and SVM use **TF-IDF** or **Bag-of-Words**
    to represent text. This means:

    1. **No word order**: "The movie was not good" and "The good movie was not" have the **same** representation
    2. **No context**: The word "good" always has the same feature value, whether preceded by "not" or "very"
    3. **No semantics**: "brilliant" and "outstanding" are treated as completely unrelated features
    4. **Fixed vocabulary**: New words or slang not seen during training are ignored
    """)

    st.markdown("#### How Transformers Solve This")
    st.markdown("""
    The **Transformer architecture** (Vaswani et al., 2017) introduced **self-attention**, which lets the model
    compute relationships between every pair of words in a sentence simultaneously.
    """)

    st.markdown("##### Self-Attention Mechanism")
    st.markdown("""
    For each word, the model computes:
    - **Query (Q)**: "What am I looking for?"
    - **Key (K)**: "What do I contain?"
    - **Value (V)**: "What information do I carry?"

    **Attention(Q, K, V) = softmax(QK^T / √d_k) × V**

    This means when processing "not" in "not good", the model learns that "not" strongly attends to "good"
    and flips its meaning — something impossible with TF-IDF.
    """)

    st.markdown("##### Transfer Learning: The DL Superpower")
    st.markdown("""
    1. **Pretraining**: DistilBERT is trained on **~3.3 billion words** (English Wikipedia + BookCorpus)
       using masked language modeling — it learns grammar, semantics, world knowledge
    2. **Fine-tuning**: We add a small classification layer and train on our specific task (sentiment)
       with just a few thousand examples
    3. **Result**: The model brings vast language understanding to our small task — ML models can only learn
       from the training data we provide
    """)

    st.markdown("#### DistilBERT Architecture")
    st.markdown("""
    | Component | Details |
    |-----------|---------|
    | Base model | BERT (distilled) |
    | Layers | 6 Transformer blocks |
    | Hidden size | 768 dimensions |
    | Attention heads | 12 per layer |
    | Parameters | ~66 million |
    | Pretraining data | English Wikipedia + BookCorpus |
    | Compression | 40% smaller, 60% faster than BERT, retains 97% performance |
    """)

    st.markdown("#### 5 Reasons DL Beats ML for Text Analysis")

    reasons = [
        ("1. Contextual Understanding",
         "Self-attention lets the model understand that 'not good' is negative, 'not bad' is mildly positive, and 'not the worst' is lukewarm — ML treats all three similarly because it sees individual words."),
        ("2. Transfer Learning",
         "Pretrained on billions of words, DL models bring world knowledge to any task. ML models start from scratch every time and can only learn patterns in the training data provided."),
        ("3. Semantic Similarity",
         "DL embeddings place 'brilliant', 'outstanding', and 'excellent' close together in vector space. TF-IDF treats them as completely independent features with no relationship."),
        ("4. Handling Ambiguity",
         "The word 'sick' can mean 'ill' or 'awesome' depending on context. Self-attention resolves this by looking at surrounding words. ML has no mechanism for disambiguation."),
        ("5. Scalability & Generalization",
         "Fine-tuned DL models generalize to new domains (product reviews, tweets, news) with minimal retraining. ML models need entirely new TF-IDF vocabularies and retraining for each domain."),
    ]
    for title, desc in reasons:
        st.markdown(f"**{title}**")
        st.markdown(desc)

    st.markdown("""
    <div class="info-box green">
        <strong>Bottom line:</strong> ML is sufficient for simple, keyword-based sentiment where "good" = positive and "bad" = negative.
        But real-world text is full of nuance — negation, sarcasm, irony, contrast, slang — and only Deep Learning
        with self-attention can capture these patterns. This is why every modern NLP system (GPT, BERT, Claude, Gemini)
        is built on Transformers, not on TF-IDF + Logistic Regression.
    </div>
    """, unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════
# TAB 6: GenAI & INDUSTRY APPLICATIONS
# ═════════════════════════════════════════════════════════════════════════
with tab_genai:
    st.markdown("### 🌍 Generative AI, Transformers & Industry Applications")

    st.markdown("#### Evolution of Generative AI")
    st.markdown("""
    | Era | Technology | Capability | Example |
    |-----|-----------|------------|---------|
    | Pre-2017 | RNNs, LSTMs | Sequential text generation | Basic chatbots |
    | 2017 | Transformer (Vaswani et al.) | Parallel attention mechanism | Machine translation |
    | 2018 | BERT (Google) | Bidirectional understanding | Sentiment analysis, QA |
    | 2018-19 | GPT-1, GPT-2 (OpenAI) | Autoregressive text generation | Text completion |
    | 2020 | GPT-3 | Few-shot learning, 175B params | Versatile text generation |
    | 2021 | DALL-E, Codex | Image generation, code generation | Creative AI |
    | 2022 | ChatGPT, Stable Diffusion | Conversational AI, open-source image gen | Consumer AI revolution |
    | 2023 | GPT-4, Claude, Gemini, LLaMA | Multimodal, reasoning, long-context | Enterprise AI adoption |
    | 2024-25 | Claude 3.5, Gemini 2.0, GPT-4o | Agents, tools, real-time multimodal | Autonomous AI workflows |
    """)

    st.markdown("#### Key Generative AI Models")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📝 GANs (Generative Adversarial Networks)")
        st.markdown("""
        - **Architecture**: Generator vs. Discriminator (adversarial training)
        - **Use cases**: Image generation, deepfakes, data augmentation, style transfer
        - **Limitation**: Training instability, mode collapse
        - **Indian example**: ISRO uses GANs for satellite image super-resolution
        """)

        st.markdown("##### 🔄 Transformers")
        st.markdown("""
        - **Core innovation**: Self-attention mechanism (parallel, not sequential)
        - **Foundation for**: BERT, GPT, T5, DistilBERT (used in this project)
        - **Why revolutionary**: O(1) path length between any two words vs O(n) for RNNs
        - **Our project uses DistilBERT** — a Transformer for sentiment classification
        """)

    with col2:
        st.markdown("##### 🗣️ Large Language Models (LLMs)")
        st.markdown("""
        - **GPT (OpenAI)**: Autoregressive, excels at generation (ChatGPT, GPT-4)
        - **BERT (Google)**: Bidirectional, excels at understanding (search, classification)
        - **Claude (Anthropic)**: Safety-focused, long-context reasoning
        - **Gemini (Google)**: Multimodal (text + image + audio + video)
        - **LLaMA (Meta)**: Open-source, enables research community
        """)

        st.markdown("##### 🧠 How LLMs Relate to This Project")
        st.markdown("""
        - DistilBERT is a **distilled Transformer** — same architecture family as GPT/BERT
        - Our sentiment analysis is a **classification** task (BERT-style)
        - LLMs can do sentiment analysis **zero-shot** (no training needed!)
        - But fine-tuned smaller models like DistilBERT are more efficient for specific tasks
        """)

    st.markdown("---")
    st.markdown("#### 🏭 Industry Applications of Deep Learning NLP")

    st.markdown("##### 🇮🇳 India")
    st.markdown("""
    | Company/Org | Application | DL Technology |
    |-------------|-------------|---------------|
    | **Flipkart** | Product review sentiment analysis, fake review detection | BERT-based classifiers |
    | **Jio (Reliance)** | Multilingual chatbot (JioChat), content moderation | Transformer LLMs |
    | **HDFC Bank** | Customer feedback analysis, complaint routing | Fine-tuned BERT |
    | **Swiggy/Zomato** | Restaurant review analysis, delivery feedback | Sentiment Transformers |
    | **TCS / Infosys** | Enterprise NLP services for global clients | Custom LLM pipelines |
    | **Govt. of India** | Bhashini (language translation platform for 22 Indian languages) | Multilingual Transformers |
    """)

    st.markdown("##### 🌐 Global")
    st.markdown("""
    | Company | Application | DL Technology |
    |---------|-------------|---------------|
    | **Netflix** | Content recommendation + review analysis | Deep collaborative filtering + NLP |
    | **Amazon** | Product sentiment, review summarization, Alexa NLU | BERT + custom Transformers |
    | **Google** | Search ranking (BERT), Gmail Smart Reply, Translate | BERT, T5, Gemini |
    | **X (Twitter)** | Hate speech detection, trending topic analysis | Fine-tuned LLMs |
    | **JPMorgan** | Financial sentiment from news, SEC filing analysis | FinBERT |
    | **Healthcare** | Clinical note analysis, drug review sentiment | BioBERT, PubMedBERT |
    """)

    st.markdown("---")
    st.markdown("#### ⚠️ Challenges and Limitations of Generative AI")

    challenges = [
        ("🔴 Hallucination", "LLMs can generate plausible but factually incorrect information. This is a fundamental issue with autoregressive generation — the model optimizes for fluency, not truth."),
        ("🟡 Bias & Fairness", "Models inherit biases from training data. Sentiment models may rate reviews differently based on dialect, cultural references, or slang patterns."),
        ("🟡 Computational Cost", "Training GPT-4 cost estimated $100M+. Even inference with large models requires significant GPU resources. DistilBERT is a response to this — 40% smaller, 60% faster."),
        ("🟡 Data Privacy", "Models trained on user data raise privacy concerns. India's Digital Personal Data Protection Act (2023) adds regulatory requirements."),
        ("🔴 Interpretability", "Deep learning models are 'black boxes' — it's hard to explain WHY a prediction was made, unlike simpler ML models where feature importance is clear."),
        ("🟡 Environmental Impact", "Training large models has significant carbon footprint. Distillation (like DistilBERT) and efficient architectures help mitigate this."),
        ("🟡 Adversarial Vulnerability", "Small perturbations in input can fool DL models. Adversarial attacks on sentiment models can flip predictions with minor text changes."),
    ]
    for title, desc in challenges:
        st.markdown(f"**{title}**: {desc}")

    st.markdown("""
    <div class="info-box">
        <strong>Business Relevance:</strong> Despite these challenges, the global NLP market is projected to reach $112B by 2030.
        Every major tech company and many Indian firms are investing heavily in Transformer-based NLP.
        Understanding these models — from fine-tuning DistilBERT (as in this project) to deploying LLMs —
        is a critical skill for AI practitioners.
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem 0;">
    <strong>ReviewSense</strong> — Deep Learning Sentiment Analysis Project<br>
    Built with 🤗 HuggingFace Transformers · 🔥 PyTorch · 📊 Streamlit · 🧮 scikit-learn<br>
    Demonstrating why Transformers outperform classical ML for nuanced text understanding
</div>
""", unsafe_allow_html=True)
