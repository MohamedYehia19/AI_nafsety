import streamlit as st
import pandas as pd
import joblib
import json
import os
from botg import chat  

# ====================== إعدادات الصفحة ======================
st.set_page_config(
    page_title="شات بوت الصحة النفسية - Hybrid",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 شات بوت الصحة النفسية")
st.markdown("**Hybrid AI** → دردشة طبيعية + استبيانات علمية")
st.caption("لأغراض تعليمية وبحثية فقط - لا يُغني عن استشارة طبيب نفسي")

# ====================== Session State ======================
for key, default in {
    "stage": "chat", "history": [], "messages_display": [],
    "chat_counter": 0, "ai_report": None, "selected_q": None,
    "answers": [], "current_q": 0
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ====================== تحميل الموديلات (التصحيح هنا) ======================
@st.cache_resource
def load_model(prefix):
    # بنجرب الأسماء بالشرطة (-) وبالأندر سكور (_) عشان نضمن إنه يقرأ الملف
    variants = [prefix, prefix.replace("-", "_")]
    
    for v in variants:
        try:
            model_path = f"models/{v}_model.pkl"
            encoder_path = f"models/{v}_encoder.pkl"
            map_path = f"models/{v}_map.json"
            
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                # الـ encoder والـ map مش دايماً موجودين لكل الموديلات، فبنتعامل بحذر
                le = joblib.load(encoder_path) if os.path.exists(encoder_path) else None
                label_map = {}
                if os.path.exists(map_path):
                    with open(map_path, encoding="utf-8") as f:
                        label_map = json.load(f)
                
                return clf, le, label_map
        except Exception as e:
            continue
    return None, None, None

# ====================== بيانات الاستبيانات ======================
QUESTIONNAIRES = {
    "GAD-7": {"name": "القلق العام (GAD-7)", "num": 7, "max": 3, "questions": ["حسيت بالتوتر أو القلق؟", "مقدرتش تسيطر على القلق؟", "قلقت زيادة؟", "صعوبة في الاسترخاء؟", "مضطرب لدرجة عدم الثبات؟", "تتعصب بسرعة؟", "حسيت بالخوف؟"]},
    "PHQ-9": {"name": "الاكتئاب (PHQ-9)", "num": 9, "max": 3, "questions": ["قلة اهتمام بالمتعة؟", "إحباط أو يأس؟", "صعوبة في النوم؟", "تعب وقلة طاقة؟", "شهية متغيرة؟", "شعور بالفشل؟", "صعوبة تركيز؟", "بطء أو سرعة حركية؟", "أفكار انتحارية؟"]},
    "PSS-10": {"name": "التوتر (PSS-10)", "num": 10, "max": 4, "questions": ["انزعاج لحدث مفاجئ؟", "فقدان سيطرة؟", "توتر وعصبية؟", "تعامل ناجح مع الضغوط؟", "تكيف مع التغييرات؟", "ثقة في حل المشاكل؟", "الأمور سارت كما تحب؟", "عدم قدرة على التعامل؟", "سيطرة على الوقت؟", "تراكم الصعوبات؟"]},
    "Y-BOCS": {"name": "الوسواس (Y-BOCS)", "num": 10, "max": 4, "questions": ["وقت الوساوس؟", "إعاقة الحياة؟", "الضيق الناتج؟", "مقاومة الوساوس؟", "السيطرة عليها؟", "وقت الأفعال القهرية؟", "إعاقتها للحياة؟", "الضيق الناتج عنها؟", "مقاومتها؟", "السيطرة عليها؟"]},
    "MDQ": {"name": "ثنائي القطب (MDQ)", "num": 13, "max": 1, "questions": ["نشاط مفرط؟", "عصبية زائدة؟", "ثقة مفرطة؟", "نوم قليل؟", "كلام كثير؟", "أفكار متسارعة؟", "تشتت؟", "زيادة طاقة اجتماعية؟", "تهور؟", "وضوح التغيير؟", "مشاكل؟", "حدثت معاً؟", "شدة التأثير؟"]}
}

# ====================== عرض الرسائل ======================
for msg in st.session_state.messages_display:
    with st.chat_message(msg["role"]): st.markdown(msg["content"])

# ====================== مراحل التطبيق ======================
if st.session_state.stage == "chat":
    if st.session_state.chat_counter >= 6:
        st.session_state.stage = "transition"
        st.rerun()
    user_input = st.chat_input("اكتب رسالتك هنا...")
    if user_input:
        st.session_state.chat_counter += 1
        with st.chat_message("user"): st.markdown(user_input)
        st.session_state.messages_display.append({"role": "user", "content": user_input})
        with st.spinner("جاري التفكير..."):
            result = chat(st.session_state.history, user_input)
        st.session_state.history = result["history"]
        with st.chat_message("assistant"): st.markdown(result["reply"])
        st.session_state.messages_display.append({"role": "assistant", "content": result["reply"]})
        if result.get("report") or st.session_state.chat_counter >= 6:
            st.session_state.ai_report = result.get("report")
            st.session_state.stage = "transition"
        st.rerun()

elif st.session_state.stage == "transition":
    st.success("شكراً على حديثك معايا ❤️")
    if st.button("ابدأ الاستبيان الآن", type="primary"):
        st.session_state.stage = "select"
        st.rerun()

elif st.session_state.stage == "select":
    st.subheader("اختر الاستبيان:")
    cols = st.columns(2)
    for i, (key, info) in enumerate(QUESTIONNAIRES.items()):
        with cols[i % 2]:
            if st.button(info["name"], use_container_width=True):
                st.session_state.selected_q, st.session_state.answers, st.session_state.current_q, st.session_state.stage = key, [], 0, "questionnaire"
                st.rerun()

elif st.session_state.stage == "questionnaire":
    q = QUESTIONNAIRES[st.session_state.selected_q]
    idx = st.session_state.current_q
    st.subheader(f"{q['name']} — سؤال {idx+1}")
    st.write(q["questions"][idx])
    opts = ["0 - لا", "1 - نعم"] if q["max"]==1 else ["0 - أبداً", "1 - قليلاً", "2 - أحياناً", "3 - غالباً", "4 - دائماً"]
    choice = st.radio("الإجابة:", opts, key=f"q{idx}")
    if st.button("التالي"):
        st.session_state.answers.append(int(choice[0]))
        st.session_state.current_q += 1
        if st.session_state.current_q >= q["num"]: st.session_state.stage = "done"
        st.rerun()

elif st.session_state.stage == "done":
    q = QUESTIONNAIRES[st.session_state.selected_q]
    ans_copy = st.session_state.answers.copy()
    
    # معالجة PSS-10 للأسئلة العكسية
    if st.session_state.selected_q == "PSS-10":
        for i in [3, 4, 5, 6, 8]:
            if i < len(ans_copy): ans_copy[i] = 4 - ans_copy[i]

    total = sum(ans_copy)
    clf, le, label_map = load_model(st.session_state.selected_q)
    
    model_pred = "غير متاح"
    if clf:
        try:
            df = pd.DataFrame([ans_copy], columns=[f"Q{i+1}" for i in range(len(ans_copy))])
            pred_enc = clf.predict(df)[0]
            # محاولة جلب الاسم من الخريطة، لو فشل جرب الـ LabelEncoder، لو فشل اعرض الرقم
            model_pred = label_map.get(str(pred_enc), label_map.get(int(pred_enc), str(pred_enc)))
            if le and model_pred == str(pred_enc):
                model_pred = le.inverse_transform([pred_enc])[0]
        except Exception as e:
            model_pred = f"خطأ: {str(e)}"

    st.success(f"✅ نتيجة **{q['name']}**")
    st.metric("المجموع", f"{total} / {q['num']*q['max']}")
    st.info(f"**تصنيف الموديل:** {model_pred}")
    
    if st.session_state.ai_report:
        st.divider()
        st.subheader("📋 تقرير الذكاء الاصطناعي")
        st.json(st.session_state.ai_report)

    if st.button("🔄 محادثة جديدة"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()