# Athlete Readiness Assistant (minimum MVP for Gate 5)
# Run it with:  streamlit run app.py
# Needs these files in the same folder:
#   athlete_recovery_synthetic.csv and readiness_model.joblib

import json
import re
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Athlete Readiness", layout="wide")


# ---------------------------------------------------------------
# Part A: Load the data and the saved model
# ---------------------------------------------------------------
def prepare(df):
    # Same new columns as the notebooks
    df = df.copy()
    df["Sleep_Missing"] = df["Sleep_Duration_Hours"].isna().astype(int)
    df["Stress_Num"] = df["Stress_Level"].map({"Low": 1, "Medium": 2, "High": 3})
    df["Training_Load"] = df["Training_Duration_Min"] * df["Training_Intensity"]
    return df


@st.cache_data
def load_demo_data():
    df = prepare(pd.read_csv("athlete_recovery_synthetic.csv"))

    # Same 60 test athletes as in the notebooks (the model never saw them)
    athlete_ids = df["Athlete_ID"].unique()
    train_ids, test_ids = train_test_split(athlete_ids, test_size=0.2, random_state=42)
    return df[df["Athlete_ID"].isin(test_ids)].copy()


@st.cache_resource
def load_model():
    return joblib.load("readiness_model.joblib")


saved = load_model()
model = saved["model"]
features = saved["features"]
medians = saved["medians"]

# Weights and team averages from the model (used to find the top reasons)
scaler = model.named_steps["scale"]
logreg = model.named_steps["logreg"]
weights = pd.Series(logreg.coef_[list(logreg.classes_).index("Red")], index=features)
team_average = pd.Series(scaler.mean_, index=features)


# ---------------------------------------------------------------
# Part B: The note writer (same code as the Notes notebook)
# ---------------------------------------------------------------
bands = {"Green": "75-100", "Orange": "40-74", "Red": "0-39"}

action_by_tier = {
    "Green": "Full training load and match readiness.",
    "Orange": "Modified training volume and targeted recovery; monitor closely.",
    "Red": "Active rest and a check with the medical or athletic therapy staff.",
}

banned_words = ["diagnosis", "diagnose", "injury", "injured", "torn", "fracture", "medication", "sprain"]

system_prompt = (
    "You write short readiness notes for athletic coaching staff. "
    "Start the summary with the athlete and the tier, like: A1005 is in the Red tier. "
    "Then give the reasons. "
    "Use only the tier and reasons in the input. "
    "Do not add numbers or facts that are not in the input. "
    "Do not give medical diagnoses or mention injuries. "
    "For a Red tier, tell the coach to check with the medical or athletic therapy staff. "
    "Reply in JSON with two fields: summary (1 or 2 sentences) and action (1 sentence). "
    "Keep it under 60 words."
)


def fallback_note(case):
    if case["drivers"]:
        reasons = "Main reasons: " + "; ".join(case["drivers"]) + "."
    else:
        reasons = "No single clear reason."
    return {
        "summary": f"{case['athlete']} is in the {case['tier']} tier. {reasons}",
        "action": action_by_tier[case["tier"]],
    }


def has_word(text, word):
    return re.search(r"\b" + word.lower() + r"\b", text.lower()) is not None


def check_note(note, case):
    # The 6 rules from the task card. Empty list = the note is fine
    if not isinstance(note, dict) or "summary" not in note or "action" not in note:
        return ["missing_part"]

    text = note["summary"] + " " + note["action"]
    problems = []

    if len(text.split()) > 60:
        problems.append("too_long")
    if not has_word(text, case["tier"]):
        problems.append("tier_missing")
    for other in ["Green", "Orange", "Red"]:
        if other != case["tier"] and has_word(text, other):
            problems.append("wrong_tier")

    allowed = re.findall(r"\d+(?:\.\d+)?", json.dumps(case))
    for number in re.findall(r"\d+(?:\.\d+)?", text):
        if number not in allowed:
            problems.append("made_up_number")
            break

    if any(has_word(text, w) for w in banned_words):
        problems.append("medical_word")
    if case["tier"] == "Red" and not re.search(r"medical|therap", text.lower()):
        problems.append("no_referral")
    return problems


def ollama_note(case):
    import ollama
    reply = ollama.chat(
        model="llama3.2",
        format="json",
        options={"temperature": 0},
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": json.dumps(case)}],
    )
    return json.loads(reply["message"]["content"])


def get_note(case, use_llm):
    # Try the LLM, check the note, and use the fallback if anything goes wrong
    if use_llm and case["drivers"]:
        try:
            note = ollama_note(case)
            problems = check_note(note, case)
            if problems == []:
                return note, "LLM", []
        except Exception:
            problems = ["llm_error"]
        return fallback_note(case), "fallback", problems
    return fallback_note(case), "fallback", []


# ---------------------------------------------------------------
# Part C: Score athletes and find the top reasons
# ---------------------------------------------------------------
names = {
    "HRV_ms": "HRV",
    "Sleep_Duration_Hours": "Sleep",
    "Muscle_Soreness": "Muscle soreness",
    "Stress_Num": "Stress (1 to 3)",
    "Resting_Heart_Rate": "Resting heart rate",
}


def score(df, labels):
    # Adds Label, Tier and Drivers columns to a prepared table
    df = df.copy()
    df["Label"] = labels
    X = df[features].fillna(medians)
    df["Tier"] = model.predict(X)

    # How much each factor pushes toward Red (weight x distance from team average)
    push = pd.DataFrame(scaler.transform(X), columns=features, index=df.index) * weights

    all_drivers = []
    for i in df.index:
        pushes = push.loc[i, list(names) + ["Sleep_Missing"]].sort_values(ascending=False)
        pushes = pushes[pushes >= 0.5].head(3)

        texts = []
        for col in pushes.index:
            if col == "Sleep_Missing":
                texts.append("Sleep was not logged")
            else:
                value = df.loc[i, col]
                word = "low" if value < team_average[col] else "high"
                texts.append(f"{names[col]} is {word}: {value:.1f} vs team average {team_average[col]:.1f}")
        all_drivers.append(texts)

    df["Drivers"] = all_drivers
    return df


def color_tier(value):
    colors = {"Red": "#fecaca", "Orange": "#fed7aa", "Green": "#bbf7d0"}
    return "background-color: " + colors[value]


def show_details(row, use_llm):
    # Shows the tier, the top reasons and the note for one athlete
    case = {
        "athlete": row["Label"],
        "day": int(row["Day"]) if "Day" in row and pd.notna(row["Day"]) else 0,
        "tier": row["Tier"],
        "band": bands[row["Tier"]],
        "drivers": row["Drivers"],
    }

    st.markdown(f"### {case['athlete']}: {case['tier']} tier")

    st.write("**Top reasons**")
    if case["drivers"]:
        for reason in case["drivers"]:
            st.write("- " + reason)
    else:
        st.write("No single clear reason.")

    note, source, problems = get_note(case, use_llm)
    st.write("**Note for the coach**")
    st.info(note["summary"] + "\n\n" + note["action"])
    st.caption("Written by: " + source + (" (LLM note rejected: " + ", ".join(problems) + ")" if problems else ""))


def show_roster(df, use_llm):
    # Counts, the table of athletes (Red first) and the details of one athlete
    counts = df["Tier"].value_counts()
    col1, col2, col3 = st.columns(3)
    col1.metric("Red", int(counts.get("Red", 0)))
    col2.metric("Orange", int(counts.get("Orange", 0)))
    col3.metric("Green", int(counts.get("Green", 0)))

    table = pd.DataFrame({"Athlete": df["Label"], "Tier": df["Tier"], "Reasons": df["Drivers"].apply(len)})
    table["order"] = table["Tier"].map({"Red": 0, "Orange": 1, "Green": 2})
    table = table.sort_values(["order", "Reasons"], ascending=[True, False]).drop(columns="order")

    st.subheader("Athletes")
    st.dataframe(table.style.map(color_tier, subset=["Tier"]), hide_index=True)

    st.subheader("Athlete details")
    chosen = st.selectbox("Pick an athlete", table["Athlete"].tolist())
    show_details(df[df["Label"] == chosen].iloc[0], use_llm)


# ---------------------------------------------------------------
# Part D: Check an uploaded CSV
# ---------------------------------------------------------------
required_columns = ["Athlete_ID", "Sleep_Duration_Hours", "Stress_Level", "HRV_ms",
                    "Resting_Heart_Rate", "Muscle_Soreness", "Caffeine_Intake_mg",
                    "Training_Duration_Min", "Training_Intensity"]
number_columns = [c for c in required_columns if c not in ("Athlete_ID", "Stress_Level")]


def check_upload(df):
    # Returns (clean table, error message). If the error is empty, the table is ready
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return None, "Missing columns: " + ", ".join(missing)

    df = df.copy()

    # Stress must be Low, Medium or High
    df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.title()
    bad_stress = df.loc[~df["Stress_Level"].isin(["Low", "Medium", "High"]), "Stress_Level"].unique()
    if len(bad_stress) > 0:
        return None, "Stress_Level must be Low, Medium or High. Found: " + ", ".join(bad_stress)

    # The other columns must be numbers (blank is allowed, text is not)
    for col in number_columns:
        as_numbers = pd.to_numeric(df[col], errors="coerce")
        if (as_numbers.isna() & df[col].notna()).any():
            return None, "Column " + col + " must contain numbers only."
        df[col] = as_numbers

    return prepare(df), ""


template = pd.DataFrame({
    "Athlete_ID": [1001, 1002],
    "Sleep_Duration_Hours": [8.0, 5.8],
    "Stress_Level": ["Low", "High"],
    "HRV_ms": [85, 52],
    "Resting_Heart_Rate": [54, 66],
    "Muscle_Soreness": [2.0, 8.0],
    "Caffeine_Intake_mg": [100, 320],
    "Training_Duration_Min": [45, 90],
    "Training_Intensity": [5.0, 8.5],
})


# ---------------------------------------------------------------
# Part E: The page
# ---------------------------------------------------------------
st.title("Athlete Readiness Assistant")

mode = st.sidebar.radio("Where do the numbers come from?",
                        ["Demo data", "Upload a CSV", "Type today's numbers"])
use_llm = st.sidebar.checkbox("Use the LLM for notes (needs Ollama running)", value=False)

if mode == "Demo data":
    st.caption("Shows the 60 test athletes from the synthetic dataset.")
    day = st.sidebar.slider("Day", 1, 28, 14)

    demo = load_demo_data()
    demo = demo[demo["Day"] == day]
    demo = score(demo, "A" + demo["Athlete_ID"].astype(str))
    show_roster(demo, use_llm)

elif mode == "Upload a CSV":
    st.caption("Upload today's numbers: one row per athlete.")
    st.sidebar.download_button("Download a CSV template", template.to_csv(index=False), "template.csv")

    st.write("**Columns the CSV must have:** " + ", ".join(required_columns))
    st.write("Sleep can be blank (it is treated as not logged). Other blanks are filled with typical values.")
    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded is not None:
        clean, error = check_upload(pd.read_csv(uploaded))
        if error != "":
            st.error(error)
        elif clean["Athlete_ID"].duplicated().any():
            st.error("Athlete_ID appears more than once. Use one row per athlete.")
        else:
            labels = clean["Athlete_ID"].astype(str).apply(lambda x: x if x.startswith("A") else "A" + x)
            show_roster(score(clean, labels), use_llm)

else:
    st.caption("Type one athlete's numbers for today.")
    name = st.text_input("Athlete name or ID", "A1")

    left, right = st.columns(2)
    sleep_logged = left.checkbox("Sleep was logged", value=True)
    sleep = left.number_input("Sleep (hours)", 0.0, 14.0, 7.5, 0.1, disabled=not sleep_logged)
    stress = left.selectbox("Stress", ["Low", "Medium", "High"], index=1)
    hrv = left.number_input("HRV (ms)", 10.0, 150.0, 74.0, 1.0)
    rhr = left.number_input("Resting heart rate (bpm)", 30.0, 110.0, 58.0, 1.0)
    soreness = right.number_input("Muscle soreness (1 to 10)", 1.0, 10.0, 5.0, 0.5)
    caffeine = right.number_input("Caffeine (mg)", 0.0, 800.0, 200.0, 10.0)
    duration = right.number_input("Training duration (minutes)", 0.0, 240.0, 55.0, 5.0)
    intensity = right.number_input("Training intensity (1 to 10)", 1.0, 10.0, 6.5, 0.5)

    one = pd.DataFrame({
        "Athlete_ID": [name],
        "Sleep_Duration_Hours": [sleep if sleep_logged else None],
        "Stress_Level": [stress],
        "HRV_ms": [hrv],
        "Resting_Heart_Rate": [rhr],
        "Muscle_Soreness": [soreness],
        "Caffeine_Intake_mg": [caffeine],
        "Training_Duration_Min": [duration],
        "Training_Intensity": [intensity],
    })
    one["Sleep_Duration_Hours"] = one["Sleep_Duration_Hours"].astype(float)

    scored = score(prepare(one), [name])
    show_details(scored.iloc[0], use_llm)