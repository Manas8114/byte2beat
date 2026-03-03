# Plan: UI Slider and Text Input

## 🎯 Goal

Improve the user experience in the byte2beat `dashboard.py` sidebar by allowing users to input patient vitals through both visual sliders and precise text/number input boxes.

## 🧠 Socratic Gate (Context Check)

Before moving to implementation, please confirm the following:

1. **Layout Preference:** Do you want the text input immediately below the slider, or side-by-side on the same line (using Streamlit columns)?
2. **State Management:** When typing in the box out of bounds (e.g., max age is 100, user types 150), should it automatically clamp to 100, or throw a visual warning?
3. **Other "Make it better" Ideas:** Beyond text inputs, should we add grouping (e.g., putting all "Vitals" in an expander, and all "Lab Results" in another expander) to make the long sidebar cleaner?

## 🛠️ Task Breakdown

1. **State Synchronization:** Implement Streamlit `st.session_state` callbacks to bind a `st.slider` and an `st.number_input` to the exact same underlying value.
2. **UI Layout Update (`dashboard.py` L300-340):**
   - Refactor the dynamic input generation loop.
   - For every continuous variable (Age, Cholesterol, BP, MaxHR), yield a column layout (e.g., `col1, col2 = st.columns([3, 1])`).
   - Place the slider in `col1` and the number input in `col2`.
3. **Input Validation:** Ensure boundaries (`min_value`, `max_value`) and `step` sizes are properly passed to both components to avoid Streamlit ValueErrors.
4. **General Dashboard Polish:** (Optional, based on answer to #3) Add `st.expander` groupings to categorize features for better readability.

## 🤖 Agent Assignments

- **Frontend Specialist:** Responsible for editing `dashboard.py`, setting up the Streamlit session state callbacks, and configuring column layouts.
- **Debugger:** (Standby) Ensuring SHAP/MC Dropout types (`float64`) do not break when interacting with the newly introduced text box inputs.

## ✅ Verification Checklist

- [ ] Text box updates automatically when slider is dragged.
- [ ] Slider moves automatically when text box is typed into and 'Enter' is pressed.
- [ ] Dashboard continues to successfully run inference without TypeErrors.
- [ ] Edge-case values (typing letters or extremes) are handled properly by Streamlit's native validation.
