import streamlit as st
import pickle

# Load the trained model and vectorizer
with open('spam_model (1).pkl', 'rb') as f:
    clf = pickle.load(f)
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)

st.title("Spam Detection System")

# Create a form for input and button
with st.form(key="spam_form"):
    user_input = st.text_area("Enter any Message or Email: ")
    submit_button = st.form_submit_button("Submit")

# Process input when the button is clicked
if submit_button:
    if len(user_input) < 1:
        st.warning("Please enter a message before submitting.")
    else:
        data = cv.transform([user_input]).toarray()
        prediction = clf.predict(data)

        # Convert the prediction to readable format
        if prediction[0] == "spam":
            st.error("🚨 This message is **Spam** 🚨")
        else:
            st.success("✅ This message is **Not Spam** ✅")
