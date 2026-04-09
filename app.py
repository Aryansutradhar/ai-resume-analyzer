import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page config
st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="centered")

# Title
st.markdown("<h1 style='text-align: center;'>📄 AI Resume Analyzer</h1>", unsafe_allow_html=True)
st.write("Compare your resume with a job description and get a match score with skill insights.")

# Input fields
resume = st.text_area("📌 Paste your Resume here")
job_description = st.text_area("📌 Paste Job Description here")

# Skills list
skills = ["python", "machine learning", "deep learning", "nlp", "data analysis", "pandas", "numpy", "tensorflow", "sql"]

# Button
if st.button("Analyze Resume"):
    if resume and job_description:

        # Convert text into vectors using TF-IDF
        text = [resume, job_description]
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(text)

        # Calculate similarity
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        match = round(similarity * 100, 2)

        st.subheader(f"📊 Match Score: {match}%")

        # Skill analysis
        resume_lower = resume.lower()
        matched = [skill for skill in skills if skill in resume_lower]
        missing = [skill for skill in skills if skill not in resume_lower]

        # Display results
        st.subheader("✅ Matched Skills")
        st.write(matched if matched else "No matching skills found")

        st.subheader("❌ Missing Skills")
        st.write(missing if missing else "No missing skills")

        # Suggestions
        st.subheader("💡 Suggestions")
        if missing:
            for skill in missing:
                st.write(f"- Consider learning {skill}")
        else:
            st.success("Great! Your resume matches well with the job description.")

    else:
        st.warning("⚠️ Please enter both resume and job description")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Aryan 🚀</p>", unsafe_allow_html=True)