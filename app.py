# requirements.txt
"""
streamlit
langchain
langchain-community
langchain-core
langchain-text-splitters
langchain-huggingface
langchain-groq
chromadb
sentence-transformers
pypdf
python-docx
"""

# study_buddy_groq.py

import os
import json
import random
import shutil
import tempfile
import streamlit as st

from typing import List

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in .env file. Please set it.")
    st.stop()


from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ========================
# Configuration
# ========================

PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


class StudyBuddyGroq:
    def __init__(self):

        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=0.3,
            max_tokens=1024,
        )

        self.vectorstore = None

    # ========================
    # Document Loading
    # ========================
    def load_document(self, file_path: str):

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".pdf":
            loader = PyPDFLoader(file_path)

        elif ext == ".txt":
            loader = TextLoader(file_path, encoding="utf-8")

        elif ext == ".docx":
            loader = Docx2txtLoader(file_path)

        else:
            raise ValueError(f"Unsupported file type: {ext}")

        return loader.load()

    # ========================
    # Ingest Document
    # ========================
    def ingest_document(self, uploaded_file):

        # Clear any existing vectorstore
        if self.vectorstore:
            # Try to persist and close the vectorstore
            try:
                self.vectorstore.persist()
            except:
                pass
            self.vectorstore = None

        # Clear the persistence directory if it exists
        if os.path.exists(PERSIST_DIR):
            import shutil
            import time
            # Try to remove the directory, with retries for Windows file locking
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    shutil.rmtree(PERSIST_DIR)
                    break
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Wait a bit before retrying
                    else:
                        # If still failing, try to remove individual files
                        try:
                            for root, dirs, files in os.walk(PERSIST_DIR, topdown=False):
                                for file in files:
                                    try:
                                        os.remove(os.path.join(root, file))
                                    except:
                                        pass
                                for dir_name in dirs:
                                    try:
                                        os.rmdir(os.path.join(root, dir_name))
                                    except:
                                        pass
                            os.rmdir(PERSIST_DIR)
                        except:
                            pass  # If all else fails, continue anyway

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1],
        ) as tmp_file:

            shutil.copyfileobj(uploaded_file, tmp_file)
            tmp_path = tmp_file.name

        documents = self.load_document(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        chunks = splitter.split_documents(documents)

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=PERSIST_DIR,
        )

        os.unlink(tmp_path)

        return f"✅ Ingested {len(chunks)} chunks from {uploaded_file.name}"

    # ========================
    # Ask Questions
    # ========================
    def ask_question(self, question: str):

        if not self.vectorstore:
            return "Please upload a document first."

        docs = self.vectorstore.similarity_search(question, k=4)

        context = "\n\n".join(
            [doc.page_content for doc in docs]
        )

        prompt = PromptTemplate.from_template(
            """
You are a helpful study assistant.

Answer ONLY using the provided context.

If the answer is not found, say:
"I don't have enough information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
        )

        chain = prompt | self.llm | StrOutputParser()

        answer = chain.invoke({
            "context": context,
            "question": question
        })

        return answer

    # ========================
    # Generate Quiz
    # ========================
    def generate_quiz(self, num_questions=5):

        if not self.vectorstore:
            return []

        docs = self.vectorstore.get()["documents"]

        sampled = random.sample(
            docs,
            min(len(docs), num_questions * 2)
        )

        context = "\n".join(sampled)

        prompt = PromptTemplate.from_template(
            """
Generate {num_questions} multiple choice questions from this text.

Return ONLY valid JSON with this exact structure:
[
  {{
    "question": "Question text here?",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "correct_answer": "Option A",
    "explanation": "Brief explanation why this is correct."
  }}
]

Text:
{context}
"""
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "num_questions": num_questions
        })

        try:
            return json.loads(response)
        except:
            return []

    # ========================
    # Summary
    # ========================
    def generate_summary(self):

        if not self.vectorstore:
            return "Upload document first."

        docs = self.vectorstore.get()["documents"]

        context = " ".join(docs[:10])

        prompt = PromptTemplate.from_template(
            """
Summarize this into concise study notes.

Text:
{text}
"""
        )

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({"text": context})

    # ========================
    # Flashcards
    # ========================
    def generate_flashcards(self, num_cards=10):

        if not self.vectorstore:
            return []

        docs = self.vectorstore.get()["documents"]

        context = " ".join(docs[:10])

        prompt = PromptTemplate.from_template(
            """
Create {num_cards} flashcards from this text.

Return ONLY valid JSON.

Example:
[
  {{
    "front": "What is Python?",
    "back": "A programming language"
  }}
]

Text:
{context}
"""
        )

        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "num_cards": num_cards
        })

        try:
            return json.loads(response)

        except:
            return []


    # ========================
    # Gen Z Breakdown
    # ========================
    def generate_genz_breakdown(self):

        if not self.vectorstore:
            return "Upload document first."

        docs = self.vectorstore.get()["documents"]

        context = " ".join(docs[:15])  # Use more content for better breakdown

        prompt = PromptTemplate.from_template(
            """
Break down this document content in SUPER simple terms, like you're explaining it to your Gen Z bestie who's never heard of this topic before. 

Use:
- Casual language like "basically", "kinda", "super", "lit", "vibes"
- Emojis to make it fun 🎉📚🤔
- Short sentences, no fancy words
- Relatable examples from everyday life
- Keep it engaging and not boring

Document content:
{text}

Gen Z breakdown:
"""
        )

        chain = prompt | self.llm | StrOutputParser()

        return chain.invoke({"text": context})


# ========================
# Streamlit UI
# ========================
def main():

    st.set_page_config(
        page_title="Study Buddy AI",
        page_icon="📚",
        layout="wide",
    )

    st.title("📚 Study Buddy AI")

    st.write("""
    Welcome to Study Buddy AI! This app helps you study and understand documents by allowing you to ask questions, generate quizzes, create summaries, make flashcards, and even get fun Gen Z-style breakdowns of the content. Simply upload a document to get started!
    """)

    if st.button("Upload Document"):
        st.info("👈 Use the sidebar on the left to upload your document and begin!")

    with st.sidebar:

        uploaded_file = st.file_uploader(
            "Upload File",
            type=["pdf", "txt", "docx"]
        )

        if uploaded_file and st.button("Ingest Document"):

            with st.spinner("Processing document... This may take a few minutes."):
                buddy = StudyBuddyGroq()

                msg = buddy.ingest_document(uploaded_file)

            st.success(msg)

            # Clear any cached data from previous documents
            for key in ["quiz_questions", "quiz_answers", "show_results"]:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state["buddy"] = buddy
            
            st.info("📄 New document ingested! Previous quiz data and cached results have been cleared.")

    if "buddy" in st.session_state:

        buddy = st.session_state["buddy"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Ask",
            "Quiz",
            "Summary",
            "Flashcards",
            "Gen Z Breakdown"
        ])

        with tab1:

            question = st.text_input("Ask a question about the document")

            if st.button("Submit Question", disabled=st.session_state.get("processing_question", False)):

                with st.spinner("Thinking..."):
                    st.session_state["processing_question"] = True
                    answer = buddy.ask_question(question)
                    st.session_state["processing_question"] = False

                st.write("**Answer:**")
                st.write(answer)

        with tab2:

            if st.button("Generate Quiz", disabled=st.session_state.get("processing_quiz", False)):

                with st.spinner("Generating quiz questions..."):
                    st.session_state["processing_quiz"] = True
                    quiz = buddy.generate_quiz()
                    st.session_state["processing_quiz"] = False

                if quiz:
                    st.session_state["quiz_questions"] = quiz
                    st.session_state["quiz_answers"] = {}
                    st.session_state["show_results"] = False
                    st.rerun()

            # Display quiz if questions are loaded
            if "quiz_questions" in st.session_state and st.session_state["quiz_questions"]:
                quiz_questions = st.session_state["quiz_questions"]
                quiz_answers = st.session_state["quiz_answers"]

                st.write("### 📝 Quiz")

                # Display each question
                for idx, q in enumerate(quiz_questions):
                    st.write(f"**Question {idx + 1}: {q['question']}**")
                    
                    # Store the answer selection with default unselected state
                    options_with_default = ["Select an answer..."] + q["options"]
                    selected = st.radio(
                        label="Select your answer:",
                        options=options_with_default,
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                        index=0  # Start with "Select an answer..." selected
                    )
                    
                    # Only store the answer if user actually selected an option
                    if selected != "Select an answer...":
                        quiz_answers[idx] = selected
                    else:
                        quiz_answers.pop(idx, None)  # Remove if they went back to default
                    
                    st.write("---")

                # Submit button
                if st.button("Submit Quiz", type="primary"):
                    st.session_state["show_results"] = True
                    st.rerun()

                # Show results if submitted
                if st.session_state.get("show_results", False):
                    st.write("### 📊 Quiz Results")
                    
                    correct_count = 0
                    wrong_answers = []

                    for idx, q in enumerate(quiz_questions):
                        user_answer = quiz_answers.get(idx)
                        correct_answer = q["correct_answer"]
                        is_correct = user_answer == correct_answer

                        if is_correct:
                            correct_count += 1
                        else:
                            wrong_answers.append({
                                "question": q["question"],
                                "user_answer": user_answer,
                                "correct_answer": correct_answer,
                                "explanation": q["explanation"]
                            })

                    # Show score
                    score_percentage = (correct_count / len(quiz_questions)) * 100
                    st.metric("Your Score", f"{correct_count}/{len(quiz_questions)} ({score_percentage:.0f}%)")

                    # Show wrong answers
                    if wrong_answers:
                        st.write("### ❌ Incorrect Answers")
                        for item in wrong_answers:
                            with st.expander(f"Q: {item['question']}"):
                                st.write(f"**Your answer:** {item['user_answer']}")
                                st.write(f"**Correct answer:** {item['correct_answer']}")
                                st.write(f"**Explanation:** {item['explanation']}")
                    else:
                        st.success("🎉 Perfect! You got all questions correct!")

                    if st.button("Take Another Quiz"):
                        st.session_state["quiz_questions"] = None
                        st.session_state["quiz_answers"] = {}
                        st.session_state["show_results"] = False
                        st.rerun()

        with tab3:

            if st.button("Generate Summary", disabled=st.session_state.get("processing_summary", False)):

                with st.spinner("Creating summary..."):
                    st.session_state["processing_summary"] = True
                    summary = buddy.generate_summary()
                    st.session_state["processing_summary"] = False

                st.write("**Summary:**")
                st.write(summary)

        with tab4:

            if st.button("Generate Flashcards", disabled=st.session_state.get("processing_flashcards", False)):

                with st.spinner("Creating flashcards..."):
                    st.session_state["processing_flashcards"] = True
                    cards = buddy.generate_flashcards()
                    st.session_state["processing_flashcards"] = False

                if cards:
                    st.write("**Flashcards:**")
                    for card in cards:
                        with st.expander(card["front"]):
                            st.write(card["back"])
                else:
                    st.write("Could not generate flashcards. Please try again.")

        with tab5:

            if st.button("Generate Gen Z Breakdown", disabled=st.session_state.get("processing_genz", False)):

                with st.spinner("Breaking it down in Gen Z style..."):
                    st.session_state["processing_genz"] = True
                    breakdown = buddy.generate_genz_breakdown()
                    st.session_state["processing_genz"] = False

                st.write("**🤩 Gen Z Breakdown:**")
                st.write(breakdown)


if __name__ == "__main__":
    main()