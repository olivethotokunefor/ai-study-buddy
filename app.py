# study_buddy_groq.py

import os
import json
import random
import tempfile
import streamlit as st

from dotenv import load_dotenv

load_dotenv()

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

import chromadb

# ========================
# Configuration
# ========================

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"


# ========================
# Cached resource loaders
# Loaded ONCE per app session, not on every document upload.
# This is the biggest speed win — the embedding model takes 3-5s to load.
# ========================

@st.cache_resource(show_spinner="Loading embedding model (one-time)...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": 64},
    )


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.3,
        max_tokens=4096,
    )


# ========================
# Study Buddy Class
# ========================

class StudyBuddyGroq:
    def __init__(self):
        # Pull from cache — no cold-start after first load
        self.embeddings = get_embeddings()
        self.llm = get_llm()
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
        self.vectorstore = None

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1],
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        try:
            documents = self.load_document(tmp_path)
        finally:
            os.unlink(tmp_path)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        # In-memory client — no disk writes, works on Streamlit Cloud
        chroma_client = chromadb.EphemeralClient()

        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            client=chroma_client,
            collection_name="study_buddy",
        )

        return f"Ingested {len(chunks)} chunks from {uploaded_file.name}"

    # ========================
    # Ask Questions (with history)
    # ========================
    def ask_question(self, question: str, history: list):
        if not self.vectorstore:
            return "Please upload a document first."

        docs = self.vectorstore.similarity_search(question, k=4)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Keep last 6 exchanges to stay within token budget
        recent = history[-6:] if len(history) > 6 else history
        history_text = "\n".join(
            f"{'You' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        ) if recent else "None yet."

        prompt = PromptTemplate.from_template(
            """You are a helpful study assistant.

Answer ONLY using the provided document context.
If the answer is not in the context, say: "I don't have enough information in the uploaded document."

Previous conversation (use only to understand follow-up references):
{history}

Document context:
{context}

Current question:
{question}

Answer:"""
        )

        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context,
            "question": question,
            "history": history_text,
        })

    # ========================
    # Generate Quiz
    # ========================
    def generate_quiz(self, num_questions=5):
        if not self.vectorstore:
            return []

        docs = self.vectorstore.get()["documents"]
        sampled = random.sample(docs, min(len(docs), num_questions * 2))
        context = "\n".join(sampled)

        prompt = PromptTemplate.from_template(
            """Generate {num_questions} multiple choice questions from this text.

Return ONLY valid JSON with this exact structure:
[
  {{
    "question": "Question text here?",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "correct_answer": "Option A",
    "explanation": "Brief explanation why this is correct."
  }}
]

Text:
{context}"""
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "num_questions": num_questions})

        try:
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except Exception:
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
            "Summarize this into concise study notes.\n\nText:\n{text}"
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
            """Create {num_cards} flashcards from this text.

Return ONLY valid JSON.

Example:
[
  {{
    "front": "What is Python?",
    "back": "A programming language"
  }}
]

Text:
{context}"""
        )

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": context, "num_cards": num_cards})

        try:
            clean = response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(clean)
        except Exception:
            return []

    # ========================
    # Gen Z Breakdown
    # ========================
    def generate_genz_breakdown(self):
        if not self.vectorstore:
            return "Upload document first."

        docs = self.vectorstore.get()["documents"]
        context = " ".join(docs[:15])

        prompt = PromptTemplate.from_template(
            """Break down this document content in SUPER simple terms, like you're explaining it to your Gen Z bestie who's never heard of this topic before.

Use:
- Casual language like "basically", "kinda", "super", "lit", "vibes"
- Emojis to make it fun 🎉📚🤔
- Short sentences, no fancy words
- Relatable examples from everyday life
- Keep it engaging and not boring

Document content:
{text}

Gen Z breakdown:"""
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
    st.write(
        "Welcome to Study Buddy AI! Upload a document to ask questions, generate quizzes, "
        "create summaries, make flashcards, and get fun Gen Z-style breakdowns."
    )

    if st.button("Upload Document"):
        st.info("👈 Use the sidebar on the left to upload your document and begin!")

    with st.sidebar:
        uploaded_file = st.file_uploader("Upload File", type=["pdf", "txt", "docx"])

        if uploaded_file and st.button("Ingest Document"):
            with st.spinner("Processing document..."):
                buddy = StudyBuddyGroq()
                msg = buddy.ingest_document(uploaded_file)

            st.success(f"✅ {msg}")

            # Clear all cached data from the previous document
            for key in ["quiz_questions", "quiz_answers", "show_results", "chat_history"]:
                if key in st.session_state:
                    del st.session_state[key]

            st.session_state["buddy"] = buddy
            st.info("📄 New document ingested! Previous data has been cleared.")

    if "buddy" in st.session_state:
        buddy = st.session_state["buddy"]

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "💬 Ask",
            "📝 Quiz",
            "📖 Summary",
            "🃏 Flashcards",
            "🤩 Gen Z Breakdown",
        ])

        # ========================
        # Tab 1: Ask with chat history
        # ========================
        with tab1:
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            chat_history: list = st.session_state["chat_history"]

            # Render all previous messages
            for msg in chat_history:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

            # st.chat_input pins itself to the bottom of the page
            user_input = st.chat_input("Ask a question about the document...")

            if user_input:
                with st.chat_message("user"):
                    st.write(user_input)

                # Append user message before calling LLM so history is complete
                chat_history.append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        # Pass history excluding the message just added
                        answer = buddy.ask_question(user_input, chat_history[:-1])
                    st.write(answer)

                chat_history.append({"role": "assistant", "content": answer})
                st.session_state["chat_history"] = chat_history

            if chat_history:
                if st.button("🗑️ Clear chat history"):
                    st.session_state["chat_history"] = []
                    st.rerun()

        # ========================
        # Tab 2: Quiz
        # ========================
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

            if "quiz_questions" in st.session_state and st.session_state["quiz_questions"]:
                quiz_questions = st.session_state["quiz_questions"]
                quiz_answers = st.session_state["quiz_answers"]

                st.write("### 📝 Quiz")

                for idx, q in enumerate(quiz_questions):
                    st.write(f"**Question {idx + 1}: {q['question']}**")
                    options_with_default = ["Select an answer..."] + q["options"]
                    selected = st.radio(
                        label="Select your answer:",
                        options=options_with_default,
                        key=f"quiz_q_{idx}",
                        label_visibility="collapsed",
                        index=0,
                    )
                    if selected != "Select an answer...":
                        quiz_answers[idx] = selected
                    else:
                        quiz_answers.pop(idx, None)
                    st.write("---")

                if st.button("Submit Quiz", type="primary"):
                    st.session_state["show_results"] = True
                    st.rerun()

                if st.session_state.get("show_results", False):
                    st.write("### 📊 Quiz Results")

                    correct_count = 0
                    wrong_answers = []

                    for idx, q in enumerate(quiz_questions):
                        user_answer = quiz_answers.get(idx)
                        correct_answer = q["correct_answer"]
                        if user_answer == correct_answer:
                            correct_count += 1
                        else:
                            wrong_answers.append({
                                "question": q["question"],
                                "user_answer": user_answer,
                                "correct_answer": correct_answer,
                                "explanation": q["explanation"],
                            })

                    score_pct = (correct_count / len(quiz_questions)) * 100
                    st.metric("Your Score", f"{correct_count}/{len(quiz_questions)} ({score_pct:.0f}%)")

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

        # ========================
        # Tab 3: Summary
        # ========================
        with tab3:
            if st.button("Generate Summary", disabled=st.session_state.get("processing_summary", False)):
                with st.spinner("Creating summary..."):
                    st.session_state["processing_summary"] = True
                    summary = buddy.generate_summary()
                    st.session_state["processing_summary"] = False

                st.write("**Summary:**")
                st.write(summary)

        # ========================
        # Tab 4: Flashcards
        # ========================
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

        # ========================
        # Tab 5: Gen Z Breakdown
        # ========================
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