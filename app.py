import streamlit as st
from utils import load_text
from rag_chain import build_qa_chain

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Ask the Docs", page_icon="📄", layout="wide")

# 🔁 Session state initialization
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'docs' not in st.session_state:
    st.session_state['docs'] = []

st.title("📄 Ask the Docs – RAG App")

# 🔖 Two main tabs
tab1, tab2 = st.tabs(["📤 Upload Files", "❓ Ask Questions"])

# 📤 Upload Tab
with tab1:
    st.header("Upload your PDF or TXT files")
    uploaded_files = st.file_uploader("Choose files", type=['pdf', 'txt'], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            text = load_text(uploaded_file)
            st.session_state.docs.append(text)
        st.success(f"✅ Successfully loaded {len(uploaded_files)} file(s).")

# ❓ Ask Questions Tab
with tab2:
    st.header("Ask a question about your document(s)")
    if not st.session_state.docs:
        st.info("👆 Please upload at least one document first.")
    else:
        question = st.text_input("🧠 Enter your question:")

        if st.button("Get Answer") and question:
            with st.spinner("🔍 Thinking..."):
                combined = "\n".join(st.session_state.docs)
                qa_chain = build_qa_chain(combined)
                answer = qa_chain.run(question)

                # ✅ Visible Answer Box
                st.markdown(f"**Q:** {question}")
                st.markdown(
                    f"""
                    <div style='
                        background-color:#f1f3f5;
                        padding:16px;
                        border-radius:8px;
                        color:#000000;
                        font-size:16px;
                        font-weight:normal;
                    '><b>A:</b> {answer}</div>
                    """,
                    unsafe_allow_html=True
                )

                # 📝 Save to history
                st.session_state.history.append({'question': question, 'answer': answer})

# 📜 Sidebar: History
if st.session_state.history:
    st.sidebar.header("🕑 Question History")
    for i, qa in enumerate(reversed(st.session_state.history), 1):
        st.sidebar.markdown(
            f"""
            <div style='margin-bottom:12px;'>
                <b>{i}. Q:</b> {qa['question']}<br>
                <span style='color:#444'><b>A:</b> {qa['answer']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
