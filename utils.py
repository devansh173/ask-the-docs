import PyPDF2

def load_text(file):
    if file.name.endswith(".pdf"):
        try:
            pdf = PyPDF2.PdfReader(file)
            return "\n".join([page.extract_text() or "" for page in pdf.pages])
        except Exception as e:
            return f"❌ Error reading PDF: {e}"
    elif file.name.endswith(".txt"):
        try:
            return file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            return f"❌ Error reading TXT: {e}"
    else:
        return "❌ Unsupported file type"
