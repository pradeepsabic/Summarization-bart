from langchain_community.document_loaders import PyPDFLoader
from transformers import BartForConditionalGeneration, BartTokenizer
from keybert import KeyBERT

# Load the BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)

# Load KeyBERT for keyword extraction
kw_model = KeyBERT()

def load_pdf(pdf_path):
    """Load PDF document and return its content."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents

def chunk_text(documents, chunk_size=400):
    """Chunk the loaded documents into smaller pieces."""
    chunks = []
    for doc in documents:
        text = doc.page_content
        for i in range(0, len(text), chunk_size):
            chunks.append(text[i:i + chunk_size])
    return chunks

def summarize_text(text):
    """Generate a summary using the BART model."""
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(
        inputs['input_ids'],
        max_length=60,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def extract_keywords(text):
    """Extract keywords using KeyBERT."""
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return [keyword[0] for keyword in keywords]  # Return only the keyword strings

# Example usage
pdf_path = "Bart tut.pdf"  
documents = load_pdf(pdf_path)
chunks = chunk_text(documents)

# List to hold the individual summaries
combined_summaries = []

# Summarize each chunk and highlight main points
for i, chunk in enumerate(chunks):
    summary = summarize_text(chunk)
    keywords = extract_keywords(chunk)  # Extract keywords using KeyBERT
    highlighted_summary = f"**Summary of chunk {i + 1}:** {summary} (Key Points: {', '.join(keywords)})"
    combined_summaries.append(highlighted_summary)  # Store the summary for each chunk
    print(highlighted_summary)

# Combine all summaries into one cohesive summary
final_summary = "\n".join(combined_summaries)  # Combine using newline or another separator

# Optionally, you can enhance the final summary
final_summary = f"Combined Summary:\n{final_summary}"

# Print the final combined summary
print(final_summary)