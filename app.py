import streamlit as st
import nltk
from nltk.tokenize import sent_tokenize
from newspaper import Article
from transformers import pipeline

# Make sure we download BOTH "punkt" and "punkt_tab" just in case
nltk.download('punkt')
nltk.download('punkt_tab')

st.title("Speculation vs. Fact Classifier")

# 1) Initialize a zero-shot classifier pipeline
#    (loads a large pretrained model from Hugging Face)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Define the labels we want the classifier to choose from
labels = ["Speculation", "Verifiable Fact"]

def extract_article_text(url: str) -> str:
    """
    Extract main text content from the given news article URL
    using newspaper3k.
    """
    article = Article(url)
    article.download()
    article.parse()
    return article.text

def classify_sentence(sentence: str) -> str:
    """
    Classify a single sentence as either "Speculation" or "Verifiable Fact"
    using zero-shot classification.
    """
    result = classifier(sentence, labels)
    # The 'labels' are returned in order of descending confidence score
    predicted_label = result["labels"][0]
    return predicted_label

def main():
    # 2) Ask user for a news article URL
    url = st.text_input("Enter a news article URL:")
    
    if st.button("Classify Article"):
        # Extract the text from the URL
        st.write("### Extracting article text...")
        try:
            article_text = extract_article_text(url)
        except Exception as e:
            st.error(f"Error extracting article: {e}")
            return
        
        # 3) Split the text into individual sentences
        st.write("### Splitting article into sentences...")
        sentences = sent_tokenize(article_text)

        # 4) Classify each sentence
        st.write("### Classifying each sentence...")
        results = []
        for sent in sentences:
            label = classify_sentence(sent)
            results.append((sent, label))
        
        # 5) Display the results
        st.subheader("Classification Results")
        for (sent, label) in results:
            if label == "Speculation":
                # Example styling: highlight speculation in orange
                st.markdown(f"**:orange[Speculation]**: {sent}")
            else:
                # Example styling: highlight factual sentences in green
                st.markdown(f"**:green[Verifiable Fact]**: {sent}")

if __name__ == "__main__":
    main()
