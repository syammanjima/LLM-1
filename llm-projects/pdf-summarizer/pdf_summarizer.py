# Enhanced Document Summarization Tool
# Supports: PDF, DOCX, TXT, RTF + Hugging Face + Ollama integration

import streamlit as st
import PyPDF2
import docx
import requests
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import io
import re
from datetime import datetime
import nltk
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class DocumentSummarizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.hf_summarizer = None
        self.sentence_model = None
        self.ollama_backend = "http://localhost:8000"
        
    @st.cache_resource
    def load_hf_model(_self, model_name):
        """Load Hugging Face summarization model"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            summarizer = pipeline(
                "summarization", 
                model=model, 
                tokenizer=tokenizer,
                device=0 if _self.device == "cuda" else -1
            )
            return summarizer
        except Exception as e:
            st.error(f"Error loading Hugging Face model: {e}")
            return None
    
    @st.cache_resource
    def load_sentence_model(_self):
        """Load sentence embedding model for extractive summarization"""
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    def get_ollama_models(self):
        """Get available Ollama models"""
        try:
            response = requests.get(f"{self.ollama_backend}/models", timeout=5)
            if response.status_code == 200:
                return response.json()["models"]
            else:
                return ["llama2", "mistral", "codellama", "gemma3"]
        except:
            return ["llama2", "mistral", "codellama", "gemma3"]
    
    def ollama_summarize(self, text, model="llama2", max_length=200):
        """Summarize using Ollama local LLM"""
        try:
            # Create summarization prompt
            prompt = f"""Please provide a concise summary of the following text in about {max_length} words:

{text}

Summary:"""
            
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": model
            }
            
            response = requests.post(
                f"{self.ollama_backend}/chat",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            
            if pdf_reader.is_encrypted:
                st.error("This PDF is password protected.")
                return None
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        cleaned_text = re.sub(r'/[a-zA-Z]*\d+', ' ', page_text)
                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
                        cleaned_text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', cleaned_text)
                        
                        if len(cleaned_text) > 20:
                            text += f"\n{cleaned_text}"
                            
                except Exception as e:
                    st.warning(f"Error on page {page_num + 1}: {e}")
                    continue
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return None
    
    def extract_text_from_docx(self, docx_file):
        """Extract text from Word document"""
        try:
            doc = docx.Document(docx_file)
            text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text += paragraph.text + "\n"
            
            return text.strip()
            
        except Exception as e:
            st.error(f"Error reading Word document: {e}")
            return None
    
    def extract_text_from_txt(self, txt_file):
        """Extract text from text file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    txt_file.seek(0)
                    content = txt_file.read()
                    if isinstance(content, bytes):
                        text = content.decode(encoding)
                    else:
                        text = content
                    return text.strip()
                except UnicodeDecodeError:
                    continue
            
            st.error("Could not decode text file. Please check file encoding.")
            return None
            
        except Exception as e:
            st.error(f"Error reading text file: {e}")
            return None
    
    def extract_text_from_file(self, uploaded_file):
        """Extract text based on file type"""
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'pdf':
            return self.extract_text_from_pdf(uploaded_file)
        elif file_type in ['docx', 'doc']:
            return self.extract_text_from_docx(uploaded_file)
        elif file_type in ['txt', 'rtf']:
            return self.extract_text_from_txt(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_type}")
            return None
    
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        return text.strip()
    
    def split_text_into_chunks(self, text, max_length=1000):
        """Split text into chunks for processing"""
        if not text or len(text.strip()) == 0:
            return []
        
        try:
            sentences = nltk.sent_tokenize(text)
            if not sentences:
                sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
                
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                    
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += " " + sentence if current_chunk else sentence
                else:
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            chunks = [chunk for chunk in chunks if len(chunk.split()) >= 5]
            return chunks
            
        except Exception as e:
            st.error(f"Error splitting text: {e}")
            return [text] if text.strip() else []
    
    def hf_abstractive_summarization(self, text, max_length=150, min_length=50):
        """Generate abstractive summary using Hugging Face models"""
        if not self.hf_summarizer:
            return "Please load a Hugging Face model first."
        
        try:
            chunks = self.split_text_into_chunks(text, max_length=1000)
            summaries = []
            
            if not chunks:
                return "No text chunks found to summarize."
            
            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                if len(chunk.split()) > 20:
                    try:
                        chunk_words = len(chunk.split())
                        adjusted_min_length = min(min_length, max(10, chunk_words // 3))
                        adjusted_max_length = min(max_length, chunk_words)
                        
                        summary = self.hf_summarizer(
                            chunk,
                            max_length=adjusted_max_length,
                            min_length=adjusted_min_length,
                            do_sample=False
                        )
                        
                        if summary and len(summary) > 0 and 'summary_text' in summary[0]:
                            summaries.append(summary[0]['summary_text'])
                        
                    except Exception as chunk_error:
                        st.warning(f"Error processing chunk {i+1}: {chunk_error}")
                        continue
                
                progress_bar.progress((i + 1) / len(chunks))
            
            if len(summaries) > 1:
                combined_summary = " ".join(summaries)
                combined_words = len(combined_summary.split())
                
                if combined_words > max_length:
                    try:
                        final_summary = self.hf_summarizer(
                            combined_summary,
                            max_length=max_length,
                            min_length=min(min_length, combined_words // 3),
                            do_sample=False
                        )
                        if final_summary and len(final_summary) > 0:
                            return final_summary[0]['summary_text']
                    except Exception:
                        return combined_summary[:max_length * 5]
                
                return combined_summary
            elif summaries:
                return summaries[0]
            else:
                return "Unable to generate summary. Try extractive summarization."
                
        except Exception as e:
            st.error(f"Error in Hugging Face summarization: {e}")
            return None
    
    def extractive_summarization(self, text, num_sentences=5):
        """Generate extractive summary using sentence embeddings"""
        if not self.sentence_model:
            self.sentence_model = self.load_sentence_model()
        
        try:
            text = re.sub(r'\s+', ' ', text).strip()
            sentences = nltk.sent_tokenize(text)
            
            if not sentences:
                return "No sentences found in the text."
                
            sentences = [s.strip() for s in sentences if len(s.strip().split()) >= 3]
            
            if len(sentences) <= num_sentences:
                return " ".join(sentences)
            
            if len(sentences) > 1000:
                step = len(sentences) // 500
                sentences = sentences[::max(1, step)]
            
            with st.spinner("Generating sentence embeddings..."):
                embeddings = self.sentence_model.encode(sentences, show_progress_bar=False)
            
            n_clusters = min(num_sentences, len(sentences))
            
            with st.spinner("Clustering sentences..."):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings)
            
            selected_sentences = []
            for i in range(n_clusters):
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(embeddings - center, axis=1)
                closest_idx = np.argmin(distances)
                selected_sentences.append((closest_idx, sentences[closest_idx]))
            
            selected_sentences.sort(key=lambda x: x[0])
            summary = " ".join([sent[1] for sent in selected_sentences])
            
            return summary
            
        except Exception as e:
            st.error(f"Error in extractive summarization: {e}")
            try:
                sentences = nltk.sent_tokenize(text[:5000])
                return " ".join(sentences[:num_sentences])
            except:
                return f"Extractive summarization failed. Text preview: {text[:500]}..."

def main():
    st.set_page_config(
        page_title="Document Summarization Tool",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Enhanced Document Summarization Tool")
    st.markdown("Upload documents (PDF, Word, TXT) and generate summaries using **Hugging Face** or **Ollama Local LLMs**")
    
    # Initialize summarizer
    if "summarizer_obj" not in st.session_state:
        st.session_state.summarizer_obj = DocumentSummarizer()
    
    summarizer = st.session_state.summarizer_obj
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # AI Backend Selection
        st.subheader("🤖 AI Backend")
        ai_backend = st.radio(
            "Choose AI Backend",
            ["Hugging Face (Cloud)", "Ollama (Local)"],
            help="Hugging Face: Better quality, needs internet. Ollama: Private, works offline."
        )
        
        if ai_backend == "Hugging Face (Cloud)":
            # Hugging Face Models
            hf_models = {
                "BART (Recommended)": "facebook/bart-large-cnn",
                "T5 Small": "t5-small",
                "T5 Base": "t5-base",
                "Pegasus": "google/pegasus-xsum"
            }
            
            selected_hf_model = st.selectbox(
                "Hugging Face Model",
                options=list(hf_models.keys()),
                index=0
            )
            
            if st.button("Load HF Model"):
                with st.spinner(f"Loading {selected_hf_model}..."):
                    summarizer.hf_summarizer = summarizer.load_hf_model(hf_models[selected_hf_model])
                    if summarizer.hf_summarizer:
                        st.success("Model loaded successfully!")
                        st.session_state.hf_model_loaded = True
        
        else:  # Ollama Local
            ollama_models = summarizer.get_ollama_models()
            selected_ollama_model = st.selectbox(
                "Ollama Model",
                options=ollama_models,
                index=0
            )
            
            backend_url = st.text_input(
                "Ollama Backend URL",
                value="http://localhost:8000",
                help="Make sure your chat app backend is running"
            )
            summarizer.ollama_backend = backend_url
            
            # Test connection
            if st.button("Test Ollama Connection"):
                try:
                    response = requests.get(f"{backend_url}/models", timeout=5)
                    if response.status_code == 200:
                        st.success("✅ Connected to Ollama!")
                    else:
                        st.error("❌ Connection failed")
                except:
                    st.error("❌ Cannot connect to Ollama backend")
        
        st.divider()
        
        # Summary Settings
        st.subheader("📊 Summary Settings")
        
        if ai_backend == "Hugging Face (Cloud)":
            summary_type = st.radio(
                "Summary Type",
                ["Abstractive", "Extractive", "Both"],
                help="Abstractive: AI generates new sentences. Extractive: Selects key sentences."
            )
            
            if summary_type in ["Abstractive", "Both"]:
                max_length = st.slider("Max Summary Length", 50, 500, 150)
                min_length = st.slider("Min Summary Length", 20, 100, 50)
            
            if summary_type in ["Extractive", "Both"]:
                num_sentences = st.slider("Number of Key Sentences", 3, 10, 5)
        
        else:  # Ollama
            summary_length = st.slider("Summary Length (words)", 50, 500, 200)
        
        st.info(f"Device: {summarizer.device.upper()}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a document file",
            type=["pdf", "docx", "txt", "rtf"],
            help="Upload PDF, Word, or text documents"
        )
        
        # Direct text input
        st.markdown("**OR paste text directly:**")
        direct_text = st.text_area(
            "Paste your text here",
            height=150,
            placeholder="Paste your document text here..."
        )
        
        if direct_text:
            if st.button("Process Text", type="primary"):
                st.session_state.extracted_text = direct_text
                st.session_state.word_count = len(direct_text.split())
                st.success(f"Text processed! Word count: {st.session_state.word_count}")
        
        if uploaded_file is not None:
            st.info(f"**File**: {uploaded_file.name} ({uploaded_file.size} bytes)")
            
            if st.button("Extract Text", type="primary"):
                with st.spinner("Extracting text..."):
                    extracted_text = summarizer.extract_text_from_file(uploaded_file)
                    
                    if extracted_text:
                        st.session_state.extracted_text = summarizer.preprocess_text(extracted_text)
                        st.session_state.word_count = len(st.session_state.extracted_text.split())
                        st.success(f"Text extracted! Word count: {st.session_state.word_count}")
                    else:
                        st.error("Failed to extract text")
        
        # Text preview
        if "extracted_text" in st.session_state:
            st.subheader("📖 Text Preview")
            preview = st.session_state.extracted_text[:1000] + "..." if len(st.session_state.extracted_text) > 1000 else st.session_state.extracted_text
            st.text_area("Preview", preview, height=200, disabled=True)
    
    with col2:
        st.subheader("📋 Summary Results")
        
        if "extracted_text" in st.session_state:
            if st.button("Generate Summary", type="primary"):
                text = st.session_state.extracted_text
                
                if ai_backend == "Hugging Face (Cloud)":
                    # Check if model is loaded
                    if summary_type in ["Abstractive", "Both"] and not hasattr(st.session_state, 'hf_model_loaded'):
                        st.error("Please load a Hugging Face model first!")
                    else:
                        summaries = {}
                        
                        if summary_type in ["Abstractive", "Both"]:
                            with st.spinner("Generating abstractive summary..."):
                                abs_summary = summarizer.hf_abstractive_summarization(
                                    text, max_length=max_length, min_length=min_length
                                )
                                if abs_summary:
                                    summaries["Abstractive (HF)"] = abs_summary
                        
                        if summary_type in ["Extractive", "Both"]:
                            with st.spinner("Generating extractive summary..."):
                                ext_summary = summarizer.extractive_summarization(
                                    text, num_sentences=num_sentences
                                )
                                if ext_summary:
                                    summaries["Extractive"] = ext_summary
                        
                        st.session_state.summaries = summaries
                
                else:  # Ollama Local
                    with st.spinner(f"Generating summary with {selected_ollama_model}..."):
                        ollama_summary = summarizer.ollama_summarize(
                            text, model=selected_ollama_model, max_length=summary_length
                        )
                        if ollama_summary:
                            st.session_state.summaries = {f"Ollama ({selected_ollama_model})": ollama_summary}
        
        # Display summaries
        if "summaries" in st.session_state:
            for summary_type_name, summary_text in st.session_state.summaries.items():
                st.subheader(f"🤖 {summary_type_name}")
                st.write(summary_text)
                
                # Stats
                word_count = len(summary_text.split())
                if "word_count" in st.session_state:
                    original_count = st.session_state.word_count
                    compression_ratio = round((1 - word_count/original_count) * 100, 1)
                    
                    st.metric(
                        "Compression", 
                        f"{compression_ratio}%", 
                        f"{word_count}/{original_count} words"
                    )
                
                # Download
                st.download_button(
                    label=f"Download {summary_type_name} Summary",
                    data=summary_text,
                    file_name=f"{summary_type_name.lower().replace(' ', '_')}_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
                
                st.divider()
    
    # Instructions
    with st.expander("📋 Usage Instructions"):
        st.markdown("""
        ### How to Use:
        
        1. **Choose AI Backend:**
           - **Hugging Face**: Better quality, needs internet, uses GPU/CPU
           - **Ollama**: Private, offline, uses your local models
        
        2. **For Hugging Face:**
           - Select and load a model (BART recommended)
           - Choose summary type (Abstractive/Extractive/Both)
           - Adjust length parameters
        
        3. **For Ollama:**
           - Make sure your chat app backend is running
           - Select your local model
           - Set summary length
        
        4. **Upload Document:**
           - Supports: PDF, Word (.docx), Text (.txt), RTF
           - Or paste text directly
        
        5. **Generate Summary:**
           - Click "Generate Summary"
           - Download results
        
        ### Supported File Types:
        - **PDF**: Most PDF documents (not scanned images)
        - **Word**: .docx files
        - **Text**: .txt, .rtf files
        - **Direct paste**: Any text content
        
        ### AI Backends:
        - **Hugging Face**: BART, T5, Pegasus models
        - **Ollama**: Your local LLMs (llama2, mistral, etc.)
        """)

if __name__ == "__main__":
    main()
