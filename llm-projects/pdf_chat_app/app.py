
import streamlit as st
import PyPDF2
import io
import re
import json
import datetime
from typing import List, Dict, Optional, Tuple
from collections import Counter
import difflib
import numpy as np
import pandas as pd

# Enhanced imports for advanced features
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    st.warning("⚠️ For better search results, install: pip install sentence-transformers scikit-learn")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced PDF Chat Assistant", 
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedPDFChatAssistant:
    def __init__(self):
        self.documents = {}  # Store multiple documents
        self.current_doc_id = None
        self.conversation_context = []  # Store conversation history for context
        
        # Initialize embeddings model if available
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                self.use_embeddings = True
            except Exception as e:
                st.warning(f"Could not load embeddings model: {e}")
                self.use_embeddings = False
        else:
            self.use_embeddings = False
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        """Extract text from uploaded PDF file with enhanced error handling"""
        try:
            uploaded_file.seek(0)
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            metadata = {
                'total_pages': len(pdf_reader.pages),
                'extracted_pages': 0,
                'failed_pages': []
            }
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                        metadata['extracted_pages'] += 1
                except Exception as e:
                    metadata['failed_pages'].append(page_num + 1)
                    continue
            
            return text, metadata
        except Exception as e:
            st.error(f"Error reading PDF: {str(e)}")
            return "", {}
    
    def clean_and_chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Enhanced text cleaning and chunking with metadata"""
        # Clean the text
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', text)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        current_sentences = []
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if adding this sentence exceeds chunk size
            if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                chunk_data = {
                    'text': current_chunk.strip(),
                    'sentence_count': len(current_sentences),
                    'char_count': len(current_chunk),
                    'chunk_id': len(chunks)
                }
                chunks.append(chunk_data)
                
                # Create overlap
                overlap_sentences = current_sentences[-max(1, overlap//100):]
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_data = {
                'text': current_chunk.strip(),
                'sentence_count': len(current_sentences),
                'char_count': len(current_chunk),
                'chunk_id': len(chunks)
            }
            chunks.append(chunk_data)
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict]) -> np.ndarray:
        """Create embeddings for chunks if available"""
        if not self.use_embeddings:
            return None
        
        try:
            texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedder.encode(texts, show_progress_bar=True)
            return embeddings
        except Exception as e:
            st.error(f"Error creating embeddings: {e}")
            return None
    
    def add_document(self, name: str, uploaded_file) -> bool:
        """Process and add a new document"""
        with st.spinner(f"Processing {name}..."):
            # Extract text
            text, metadata = self.extract_text_from_pdf(uploaded_file)
            
            if not text.strip():
                st.error("Could not extract text from the PDF.")
                return False
            
            # Create chunks
            chunks = self.clean_and_chunk_text(text)
            
            # Create embeddings if available
            embeddings = self.create_embeddings(chunks) if self.use_embeddings else None
            
            # Store document
            doc_id = f"doc_{len(self.documents)}"
            self.documents[doc_id] = {
                'name': name,
                'text': text,
                'chunks': chunks,
                'embeddings': embeddings,
                'metadata': metadata,
                'upload_time': datetime.datetime.now(),
                'file_size': uploaded_file.size if hasattr(uploaded_file, 'size') else 0
            }
            
            self.current_doc_id = doc_id
            return True
    
    def search_with_embeddings(self, question: str, doc_id: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Search using semantic embeddings"""
        if not self.use_embeddings or doc_id not in self.documents:
            return []
        
        doc = self.documents[doc_id]
        if doc['embeddings'] is None:
            return []
        
        # Get question embedding
        question_embedding = self.embedder.encode([question])
        
        # Calculate similarities
        similarities = cosine_similarity(question_embedding, doc['embeddings'])[0]
        
        # Get top k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = []
        
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append((doc['chunks'][idx], similarities[idx]))
        
        return results
    
    def search_with_keywords(self, question: str, doc_id: str, max_chunks: int = 5) -> List[Tuple[Dict, float]]:
        """Enhanced keyword-based search"""
        if doc_id not in self.documents:
            return []
        
        doc = self.documents[doc_id]
        question_lower = question.lower()
        question_words = set(re.findall(r'\w+', question_lower))
        
        chunk_scores = []
        for chunk in doc['chunks']:
            chunk_lower = chunk['text'].lower()
            chunk_words = set(re.findall(r'\w+', chunk_lower))
            
            # Calculate various scores
            overlap = len(question_words.intersection(chunk_words))
            
            # Phrase matching bonus
            phrase_bonus = 5 if question_lower in chunk_lower else 0
            
            # Length penalty for very short chunks
            length_penalty = -2 if len(chunk['text']) < 50 else 0
            
            # Similarity bonus
            similarity_bonus = sum(
                1 for q_word in question_words 
                for c_word in chunk_words 
                if len(q_word) > 3 and len(c_word) > 3 
                and difflib.SequenceMatcher(None, q_word, c_word).ratio() > 0.8
            )
            
            total_score = overlap + phrase_bonus + similarity_bonus + length_penalty
            chunk_scores.append((chunk, total_score))
        
        # Sort and return top results
        chunk_scores.sort(reverse=True, key=lambda x: x[1])
        return [(chunk, score) for chunk, score in chunk_scores[:max_chunks] if score > 0]
    
    def get_relevant_content(self, question: str, doc_ids: List[str] = None) -> List[Dict]:
        """Get relevant content from one or multiple documents"""
        if not doc_ids:
            doc_ids = [self.current_doc_id] if self.current_doc_id else list(self.documents.keys())
        
        all_results = []
        
        for doc_id in doc_ids:
            if doc_id not in self.documents:
                continue
            
            # Use embeddings if available, otherwise use keyword search
            if self.use_embeddings:
                results = self.search_with_embeddings(question, doc_id)
            else:
                results = self.search_with_keywords(question, doc_id)
            
            # Add document info to results
            for chunk, score in results:
                all_results.append({
                    'chunk': chunk,
                    'score': score,
                    'doc_id': doc_id,
                    'doc_name': self.documents[doc_id]['name']
                })
        
        # Sort all results by score
        all_results.sort(reverse=True, key=lambda x: x['score'])
        return all_results[:5]  # Return top 5 across all documents
    
    def generate_answer_with_llm(self, question: str, context: str, api_key: str, model_type: str) -> str:
        """Generate answer using LLM (OpenAI or Anthropic)"""
        try:
            # Add conversation context for better responses
            context_prompt = ""
            if self.conversation_context:
                recent_context = self.conversation_context[-3:]  # Last 3 exchanges
                context_prompt = "\nRecent conversation:\n" + "\n".join([
                    f"Q: {ctx['question']}\nA: {ctx['answer'][:200]}..." 
                    for ctx in recent_context
                ]) + "\n"
            
            prompt = f"""Based on the following document content and conversation context, please provide a comprehensive and accurate answer to the question. If the answer cannot be found in the provided content, clearly state this.

{context_prompt}

Document Content:
{context}

Question: {question}

Instructions:
- Provide a clear, detailed answer based on the document content
- If information is not available, say so clearly
- Include relevant details and context when available
- Keep the answer focused and relevant to the question

Answer:"""
            
            if model_type == "OpenAI" and OPENAI_AVAILABLE:
                openai.api_key = api_key
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful document analysis assistant. Provide accurate answers based on the given document content."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=800,
                    temperature=0.1
                )
                return response.choices[0].message.content.strip()
            
            elif model_type == "Anthropic" and ANTHROPIC_AVAILABLE:
                client = anthropic.Client(api_key=api_key)
                response = client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=800,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
        except Exception as e:
            st.error(f"Error with {model_type} API: {str(e)}")
        
        return self.generate_fallback_answer(question, context)
    
    def generate_fallback_answer(self, question: str, context: str) -> str:
        """Enhanced fallback answer generation"""
        question_lower = question.lower()
        
        # Determine question type
        question_types = {
            'what': ['what', 'which'],
            'how': ['how'],
            'when': ['when'],
            'where': ['where'],
            'why': ['why'],
            'who': ['who'],
            'yes_no': ['is', 'are', 'does', 'do', 'can', 'will', 'should', 'would', 'could']
        }
        
        q_type = 'general'
        for qtype, keywords in question_types.items():
            if any(keyword in question_lower.split()[:3] for keyword in keywords):
                q_type = qtype
                break
        
        # Extract relevant sentences
        sentences = re.split(r'[.!?]+', context)
        question_words = set(re.findall(r'\w+', question_lower))
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            sentence_words = set(re.findall(r'\w+', sentence.lower()))
            overlap = len(question_words.intersection(sentence_words))
            
            # Bonus for sentence position and length
            length_bonus = 1 if 50 <= len(sentence) <= 200 else 0
            
            total_score = overlap + length_bonus
            if total_score > 0:
                scored_sentences.append((total_score, sentence))
        
        # Sort and select best sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        
        if scored_sentences:
            # Customize response based on question type
            if q_type == 'yes_no':
                prefix = "Based on the document, "
            elif q_type in ['what', 'which']:
                prefix = "According to the document, "
            elif q_type == 'how':
                prefix = "The document explains that "
            else:
                prefix = "The document indicates that "
            
            # Combine top sentences
            top_sentences = [sent[1] for sent in scored_sentences[:3]]
            answer = ". ".join(top_sentences)
            answer = re.sub(r'\s+', ' ', answer).strip()
            
            return f"{prefix}{answer}."
        
        return "I couldn't find specific information to answer this question in the uploaded document(s). Please try rephrasing your question or asking about different topics covered in the document."
    
    def get_answer(self, question: str, use_llm: bool = False, api_key: str = None, model_type: str = None, doc_ids: List[str] = None) -> str:
        """Main method to get answer to a question"""
        if not self.documents:
            return "❌ Please upload and process at least one PDF document first."
        
        # Get relevant content
        relevant_results = self.get_relevant_content(question, doc_ids)
        
        if not relevant_results:
            return "❌ I couldn't find relevant information to answer your question. Try rephrasing or asking about different topics."
        
        # Combine context from multiple sources
        context_parts = []
        for result in relevant_results:
            chunk_text = result['chunk']['text']
            doc_name = result['doc_name']
            context_parts.append(f"[From {doc_name}]: {chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        if use_llm and api_key and model_type:
            answer = self.generate_answer_with_llm(question, context, api_key, model_type)
        else:
            answer = self.generate_fallback_answer(question, context)
        
        # Store in conversation context
        self.conversation_context.append({
            'question': question,
            'answer': answer,
            'timestamp': datetime.datetime.now(),
            'sources': [r['doc_name'] for r in relevant_results]
        })
        
        # Add source information
        sources = list(set([r['doc_name'] for r in relevant_results]))
        if len(sources) > 1:
            answer += f"\n\n📚 *Sources: {', '.join(sources)}*"
        elif sources:
            answer += f"\n\n📚 *Source: {sources[0]}*"
        
        return answer
    
    def export_conversation(self, format_type: str = "json") -> str:
        """Export conversation history"""
        if not self.conversation_context:
            return ""
        
        if format_type == "json":
            export_data = {
                'export_date': datetime.datetime.now().isoformat(),
                'conversation_count': len(self.conversation_context),
                'documents': [doc['name'] for doc in self.documents.values()],
                'conversation': [
                    {
                        'question': ctx['question'],
                        'answer': ctx['answer'],
                        'timestamp': ctx['timestamp'].isoformat(),
                        'sources': ctx['sources']
                    }
                    for ctx in self.conversation_context
                ]
            }
            return json.dumps(export_data, indent=2)
        
        elif format_type == "markdown":
            md_content = f"# PDF Chat Conversation Export\n\n"
            md_content += f"**Export Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_content += f"**Documents:** {', '.join([doc['name'] for doc in self.documents.values()])}\n\n"
            md_content += "---\n\n"
            
            for i, ctx in enumerate(self.conversation_context, 1):
                md_content += f"## Question {i}\n\n"
                md_content += f"**Q:** {ctx['question']}\n\n"
                md_content += f"**A:** {ctx['answer']}\n\n"
                md_content += f"*Time: {ctx['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}*\n\n"
                md_content += "---\n\n"
            
            return md_content

def main():
    st.title("🚀 Advanced PDF Chat Assistant")
    st.markdown("Upload PDF documents and have intelligent conversations with enhanced AI capabilities!")
    
    # Initialize session state
    if 'assistant' not in st.session_state:
        st.session_state.assistant = AdvancedPDFChatAssistant()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for settings and document management
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # LLM Configuration
        use_llm = st.checkbox("🧠 Use Advanced LLM", help="Enable OpenAI or Anthropic for better answers")
        
        if use_llm:
            model_type = st.selectbox("Choose LLM Provider", ["OpenAI", "Anthropic"])
            api_key = st.text_input(f"{model_type} API Key", type="password", help=f"Enter your {model_type} API key")
        else:
            model_type = None
            api_key = None
        
        st.divider()
        
        # Document Management
        st.header("📚 Documents")
        
        # Upload new document
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF documents"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if st.button(f"📄 Process {uploaded_file.name}", key=f"process_{uploaded_file.name}"):
                    if st.session_state.assistant.add_document(uploaded_file.name, uploaded_file):
                        st.success(f"✅ {uploaded_file.name} processed!")
                        st.rerun()
        
        # Show processed documents
        if st.session_state.assistant.documents:
            st.subheader("📋 Processed Documents")
            
            for doc_id, doc in st.session_state.assistant.documents.items():
                with st.expander(f"📄 {doc['name']}", expanded=False):
                    st.write(f"**Pages:** {doc['metadata'].get('total_pages', 'N/A')}")
                    st.write(f"**Chunks:** {len(doc['chunks'])}")
                    st.write(f"**Upload:** {doc['upload_time'].strftime('%Y-%m-%d %H:%M')}")
                    
                    if st.button(f"🗑️ Remove", key=f"remove_{doc_id}"):
                        del st.session_state.assistant.documents[doc_id]
                        if st.session_state.assistant.current_doc_id == doc_id:
                            st.session_state.assistant.current_doc_id = None
                        st.rerun()
        
        st.divider()
        
        # Export options
        if st.session_state.assistant.conversation_context:
            st.header("💾 Export Chat")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📄 Export JSON"):
                    json_data = st.session_state.assistant.export_conversation("json")
                    st.download_button(
                        "⬇️ Download JSON",
                        json_data,
                        file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                if st.button("📝 Export MD"):
                    md_data = st.session_state.assistant.export_conversation("markdown")
                    st.download_button(
                        "⬇️ Download MD",
                        md_data,
                        file_name=f"chat_export_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
    
    # Main chat interface
    if not st.session_state.assistant.documents:
        st.info("👈 Please upload and process PDF documents in the sidebar to start chatting.")
        
        # Show features
        st.markdown("## 🌟 Enhanced Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🔍 Smart Search**
            - Semantic similarity search
            - Advanced keyword matching
            - Multi-document search
            """)
        
        with col2:
            st.markdown("""
            **🧠 AI Integration**
            - OpenAI GPT integration
            - Anthropic Claude support
            - Conversation context
            """)
        
        with col3:
            st.markdown("""
            **📊 Advanced Features**
            - Multiple document support
            - Export conversations
            - Document analytics
            """)
    
    else:
        # Chat interface
        st.header("💬 Chat Interface")
        
        # Document selector for multi-doc search
        if len(st.session_state.assistant.documents) > 1:
            doc_options = ["All Documents"] + [doc['name'] for doc in st.session_state.assistant.documents.values()]
            selected_docs = st.multiselect(
                "Search in documents:",
                doc_options,
                default=["All Documents"]
            )
            
            if "All Documents" in selected_docs:
                search_doc_ids = None
            else:
                search_doc_ids = [
                    doc_id for doc_id, doc in st.session_state.assistant.documents.items()
                    if doc['name'] in selected_docs
                ]
        else:
            search_doc_ids = None
        
        # Display chat history
        for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").write(f"🧑 **{timestamp}**\n\n{message}")
            else:
                st.chat_message("assistant").write(f"🤖 **{timestamp}**\n\n{message}")
        
        # Chat input
        question = st.chat_input("Ask a question about your documents...")
        
        if question:
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Add user question
            st.session_state.chat_history.append(("user", question, timestamp))
            
            # Get answer
            with st.spinner("🤔 Analyzing documents..."):
                answer = st.session_state.assistant.get_answer(
                    question=question,
                    use_llm=use_llm,
                    api_key=api_key,
                    model_type=model_type,
                    doc_ids=search_doc_ids
                )
            
            # Add assistant answer
            st.session_state.chat_history.append(("assistant", answer, timestamp))
            st.rerun()
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.session_state.assistant.conversation_context = []
                st.rerun()
        
        with col2:
            if st.button("📊 Chat Stats"):
                if st.session_state.assistant.conversation_context:
                    st.info(f"💬 Questions asked: {len(st.session_state.assistant.conversation_context)}")
                    sources = set()
                    for ctx in st.session_state.assistant.conversation_context:
                        sources.update(ctx['sources'])
                    st.info(f"📚 Documents referenced: {len(sources)}")
        
        # Quick suggestions
        if st.session_state.assistant.conversation_context:
            st.markdown("### 💡 Smart Suggestions")
            suggestions = [
                "Can you summarize the main points from all documents?",
                "What are the key differences between the uploaded documents?",
                "Are there any contradictions between the documents?",
                "What conclusions can be drawn from the information?"
            ]
            
            cols = st.columns(2)
            for i, suggestion in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(f"💭 {suggestion}", key=f"suggestion_{i}"):
                        # Simulate clicking with the suggestion
                        st.session_state.chat_history.append(("user", suggestion, datetime.datetime.now().strftime("%H:%M:%S")))
                        answer = st.session_state.assistant.get_answer(
                            question=suggestion,
                            use_llm=use_llm,
                            api_key=api_key,
                            model_type=model_type,
                            doc_ids=search_doc_ids
                        )
                        st.session_state.chat_history.append(("assistant", answer, datetime.datetime.now().strftime("%H:%M:%S")))
                        st.rerun()

if __name__ == "__main__":
    main()
