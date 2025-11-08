"""
YouTube RAG Summarizer
A terminal-based application that uses LangChain, Gemini API, and FAISS 
to create a RAG system for YouTube video transcripts.
"""

import os
import sys
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
try:
    from youtube_transcript_api.formatters import TextFormatter
except ImportError:
    # Fallback for older versions
    pass
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables from .env file
load_dotenv()


def extract_video_id(youtube_url):
    """
    Extract the video ID from a YouTube URL.
    
    Args:
        youtube_url (str): The YouTube video URL
        
    Returns:
        str: The extracted video ID
        
    Raises:
        ValueError: If the URL format is invalid
    """
    # Handle different YouTube URL formats
    if "youtu.be/" in youtube_url:
        # Short URL format: https://youtu.be/VIDEO_ID
        video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    elif "youtube.com/watch?v=" in youtube_url:
        # Standard URL format: https://www.youtube.com/watch?v=VIDEO_ID
        video_id = youtube_url.split("v=")[1].split("&")[0]
    else:
        raise ValueError("Invalid YouTube URL format. Please provide a valid YouTube URL.")
    
    return video_id


def fetch_transcript(video_id):
    """
    Fetch the transcript of a YouTube video using its video ID.

    Args:
        video_id (str): The YouTube video ID

    Returns:
        str: The transcript text or an error message starting with "Error "
    """
    print(f"📥 Fetching transcript for video ID: {video_id}...")
    
    try:
        # Try English first, then fallback to any available language
        languages_to_try = [
            ['en'],
            ['en-US'],
            ['en-GB'],
            None  # Will try any available language
        ]
        
        for languages in languages_to_try:
            try:
                if languages:
                    transcript = YouTubeTranscriptApi.get_transcript(
                        video_id,
                        languages=languages
                    )
                else:
                    # Try without language specification (gets first available)
                    transcript = YouTubeTranscriptApi.get_transcript(video_id)
                
                # Successfully got transcript
                text = " ".join([entry['text'] for entry in transcript])
                print(f"✅ Transcript fetched successfully ({len(transcript)} segments)")
                return text
                
            except Exception:
                # Try next language option
                continue
        
        # If we get here, no transcript was available
        return "Error fetching transcript: No transcript available for this video."
        
    except Exception as e:
        error_msg = str(e).lower()
        if "disabled" in error_msg:
            return "Error fetching transcript: Transcripts are disabled for this video."
        elif "unavailable" in error_msg:
            return "Error fetching transcript: Video is unavailable."
        elif "not found" in error_msg or "no transcript" in error_msg:
            return "Error fetching transcript: No transcript found for this video."
        else:
            return f"Error fetching transcript: {str(e)}"


def split_transcript(transcript_text, chunk_size=1000, chunk_overlap=200):
    """
    Split the transcript into smaller chunks for processing.
    
    Args:
        transcript_text (str): The complete transcript text
        chunk_size (int): The size of each chunk
        chunk_overlap (int): The overlap between chunks
        
    Returns:
        list: List of text chunks
    """
    print(f"✂️  Splitting transcript into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Split the text into chunks
    chunks = text_splitter.split_text(transcript_text)
    
    print(f"✅ Transcript split into {len(chunks)} chunks")
    return chunks


def create_vector_store(text_chunks, api_key):
    """
    Create a FAISS vector store from text chunks using OpenAI embeddings.

    Args:
        text_chunks (list[str]): The list of transcript text chunks
        api_key (str): OpenAI API key

    Returns:
        FAISS | None: The vector store, or None on failure
    """
    try:
        embeddings = OpenAIEmbeddings(api_key=api_key)
        vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        print("✅ Vector store created successfully")
        return vector_store
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None


def get_llm(api_key):
    """Initialize the OpenAI LLM."""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.3
        )
        return llm
    except Exception as e:
        print(f"❌ Error initializing OpenAI LLM: {str(e)}")
        return None


def create_rag_chain(vector_store, api_key):
    """
    Create a RetrievalQA chain combining FAISS retriever and OpenAI LLM.
    
    Args:
        vector_store (FAISS): The FAISS vector store
        api_key (str): OpenAI API key
        
    Returns:
        RetrievalQA: The RAG chain
    """
    print("🔗 Setting up RAG chain with OpenAI LLM...")
    
    # Initialize OpenAI LLM
    llm = get_llm(api_key)
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 most relevant chunks
    )
    
    # Define a custom prompt template for better responses
    prompt_template = """You are a helpful AI assistant analyzing a YouTube video transcript. 
Use the following pieces of context from the video transcript to answer the user's question.
If you don't know the answer based on the context, just say that you don't have enough information.

Context from the video:
{context}

Question: {question}

Provide a detailed and helpful answer based on the video content:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    # Helper to format retrieved docs
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Build LCEL chain
    rag_chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    print("✅ RAG chain created successfully")
    return rag_chain


def main():
    """Main function to run the YouTube RAG summarizer."""
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ Error: OPENAI_API_KEY not found in environment variables")
        print("ℹ️ Please create a .env file with your OpenAI API key:")
        print("   OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    try:
        # Step 1: Get YouTube URL from user
        print("📝 Step 1: Enter YouTube Video URL")
        youtube_url = input("YouTube URL: ").strip()
        
        if not youtube_url:
            print("❌ Error: YouTube URL cannot be empty.")
            sys.exit(1)
        
        # Step 2: Get user prompt
        print("\n📝 Step 2: Enter Your Prompt")
        print("Examples: 'Summarize the video', 'What are the key points?', 'Explain the main concepts'")
        user_prompt = input("Your prompt: ").strip()
        
        if not user_prompt:
            print("❌ Error: Prompt cannot be empty.")
            sys.exit(1)
        
        print("\n" + "="*60)
        print("🚀 Processing...")
        print("="*60 + "\n")
        
        # Step 3: Extract video ID
        video_id = extract_video_id(youtube_url)
        
        # Step 4: Fetch transcript
        transcript = fetch_transcript(video_id)
        if transcript.startswith("Error "):
            raise RuntimeError(transcript)
        
        # Step 5: Split transcript into chunks
        text_chunks = split_transcript(transcript)
        
        # Step 6: Create vector store with embeddings
        vector_store = create_vector_store(text_chunks, api_key)
        if vector_store is None:
            raise RuntimeError("Failed to create vector store.")
        
        # Step 7: Create RAG chain
        qa_chain = create_rag_chain(vector_store, api_key)
        
        # Step 8: Generate response
        print(f"🤖 Generating response for: '{user_prompt}'...")
        print("\n" + "="*60)
        print("📄 RESPONSE")
        print("="*60 + "\n")
        
        response = qa_chain.invoke(user_prompt)
        print(response)
        
        print("\n" + "="*60)
        print("✅ Process completed successfully!")
        print("="*60 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
