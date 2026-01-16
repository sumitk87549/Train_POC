
import os
import sys
import pypdf
import psycopg2
import logging
import shutil
import numpy as np
import soundfile as sf
import json
from datetime import datetime
from pathlib import Path

# Add subdirectories to path
sys.path.append(str(Path(__file__).parent / "Translation"))
sys.path.append(str(Path(__file__).parent / "listen"))
sys.path.append(str(Path(__file__).parent / "summarize"))

# Import from existing scripts
try:
    import translate
    import listen
    
    if not hasattr(listen, 'VitsTokenizer') and listen.DEPS_OK:
        try:
            from transformers import VitsTokenizer
            listen.VitsTokenizer = VitsTokenizer
        except ImportError:
            pass

    from importlib.machinery import SourceFileLoader
    summary_module_path = str(Path(__file__).parent / "summarize" / "summary-generate.py")
    summary_gen = SourceFileLoader("summary_gen", summary_module_path).load_module()
except ImportError as e:
    logging.error(f"Failed to import helper modules: {e}")

# EPUB libraries
try:
    from ebooklib import epub
    import ebooklib
    from bs4 import BeautifulSoup
except ImportError:
    logging.warning("EbookLib or BeautifulSoup not installed. EPUB support disabled.")

# Database config
DB_PARAMS = {
    "host": "localhost",
    "user": "postgres",
    "password": "0000",
    "dbname": "book_processing_db"
}

class BookProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("BookProcessor")

    def get_ollama_models(self):
        """Fetch available models from Ollama."""
        try:
            import ollama
            models_info = ollama.list()
            # Handle different versions of ollama lib structure
            if 'models' in models_info:
                return [m['name'] for m in models_info['models']]
            return []
        except Exception as e:
            self.logger.warning(f"Could not list Ollama models: {e}")
            return ["qwen2.5:3b", "llama3.2:3b", "deepseek-r1:1.5b", "deepseek-r1:7b"] # Fallbacks

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        self.logger.info(f"Extracting text from PDF: {pdf_path}...")
        text = ""
        try:
            reader = pypdf.PdfReader(pdf_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            self.logger.error(f"Error reading PDF: {e}")
            raise
        return text

    def extract_text_from_epub(self, epub_path):
        """Extract text from an EPUB file."""
        self.logger.info(f"Extracting text from EPUB: {epub_path}...")
        text_blocks = []
        try:
            book = epub.read_epub(epub_path)
            for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
                try:
                    content = item.get_content().decode('utf-8')
                except UnicodeDecodeError:
                    content = item.get_content().decode('latin-1')
                
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                if text:
                    text_blocks.append(text)
            
            return "\n\n".join(text_blocks)
        except Exception as e:
            self.logger.error(f"Error reading EPUB: {e}")
            raise

    def clean_text(self, text):
        """Basic text cleaning."""
        self.logger.info("Cleaning text...")
        lines = [line.strip() for line in text.split('\n')]
        cleaned = '\n'.join(line for line in lines if line)
        return cleaned

    def _yield_stream_event(self, event_type, content):
        """Helper to yield structured events."""
        return {"type": event_type, "content": content}

    def stream_summary(self, text, model="qwen2.5:3b", provider="ollama", chunk_size=2000, temperature=0.3):
        """Generator that yields summary chunks with thinking support."""
        self.logger.info(f"Generating summary with {model} ({provider})...")
        
        try:
            # Use chunks
            chunks = summary_gen.chunk_text(text, chunk_words=chunk_size) 
            
            for i, chunk in enumerate(chunks[:5]): 
                yield self._yield_stream_event("section_header", f"**Chunk {i+1}/{min(5, len(chunks))} Analysis**")
                
                system_prompt = "You are a helpful assistant. Summarize this text concisely."
                user_prompt = f"Summarize this:\n{chunk}"
                
                if provider == "ollama":
                     import ollama
                     stream = ollama.chat(
                        model=model,
                        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                        options={"num_predict": 500, "temperature": temperature},
                        stream=True
                     )
                     
                     in_thinking = False
                     
                     for part in stream:
                         if 'message' in part and 'content' in part['message']:
                             token = part['message']['content']
                             
                             # Simple thinking parser
                             if '<think>' in token:
                                 in_thinking = True
                                 token = token.replace('<think>', '')
                                 yield self._yield_stream_event("thinking", "")
                             
                             if '</think>' in token:
                                 in_thinking = False
                                 pre, post = token.split('</think>', 1)
                                 if pre: yield self._yield_stream_event("thinking", pre)
                                 yield self._yield_stream_event("thinking_done", "")
                                 if post: yield self._yield_stream_event("content", post)
                                 continue

                             if in_thinking:
                                 yield self._yield_stream_event("thinking", token)
                             else:
                                 yield self._yield_stream_event("content", token)
                else:
                    # Non-streaming fallback 
                    prov = summary_gen.Provider(provider, model, "cuda" if listen.torch.cuda.is_available() else "cpu")
                    prov.load()
                    res = prov.generate(system_prompt, user_prompt, max_tokens=500)
                    yield self._yield_stream_event("content", res)

        except Exception as e:
            self.logger.error(f"Summarization failed: {e}")
            yield self._yield_stream_event("error", f"Summarization failed: {e}")

    def stream_translation(self, text, model="qwen2.5:3b", provider="ollama", chunk_size=500, temperature=0.3):
        """Generator that yields translation chunks with thinking support."""
        self.logger.info(f"Generating translation with {model} ({provider})...")
        
        try:
            chunks = translate.chunk_text(text, chunk_words=chunk_size)
            prompts = translate.TRANSLATION_PROMPTS["BASIC"]
            
            for i, chunk in enumerate(chunks[:5]): 
                yield self._yield_stream_event("section_header", f"**Translating Segment {i+1}/{min(5, len(chunks))}**")
                
                user_prompt = prompts["user"].format(chunk=chunk)
                
                if provider == "ollama":
                    import ollama
                    stream = ollama.chat(
                        model=model,
                        messages=[
                            {"role": "system", "content": prompts["system"]},
                            {"role": "user", "content": user_prompt}
                        ],
                        options={
                            "temperature": temperature,
                            "top_p": 0.9,
                            "num_ctx": 4096
                        },
                        stream=True
                    )
                    
                    in_thinking = False
                    
                    for part in stream:
                        if 'message' in part and 'content' in part['message']:
                            token = part['message']['content']
                            
                            # Thinking Logic
                            if '<think>' in token:
                                 in_thinking = True
                                 token = token.replace('<think>', '')
                                 yield self._yield_stream_event("thinking", "")
                            
                            if '</think>' in token:
                                 in_thinking = False
                                 pre_t, post_t = token.split('</think>', 1)
                                 if pre_t: yield self._yield_stream_event("thinking", pre_t)
                                 yield self._yield_stream_event("thinking_done", "")
                                 if post_t: yield self._yield_stream_event("content", post_t)
                                 continue
                            
                            if in_thinking:
                                yield self._yield_stream_event("thinking", token)
                            else:
                                yield self._yield_stream_event("content", token)

                else:
                     prov = translate.ModelProvider(provider, model, "cuda" if listen.torch.cuda.is_available() else "cpu")
                     prov.load_model()
                     res = prov.translate_streaming(prompts["system"], user_prompt, temperature, 0.9, 4096)
                     yield self._yield_stream_event("content", res)
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            yield self._yield_stream_event("error", f"Translation failed: {e}")

    def generate_audio_from_text(self, text, model="facebook/mms-tts-hin", provider="huggingface"):
        """Generate audio using soundfile."""
        self.logger.info(f"Generating audio with {model} ({provider})...")
        
        try:
            device = "cuda" if listen.torch.cuda.is_available() else "cpu"
            model_type = "vits"
            if "bark" in model: model_type = "bark"
            elif "speecht5" in model: model_type = "speecht5"
            elif provider == "coqui": model_type = "coqui"
            
            engine = listen.TTSEngine(model, model_type, device, "BASIC")
            engine.load_model()
            
            context_proc = listen.ContextProcessor("BASIC")
            chunks = context_proc.chunk_for_tts(text[:5000]) # Limit for POC
            
            temp_files = []
            output_dir = Path("temp_audio")
            output_dir.mkdir(exist_ok=True)
            
            all_audio_data = []
            sample_rate = 16000 
            
            for i, chunk in enumerate(chunks):
                out_path = output_dir / f"part_{i}.wav"
                engine.generate_audio(chunk, str(out_path))
                
                data, sr = sf.read(str(out_path))
                sample_rate = sr
                all_audio_data.append(data)
                
            if all_audio_data:
                final_audio = np.concatenate(all_audio_data)
                temp_wav = output_dir / "full.wav"
                sf.write(str(temp_wav), final_audio, sample_rate)
                
                final_path = output_dir / "final_audio.mp3"
                if shutil.which("ffmpeg"):
                       import subprocess
                       subprocess.run(["ffmpeg", "-y", "-i", str(temp_wav), "-b:a", "128k", str(final_path)], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                       if final_path.exists():
                           file_to_read = final_path
                       else:
                           file_to_read = temp_wav
                else:
                    file_to_read = temp_wav

                with open(file_to_read, "rb") as f:
                    audio_data = f.read()
                
                shutil.rmtree(output_dir)
                return audio_data
            return None
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_to_db(self, title, filename, cleaned_text, summary, translation, audio_data):
        self.logger.info(f"Saving '{title}' to database...")
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        query = """
        INSERT INTO processed_books (title, original_filename, cleaned_text, summary, translation, audio_data)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        try:
            cur.execute(query, (title, filename, cleaned_text, summary, translation, audio_data))
            book_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            self.logger.info(f"Saved with ID: {book_id}")
            return book_id
        except Exception as e:
            self.logger.error(f"DB Save error: {e}")
            if conn: conn.rollback()
            raise

    def get_all_books(self):
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT id, title, original_filename, created_at FROM processed_books ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()
        return rows

    def get_book_details(self, book_id):
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        cur.execute("SELECT title, cleaned_text, summary, translation, audio_data FROM processed_books WHERE id = %s", (book_id,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row
