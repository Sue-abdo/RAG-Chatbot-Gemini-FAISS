import os
import sys
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if w.lower() not in stop_words]
    return " ".join(filtered)

# --- Configuration ---
# Use a fast, small model for embeddings
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' 
# Use a capable model for generation
GEMINI_MODEL_NAME = 'gemini-2.5-flash'
# Limit the dataset size for quick setup (5000 examples)
DATASET_SUBSET_SIZE = 5000
# Name of the FAISS index to build
FAISS_INDEX_NAME = 'embeddings'

class RAGChatbot:
    """A simple terminal chatbot that uses RAG on the UltraChat dataset."""

    def __init__(self):
        """Initialize the model and load the RAG index."""
        print("Initializing RAG Chatbot...")
        
        # 1. Initialize LLM Client and check API key
        self.api_key = os.environ.get("soha")  # Change to your environment variable name

        if not self.api_key:
            # If the key is not set, print an error and exit the program
            print("=========================================================")
            print(" !!! ERROR: GENAI_API_KEY environment variable not set.")
            print("  Please set your key before running the script.")
            # Reminder for Windows (PowerShell) users:
            print("  E.g., in PowerShell:  $env:GENAI_API_KEY='YOUR_API_KEY'")
            print("=========================================================")
            sys.exit(1)
        else:
            print("API key found. Proceeding with initialization.")
        # The client will automatically pick up the environment variable
        self.client = genai.Client()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # 2. Load and Prepare the Dataset
        self.dataset = self._prepare_dataset()
        
        # 3. Build the RAG Index
        self._build_faiss_index()
        
        print("Initialization complete. Chatbot ready.")
        
        


    def _prepare_dataset(self) -> Dataset:
        """
        Cleans and prepares the UltraChat dataset by extracting Q/A pairs.
        
        The dataset is a list of multi-turn conversations. We extract the first
        user query and the first assistant response to create simple Q/A documents.
        """
        print(f"Loading UltraChat dataset subset (first {DATASET_SUBSET_SIZE} examples)...")
        # Load the dataset from Hugging Face
        raw_dataset = load_dataset(
            "HuggingFaceH4/ultrachat_200k", 
            split=f"train_sft[:{DATASET_SUBSET_SIZE}]" # Use the train_sft split
        )

        def extract_q_a(example):
            """Extracts the first user query and the first assistant response."""
            # Ensure the conversation is not empty
            if not example['messages']:
                return {'query': None, 'context': None}
            
            query = None
            context = None
            
            for msg in example['messages']:
                if msg['role'] == 'user' and query is None:
                    query = msg['content']
                elif msg['role'] == 'assistant' and context is None:
                    context = msg['content']
            
            # The context will be the response that the model retrieves
            return {'query': query, 'context': context}

        # Apply the extraction function
        processed_dataset = raw_dataset.map(extract_q_a, remove_columns=raw_dataset.column_names)
        
        # Filter out examples where extraction failed (e.g., empty conversations)
        processed_dataset = processed_dataset.filter(lambda x: x['query'] is not None and x['context'] is not None)
        
        print(f"Dataset prepared with {len(processed_dataset)} clean Q/A pairs.")
        return processed_dataset


    def _build_faiss_index(self):
        """
        Creates embeddings for the 'query' column and builds a FAISS index.
        """
        print("Generating embeddings and building FAISS index...")
        
        # 1. Function to get embeddings from the SentenceTransformer model
        def embed_batch(batch):
            cleaned_queries = [remove_stopwords(q) for q in batch['query']]
            return {
                'embedding': self.embedding_model.encode(
                cleaned_queries,
                convert_to_tensor=False
            ).tolist()
        }

        
        # 2. Add a new 'embedding' column to the dataset
        self.dataset = self.dataset.map(
            embed_batch, 
            batched=True, 
            batch_size=32
        )
        
        # 3. Add the FAISS index for fast similarity search
        self.dataset.add_faiss_index(
            column='embedding', 
            index_name=FAISS_INDEX_NAME
        )
        
        print("FAISS index built successfully.")


    def retrieve_context(self, user_query: str, k: int = 3) -> list[str]:
        """
        Searches the dataset for the top 'k' most relevant contexts using the RAG index.
        """
        # 1. Embed the user query
        query_embedding = self.embedding_model.encode(
            user_query, 
            convert_to_tensor=False
        )
        
        # 2. Search the FAISS index
        scores, retrieved_examples = self.dataset.get_nearest_examples(
            index_name=FAISS_INDEX_NAME, 
            query=query_embedding, 
            k=k
        )
        
        # 3. Extract the 'context' (assistant response) from the search results
        contexts = retrieved_examples['context']
        return contexts

    def generate_response(self, user_query: str, contexts: list[str]) -> str:
        """
        Generates a final answer using the LLM based on the retrieved contexts.
        (FIXED to correctly pass System Instruction via config object)
        """
        # 1. Combine retrieved contexts into a single block
        context_block = "\n---\n".join(contexts)
        
        # 2. Define the RAG Prompt Template
        # NOTE: The system_prompt content is now passed in the config object (Step 4)
        system_prompt = (
            "You are a helpful assistant. Use the following retrieved data "
            "as context to answer the user's question. If the data does not "
            "contain the answer, state that you could not find the information "
            "in the available resources, but still try to give a helpful general answer. "
            "Do not mention that you are using a dataset or external knowledge base. "
            "Be concise and professional."
        )
        
        # This is the single prompt containing context and query, which goes in the contents list.
        full_prompt = (
            f"Context:\n{context_block}\n\n"
            f"User Query: {user_query}\n\n"
            f"Answer:"
        )
        
        # 3. Create the configuration object for the System Instruction
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,  # <-- NEW WAY: Pass the system prompt here
        )
        
        # 4. Call the Gemini API
        try:
            # We now only pass the user/RAG prompt in the contents list.
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=[full_prompt],  # <-- NEW WAY: Contents is just a list of the user/RAG prompt
                config=config,           # <-- NEW WAY: Pass the config object
            )
            return response.text
        except Exception as e:
            # Catch potential errors like API key issues or network failures
            return f"An API error occurred: {e}"


    def run_terminal_loop(self):
        """
        The main loop for the terminal-based chatbot.
        """
        print("\n--- Simple RAG Chatbot Terminal ---")
        print("Enter 'quit' or 'exit' to end the chat.")
        
        while True:
            # 1. Get user input
            user_input = input("\nYou: ")
            
            if user_input.lower() in ['quit', 'exit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue

            print("Bot: Thinking... (Retrieving context and generating response)")

            # 2. RAG Step 1: Retrieve context
            retrieved_contexts = self.retrieve_context(user_input, k=3)
            
            # 3. RAG Step 2: Generate final response
            final_answer = self.generate_response(user_input, retrieved_contexts)
            
            # 4. Print the result
            print(f"\nBot: {final_answer}")


if __name__ == "__main__":
    # Create the chatbot instance, which handles setup
    rag_bot = RAGChatbot()
    # Start the terminal loop
    rag_bot.run_terminal_loop()
