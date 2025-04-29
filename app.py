# --- Force python to use pysqlite3 based on chromadb docs ---
# This override MUST happen before any other imports that might import sqlite3
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# --- End override ---

# app.py
import streamlit as st
import os
import time
from loguru import logger
import json # Import the json library
import pandas as pd # Add pandas import
import re # Import the 're' module for regular expressions

# Import project modules
import config
from pdf_processor import process_uploaded_pdfs
from vector_store import (
    get_embedding_function,
    setup_vector_store,
    load_existing_vector_store
)
# Updated imports from llm_interface
from llm_interface import (
    initialize_llm,
    create_extraction_chain,
    run_extraction
    # get_answer_from_llm_langchain # <-- Need to potentially add back for Q&A
)
# Import the prompts
from extraction_prompts import (
    # Material Properties
    MATERIAL_PROMPT,
    MATERIAL_NAME_PROMPT,
    # Physical / Mechanical Attributes
    PULL_TO_SEAT_PROMPT,
    GENDER_PROMPT,
    HEIGHT_MM_PROMPT,
    LENGTH_MM_PROMPT,
    WIDTH_MM_PROMPT,
    NUMBER_OF_CAVITIES_PROMPT,
    NUMBER_OF_ROWS_PROMPT,
    MECHANICAL_CODING_PROMPT,
    COLOUR_PROMPT,
    COLOUR_CODING_PROMPT,
    # Sealing & Environmental
    WORKING_TEMPERATURE_PROMPT,
    HOUSING_SEAL_PROMPT,
    WIRE_SEAL_PROMPT,
    SEALING_PROMPT,
    SEALING_CLASS_PROMPT,
    # Terminals & Connections
    CONTACT_SYSTEMS_PROMPT,
    TERMINAL_POSITION_ASSURANCE_PROMPT,
    CONNECTOR_POSITION_ASSURANCE_PROMPT,
    CLOSED_CAVITIES_PROMPT,
    # Assembly & Type
    PRE_ASSEMBLED_PROMPT,
    CONNECTOR_TYPE_PROMPT,
    SET_KIT_PROMPT,
    # Specialized Attributes
    HV_QUALIFIED_PROMPT
)


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LEONI LEOHAJJA Extraction", # Updated title
    page_icon="📄", # Maybe use a LEONI logo if available?
    layout="wide"
)

# --- Custom CSS (Minimal for Layout) ---
# You might need more sophisticated CSS for exact Figma matching
st.markdown("""
<style>
    /* Remove Streamlit Header/Footer */
    /* header {visibility: hidden;} */
    /* footer {visibility: hidden;} */
    /* #MainMenu {visibility: hidden;} */

    /* Basic styling for result cards */
    .result-card {
        background-color: #e8f5e9; /* Light green background */
        border: 1px solid #c8e6c9;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        text-align: center;
        color: #388e3c; /* Darker green text */
        font-weight: bold;
        min-height: 80px; /* Ensure cards have some height */
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .result-card-error {
        background-color: #ffebee; /* Light red background */
        border: 1px solid #ffcdd2;
        color: #c62828; /* Darker red text */
    }
    .result-card-notfound {
        background-color: #fffde7; /* Light yellow background */
        border: 1px solid #fff9c4;
        color: #f57f17; /* Darker yellow text */
    }
    .result-card .prompt-name {
        font-size: 0.9em;
        color: #555; /* Grey for prompt name */
        margin-bottom: 5px;
    }
    .result-card .extracted-value {
        font-size: 1.1em;
        word-wrap: break-word;
    }

    /* Chat placeholder styling */
    .chat-container {
        position: fixed; /* Or absolute, depending on desired scroll behavior */
        right: 20px;
        top: 80px; /* Adjust as needed */
        width: 350px; /* Adjust width */
        height: 80vh; /* Adjust height */
        background-color: #f0f0f0; /* Light grey background */
        border: 1px solid #ccc;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        z-index: 1000; /* Ensure it's above other content */
        display: flex;
        flex-direction: column;
    }
    .chat-container h4 {
        margin-top: 0;
        text-align: center;
        color: #333;
    }
    .chat-messages {
        flex-grow: 1;
        overflow-y: auto;
        border: 1px solid #ddd;
        background-color: white;
        padding: 10px;
        margin-bottom: 10px;
    }
    .chat-input {
        display: flex;
    }
    .chat-input .stTextInput {
        flex-grow: 1;
        margin-right: 5px;
    }

</style>
""", unsafe_allow_html=True)


# --- Logging Configuration ---
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO")

# --- Application State ---
# Use Streamlit's session state to hold persistent data across reruns
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'extraction_chain' not in st.session_state:
    st.session_state.extraction_chain = None
if 'processed_files_info' not in st.session_state:
    # Store tuples of (filename, file_content_bytes)
    st.session_state.processed_files_info = []
if 'current_part_number' not in st.session_state:
    st.session_state.current_part_number = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = [] # List to store detailed results per field
if 'extraction_performed' not in st.session_state:
    st.session_state.extraction_performed = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False # State to control chat visibility
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # To store chat messages


# --- Global Variables / Initialization ---
# @st.cache_resource # Keep caching for performance
def initialize_embeddings():
    logger.info("Attempting to initialize embedding function...")
    try:
    embeddings = get_embedding_function()
        logger.success("Embedding function initialized successfully.")
    return embeddings
except Exception as e:
    logger.error(f"Failed to initialize embeddings: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize embedding model. Please check logs or model configuration. Error: {e}")
    st.stop()

# @st.cache_resource # Keep caching for performance
def initialize_llm_cached():
    logger.info("Attempting to initialize LLM...")
try:
        llm_instance = initialize_llm()
        logger.success("LLM initialized successfully.")
        return llm_instance
except Exception as e:
     logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        st.error(f"Fatal Error: Could not initialize LLM. Please check API key and configuration. Error: {e}")
     st.stop()

embedding_function = initialize_embeddings()
llm = initialize_llm_cached()

# --- Load existing vector store or process uploads ---
# Reset state when processing new files
def reset_processing_state():
    st.session_state.retriever = None
    st.session_state.extraction_chain = None
    st.session_state.evaluation_results = []
    st.session_state.extraction_performed = False
    st.session_state.show_chat = False # Close chat when reprocessing
    # Keep part number and file info? Or reset them too? Let's keep them for now.
    # if 'gt_editor' in st.session_state: del st.session_state['gt_editor'] # No longer needed

# Try loading existing vector store (Optional, based on config)
# This might be less relevant if persistence isn't the primary goal now
if config.CHROMA_SETTINGS.is_persistent and st.session_state.retriever is None and embedding_function and llm:
    logger.info("Attempting to load existing vector store...")
    st.session_state.retriever = load_existing_vector_store(embedding_function)
    if st.session_state.retriever:
        logger.success("Successfully loaded retriever from persistent storage.")
        st.session_state.processed_files_info = [("Existing data loaded", None)] # Indicate loaded state
        logger.info("Creating extraction chain from loaded retriever...")
        st.session_state.extraction_chain = create_extraction_chain(st.session_state.retriever, llm)
        if st.session_state.extraction_chain:
            st.session_state.extraction_performed = True # Assume loaded data means extraction is 'done'
            # We don't have results to display unless they are also persisted, which they aren't currently.
            logger.warning("Loaded vector store, but extraction results are not persisted. Need to re-process files to see results.")
            st.session_state.extraction_performed = False # Mark as not performed so user must process
            st.session_state.retriever = None # Force reprocessing
        else:
            st.warning("Failed to create extraction chain from loaded retriever.")
            st.session_state.retriever = None # Force reprocessing
    else:
        logger.warning("No existing persistent vector store found or failed to load.")

# --- Header ---
header_cols = st.columns([1, 5, 1]) # Adjust ratios as needed
with header_cols[0]:
    st.markdown("##### LEONI")
    st.markdown("##### LEOHAJJA")
with header_cols[2]:
    ask_button_label = "ASK CHATBOT" if st.session_state.extraction_performed else "ASK"
    if st.button(ask_button_label, key="ask_button"):
        st.session_state.show_chat = not st.session_state.show_chat # Toggle chat visibility


# --- Main Content ---
st.divider() # Horizontal line like in Figma

# Check if extraction has been performed and results are available
if st.session_state.extraction_performed and st.session_state.evaluation_results:
    # --- STATE 2: Display Results Grid ---
    st.subheader("Extracted Information")
    if st.session_state.current_part_number:
        st.caption(f"Part Number: {st.session_state.current_part_number}")

    results = st.session_state.evaluation_results
    num_results = len(results)
    cols_per_row = 5 # As per Figma design (approx)
    num_rows = (num_results + cols_per_row - 1) // cols_per_row

    grid_cols = st.columns(cols_per_row)

    for i, result in enumerate(results):
        col_index = i % cols_per_row
        with grid_cols[col_index]:
            # Determine card style based on result status
            card_class = "result-card"
            if result.get('Is Rate Limit'):
                card_class += " result-card-error" # Or a specific grey style
                display_value = "Rate Limit"
            elif result.get('Is Error'):
                card_class += " result-card-error"
                display_value = "Error"
            elif result.get('Is Not Found'):
                 card_class += " result-card-notfound"
                 display_value = "NOT FOUND"
            elif result.get('Is Success'):
                 display_value = result['Extracted Value']
            else: # Default/unknown state
                 display_value = result['Extracted Value'] # Show value anyway

            # Use markdown to apply the CSS class
            st.markdown(f"""
            <div class="{card_class}">
                <span class="prompt-name">{result['Prompt Name']}</span>
                <span class="extracted-value">{display_value}</span>
                <span style="font-size: 1.5em;">✔️</span> <!-- Placeholder checkmark -->
            </div>
            """, unsafe_allow_html=True)

            # Optional: Expander for details (hidden by default)
            # with st.expander("Details", expanded=False):
            #    st.json(result) # Simple json dump for debugging

else:
    # --- STATE 1: Initial Upload / Processing ---
    st.markdown("### Bonjour!") # Welcome message

    upload_col, part_num_col = st.columns(2)

    uploaded_files = None
    with upload_col:
        st.markdown("#### Upload File")
        # Use a simple button first, then show uploader if clicked? Or just show uploader.
    uploaded_files = st.file_uploader(
            "Select PDF documents",
        type="pdf",
        accept_multiple_files=True,
            key="pdf_uploader_main",
            label_visibility="collapsed" # Hide label, use markdown header instead
        )
        if uploaded_files:
            # Store file info immediately in session state if needed
            # st.session_state.processed_files_info = [(f.name, f.getvalue()) for f in uploaded_files]
            st.success(f"{len(uploaded_files)} file(s) selected.")

    part_number_input = None
    with part_num_col:
        st.markdown("#### Reference Part")
        part_number_input = st.text_input(
            "Enter Part Number",
            key="part_number_input_main",
            value=st.session_state.current_part_number if st.session_state.current_part_number else "",
            label_visibility="collapsed" # Hide label
        )
        if part_number_input:
            st.session_state.current_part_number = part_number_input # Store immediately

    st.divider()

    # --- Processing Trigger ---
    process_col1, process_col2, process_col3 = st.columns([1,1,1])
    with process_col2: # Center the button
        process_disabled = not uploaded_files or not part_number_input
        if st.button("Process Documents", key="process_button_main", type="primary", disabled=process_disabled, use_container_width=True):
        if not embedding_function or not llm:
             st.error("Core components (Embeddings or LLM) failed to initialize earlier. Cannot process documents.")
        else:
                # Store file info before processing
                st.session_state.processed_files_info = [(f.name, f.getvalue()) for f in uploaded_files]

                # Reset state before processing
                reset_processing_state()
                # Need to re-assign part number after reset if reset clears it
                st.session_state.current_part_number = part_number_input

                filenames = [f[0] for f in st.session_state.processed_files_info]
                logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)} associated with Part Number: '{st.session_state.current_part_number}'")

            # --- PDF Processing ---
                processed_docs = None
                with st.spinner("Processing PDFs... Loading, cleaning..."):
                try:
                    start_time = time.time()
                        # We need file-like objects for process_uploaded_pdfs
                        # Create temporary in-memory files or save temporarily if needed
                        # For simplicity, let's modify process_uploaded_pdfs or pass bytes directly
                        # Assuming process_uploaded_pdfs can handle bytes or file paths
                        # This part needs careful handling based on pdf_processor implementation

                        # Create file-like objects for process_uploaded_pdfs
                        from io import BytesIO
                        file_like_objects = []
                        for name, content_bytes in st.session_state.processed_files_info:
                             file_obj = BytesIO(content_bytes)
                             file_obj.name = name # PyMuPDFLoader might need the name attribute
                             file_like_objects.append(file_obj)

                        temp_dir = os.path.join(os.getcwd(), "temp_pdf_files_streamlit") # Use different dir?
                        processed_docs = process_uploaded_pdfs(file_like_objects, temp_dir) # Pass file-like objects
                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                except Exception as e:
                    logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                    st.error(f"Error processing PDFs: {e}")

                # --- Vector Store Indexing & Chain Creation ---
            if processed_docs:
                    logger.info(f"Generated {len(processed_docs)} document chunks (Note: Chunking strategy might be simple).")
                with st.spinner("Indexing documents in vector store..."):
                    try:
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")
                        except Exception as e:
                            logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                            st.error(f"Error setting up vector store: {e}")
                            st.session_state.retriever = None # Ensure it's None on failure

                        if st.session_state.retriever:
                            logger.success("Vector store setup complete. Retriever is ready.")
                        # --- Create Extraction Chain ---
                            with st.spinner("Preparing extraction engine..."):
                            try:
                                 st.session_state.extraction_chain = create_extraction_chain(st.session_state.retriever, llm)
                            if st.session_state.extraction_chain:
                                logger.success("Extraction chain created.")
                            else:
                                    st.error("Failed to create extraction chain.")
                    except Exception as e:
                                logger.error(f"Failed creating extraction chain: {e}", exc_info=True)
                                st.error(f"Error creating extraction chain: {e}")

                        # --- Run Extraction ---
                        if st.session_state.extraction_chain:
                             with st.spinner(f"Running extraction for {len(prompts_to_run)} attributes..."):
                                # Reuse extraction logic from before (run_extraction function call)
                                prompts_to_run = { # Define prompts here or import
                                    "Material Filling": MATERIAL_PROMPT, "Material Name": MATERIAL_NAME_PROMPT, "Pull-to-Seat": PULL_TO_SEAT_PROMPT,
                                    "Gender": GENDER_PROMPT, "Number of Cavities": NUMBER_OF_CAVITIES_PROMPT, "Number of Rows": NUMBER_OF_ROWS_PROMPT,
                                    "Mechanical Coding": MECHANICAL_CODING_PROMPT, "Colour": COLOUR_PROMPT, "Colour Coding": COLOUR_CODING_PROMPT,
                                    "Working Temperature": WORKING_TEMPERATURE_PROMPT, "Housing Seal": HOUSING_SEAL_PROMPT, "Wire Seal": WIRE_SEAL_PROMPT,
                                    "Sealing": SEALING_PROMPT, "Sealing Class": SEALING_CLASS_PROMPT, "Contact Systems": CONTACT_SYSTEMS_PROMPT,
                                    "Terminal Position Assurance": TERMINAL_POSITION_ASSURANCE_PROMPT, "Connector Position Assurance": CONNECTOR_POSITION_ASSURANCE_PROMPT,
                                    "Closed Cavities": CLOSED_CAVITIES_PROMPT, "Pre-Assembled": PRE_ASSEMBLED_PROMPT, "Type of Connector": CONNECTOR_TYPE_PROMPT,
                                    "Set/Kit": SET_KIT_PROMPT, "HV Qualified": HV_QUALIFIED_PROMPT
                                }
                                extraction_results_list = []
                                extraction_successful = True
                                SLEEP_INTERVAL_SECONDS = 0.5 # Consider rate limits

        for prompt_name, prompt_text in prompts_to_run.items():
                json_result_str = '{"error": "Extraction not run."}'
                                    run_time = 0.0; parse_error = None; is_rate_limit = False
                                    final_answer_value = "Error" # Default value

                                    try:
                                        start_time_ext = time.time()
                                        json_result_str = run_extraction(
                                            prompt_text, prompt_name, st.session_state.current_part_number, st.session_state.extraction_chain
                                        )
                                        run_time = time.time() - start_time_ext
                        logger.info(f"Extraction for '{prompt_name}' took {run_time:.2f} seconds.")
                                        time.sleep(SLEEP_INTERVAL_SECONDS) # Delay
                    except Exception as e:
                                         logger.error(f"Error during extraction call for '{prompt_name}' (PN: {st.session_state.current_part_number}): {e}", exc_info=True)
                        json_result_str = f'{{"error": "Exception during extraction call: {e}"}}'
                                         extraction_successful = False # Mark as potentially failed

                                    # --- Simple Parsing Logic (same as before, adapted) ---
                                    raw_llm_output = json_result_str
                                    string_to_search = raw_llm_output.strip()
                                    parsed_json = None
                                    is_success = False; is_error = False; is_not_found = False; is_rate_limit = False; parse_error = None

                                    # Basic Cleaning
                    if string_to_search.startswith("```json"):
                                        string_to_search = string_to_search[7:-3] if string_to_search.endswith("```") else string_to_search[7:]
                                    string_to_search = string_to_search.strip()

                                    # Try Parsing
                                    try:
                        match = re.search(r'\{.*\}', string_to_search, re.DOTALL)
                                        json_string_to_parse = match.group(0) if match else string_to_search
                                        parsed_json = json.loads(json_string_to_parse)

                        if isinstance(parsed_json, dict):
                            actual_keys = list(parsed_json.keys())
                            if len(actual_keys) == 1:
                                actual_key = actual_keys[0]
                                                final_answer_value = str(list(parsed_json.values())[0])
                                                if actual_key != prompt_name: logger.warning(f"Key mismatch for '{prompt_name}'. Expected '{prompt_name}', got '{actual_key}'.")
                                                # Determine status
                                                is_not_found = "not found" in final_answer_value.lower()
                                                is_success = not is_not_found
                            elif "error" in parsed_json:
                                     error_msg = parsed_json['error']
                                     final_answer_value = f"Error: {error_msg}"
                                                 is_rate_limit = "rate limit" in error_msg.lower()
                                                 is_error = not is_rate_limit
                                                 parse_error = ValueError(f"LLM Error: {error_msg}")
                            else:
                                final_answer_value = "Unexpected JSON Format"
                                                 is_error = True
                                                 parse_error = ValueError(f"Wrong keys: {actual_keys}")
                        else:
                                            final_answer_value = "Unexpected JSON Type"
                                            is_error = True
                                            parse_error = TypeError(f"Expected dict, got {type(parsed_json)}")
                    except json.JSONDecodeError as json_err:
                                        final_answer_value = "Invalid JSON"
                                        is_error = True
                        parse_error = json_err
                    except Exception as parse_exc:
                        final_answer_value = "Parsing Error"
                                        is_error = True
                                        parse_error = parse_exc

                                    # --- Store Result ---
                    result_data = {
                        'Prompt Name': prompt_name,
                        'Extracted Value': final_answer_value,
                                        'Raw Output': raw_llm_output, # Keep raw for debugging
                        'Parse Error': str(parse_error) if parse_error else None,
                        'Is Success': is_success,
                                        'Is Error': is_error,
                        'Is Not Found': is_not_found,
                        'Is Rate Limit': is_rate_limit,
                                        'Latency (s)': round(run_time, 2)
                    }
                    extraction_results_list.append(result_data)
                                # --- End Extraction Loop ---

                                if extraction_successful or extraction_results_list: # Even if some failed, show results
                                     st.session_state.evaluation_results = extraction_results_list
                                     st.session_state.extraction_performed = True
                                     logger.success(f"Extraction complete for Part Number: {st.session_state.current_part_number}.")
                                     st.rerun() # Rerun to switch view to results grid
                                else:
                                     st.error("Extraction failed to produce results.")
                        else:
                             st.error("Extraction chain not available. Cannot run extraction.")
        else:
                         st.error("Vector store setup failed. Cannot proceed.")
                elif uploaded_files: # If files were uploaded but processing failed
                    st.error("Could not process the uploaded PDFs.")


# --- Chatbot Modal / Section ---
if st.session_state.show_chat:
    # Use markdown to inject the container div with CSS class
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    st.markdown("<h4>LEONI CHATBOT</h4>", unsafe_allow_html=True) # Use H4 for title

    # Message display area
    st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
    # Display chat history (simple version)
    for msg in st.session_state.chat_history:
        st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
    st.markdown('</div>', unsafe_allow_html=True)


    # Input area using columns for alignment
    # st.markdown('<div class="chat-input">', unsafe_allow_html=True) # CSS handles flex layout
    chat_input_cols = st.columns([10, 1])
    with chat_input_cols[0]:
         user_query = st.text_input("Ask about the document...", key="chat_input", placeholder="Placeholder text")
    with chat_input_cols[1]:
         send_button = st.button("➤", key="send_chat")
    # st.markdown('</div>', unsafe_allow_html=True)


    if send_button and user_query:
        # Add user query to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        # --- Placeholder for Q&A Logic ---
        # Need to re-create a Q&A chain or function here
        # Example:
        # with st.spinner("Thinking..."):
        #   try:
        #       # response = run_qna_chain(user_query, st.session_state.retriever, llm) # Fictional function
        #       response = "I am a placeholder chatbot response." # Placeholder
        #       st.session_state.chat_history.append({"role": "assistant", "content": response})
        #   except Exception as qna_e:
        #       logger.error(f"Error during Q&A: {qna_e}")
        #       st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error."})
        # --- End Placeholder ---
        st.session_state.chat_history.append({"role": "assistant", "content": f"Placeholder response to: {user_query}"})


        # Clear input and rerun to update display
        # st.session_state.chat_input = "" # This doesn't work directly with text_input key
        st.rerun()


    # Close the chat container div
    st.markdown('</div>', unsafe_allow_html=True)


# --- (Removed) Block 2: Display Ground Truth / Metrics ---
# --- (Removed) Block 3: Handle cases where extraction ran but yielded nothing ---
# --- (Removed) Evaluation Metrics Section ---
# --- (Removed) Export Section ---