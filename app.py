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
from io import BytesIO # Needed for BytesIO object

# Import project modules
# Ensure these modules exist and are importable in your environment
try:
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
        MATERIAL_PROMPT, MATERIAL_NAME_PROMPT,
        # Physical / Mechanical Attributes
        PULL_TO_SEAT_PROMPT, GENDER_PROMPT, HEIGHT_MM_PROMPT, LENGTH_MM_PROMPT,
        WIDTH_MM_PROMPT, NUMBER_OF_CAVITIES_PROMPT, NUMBER_OF_ROWS_PROMPT,
        MECHANICAL_CODING_PROMPT, COLOUR_PROMPT, COLOUR_CODING_PROMPT,
        # Sealing & Environmental
        WORKING_TEMPERATURE_PROMPT, HOUSING_SEAL_PROMPT, WIRE_SEAL_PROMPT,
        SEALING_PROMPT, SEALING_CLASS_PROMPT,
        # Terminals & Connections
        CONTACT_SYSTEMS_PROMPT, TERMINAL_POSITION_ASSURANCE_PROMPT,
        CONNECTOR_POSITION_ASSURANCE_PROMPT, CLOSED_CAVITIES_PROMPT,
        # Assembly & Type
        PRE_ASSEMBLED_PROMPT, CONNECTOR_TYPE_PROMPT, SET_KIT_PROMPT,
        # Specialized Attributes
        HV_QUALIFIED_PROMPT
    )
    # Define the prompts_to_run dictionary globally or ensure it's defined before use
    prompts_to_run = {
        "Material Filling": MATERIAL_PROMPT, "Material Name": MATERIAL_NAME_PROMPT, "Pull-to-Seat": PULL_TO_SEAT_PROMPT,
        "Gender": GENDER_PROMPT, "Number of Cavities": NUMBER_OF_CAVITIES_PROMPT, "Number of Rows": NUMBER_OF_ROWS_PROMPT,
        "Mechanical Coding": MECHANICAL_CODING_PROMPT, "Colour": COLOUR_PROMPT, "Colour Coding": COLOUR_CODING_PROMPT,
        "Working Temperature": WORKING_TEMPERATURE_PROMPT, "Housing Seal": HOUSING_SEAL_PROMPT, "Wire Seal": WIRE_SEAL_PROMPT,
        "Sealing": SEALING_PROMPT, "Sealing Class": SEALING_CLASS_PROMPT, "Contact Systems": CONTACT_SYSTEMS_PROMPT,
        "Terminal Position Assurance": TERMINAL_POSITION_ASSURANCE_PROMPT, "Connector Position Assurance": CONNECTOR_POSITION_ASSURANCE_PROMPT,
        "Closed Cavities": CLOSED_CAVITIES_PROMPT, "Pre-Assembled": PRE_ASSEMBLED_PROMPT, "Type of Connector": CONNECTOR_TYPE_PROMPT,
        "Set/Kit": SET_KIT_PROMPT, "HV Qualified": HV_QUALIFIED_PROMPT
    }
except ImportError as e:
    st.error(f"Failed to import required modules: {e}. Please ensure 'config.py', 'pdf_processor.py', 'vector_store.py', 'llm_interface.py', and 'extraction_prompts.py' are available.")
    st.stop()


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="LEONI LEOHAJJA Extraction", # Updated title
    page_icon="📄", # Maybe use a LEONI logo if available?
    layout="wide"
)

# --- Custom CSS ---
# Removed chat-specific CSS, kept result card styles
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
        word-wrap: break-word; /* Allow long values to wrap */
    }

    /* Style for chat messages within the sidebar */
    .chat-messages-sidebar {
        height: 60vh; /* Adjust height as needed */
        overflow-y: auto;
        border: 1px solid #ddd;
        background-color: white;
        padding: 10px;
        margin-bottom: 10px;
        border-radius: 5px;
    }

</style>
""", unsafe_allow_html=True)


# --- Logging Configuration ---
# logger.add("logs/app_{time}.log", rotation="10 MB", level="INFO") # Consider re-enabling if needed

# --- Application State ---
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'extraction_chain' not in st.session_state:
    st.session_state.extraction_chain = None
if 'processed_files_info' not in st.session_state:
    st.session_state.processed_files_info = []
if 'current_part_number' not in st.session_state:
    st.session_state.current_part_number = None
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'extraction_performed' not in st.session_state:
    st.session_state.extraction_performed = False
if 'show_chat' not in st.session_state:
    st.session_state.show_chat = False # State to control chat sidebar visibility
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []


# --- Global Variables / Initialization ---
# Use @st.cache_resource for expensive initializations
@st.cache_resource
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

@st.cache_resource
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

# --- Functions ---
def reset_processing_state():
    st.session_state.retriever = None
    st.session_state.extraction_chain = None
    st.session_state.evaluation_results = []
    st.session_state.extraction_performed = False
    st.session_state.show_chat = False # Close chat when reprocessing
    st.session_state.chat_history = [] # Clear chat history on re-process
    # Keep part number and file info? Let's keep them unless files are cleared.

# --- Load existing vector store (Optional) ---
# This logic might need adjustment based on whether you want persistence across sessions
if config.CHROMA_SETTINGS.is_persistent and 'retriever_loaded' not in st.session_state and embedding_function and llm:
    logger.info("Attempting to load existing vector store...")
    retriever = load_existing_vector_store(embedding_function)
    if retriever:
        st.session_state.retriever = retriever
        st.session_state.retriever_loaded = True # Flag that we loaded successfully
        logger.success("Successfully loaded retriever from persistent storage.")
        # Note: We still require users to upload files and specify a part number
        # to perform a *new* extraction, even if a store is loaded.
        # Display a message indicating this?
        st.sidebar.info("Existing knowledge base loaded. Upload files and enter a part number to start a new extraction.")
    else:
        logger.warning("No existing persistent vector store found or failed to load.")
        st.session_state.retriever_loaded = False


# --- Header ---
header_cols = st.columns([1, 5, 1])
with header_cols[0]:
    st.markdown("##### LEONI")
    st.markdown("##### LEOHAJJA")
with header_cols[2]:
    # Only show chat button if extraction has been done (retriever exists)
    if st.session_state.retriever:
        chat_button_label = "Hide Chat" if st.session_state.show_chat else "Ask Chatbot"
        if st.button(chat_button_label, key="toggle_chat_button"):
            st.session_state.show_chat = not st.session_state.show_chat
            st.rerun() # Rerun to update sidebar visibility immediately
    else:
        st.button("ASK", key="ask_button_disabled", disabled=True)


# --- Chatbot Sidebar ---
# This block now uses st.sidebar and only renders if show_chat is True
if st.session_state.show_chat and st.session_state.retriever:
    with st.sidebar:
        st.markdown("<h4>LEONI CHATBOT</h4>", unsafe_allow_html=True)

        # Message display area
        st.markdown('<div class="chat-messages-sidebar">', unsafe_allow_html=True)
        if not st.session_state.chat_history:
             st.caption("Ask questions about the processed document(s)...")
        for msg in st.session_state.chat_history:
            st.markdown(f"**{msg['role'].capitalize()}**: {msg['content']}")
        st.markdown('</div>', unsafe_allow_html=True)

        # Input area at the bottom of the sidebar
        user_query = st.text_input(
            "Ask about the document...",
            key="chat_input_sidebar", # Use a unique key
            placeholder="Type your question here"
        )
        send_button = st.button("Send", key="send_chat_sidebar") # Unique key

        if send_button and user_query:
            st.session_state.chat_history.append({"role": "user", "content": user_query})

            # --- Placeholder for Q&A Logic ---
            # You need to implement a Q&A function/chain here using
            # st.session_state.retriever and llm
            with st.spinner("Thinking..."):
                try:
                    # Example: Replace with your actual Q&A call
                    # response = run_qna_chain(user_query, st.session_state.retriever, llm)
                    time.sleep(1) # Simulate work
                    response = f"Placeholder response to: \"{user_query}\". (Q&A logic not implemented)"
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as qna_e:
                    logger.error(f"Error during Q&A: {qna_e}", exc_info=True)
                    st.session_state.chat_history.append({"role": "assistant", "content": "Sorry, I encountered an error while trying to answer."})
            # --- End Placeholder ---

            # Rerun to update chat display in the sidebar
            st.rerun()


# --- Main Content Area ---
st.divider()

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
            tooltip_text = f"Raw: {result.get('Raw Output', 'N/A')}" # Basic tooltip
            display_value = result.get('Extracted Value', 'N/A')
            status_icon = "❓" # Default icon

            if result.get('Is Rate Limit'):
                card_class += " result-card-error"
                display_value = "Rate Limit"
                status_icon = "⏳"
            elif result.get('Is Error'):
                card_class += " result-card-error"
                display_value = "Error"
                status_icon = "❌"
                if result.get('Parse Error'):
                    tooltip_text = f"Error: {result.get('Parse Error')}\nRaw: {result.get('Raw Output', 'N/A')}"
            elif result.get('Is Not Found'):
                 card_class += " result-card-notfound"
                 display_value = "NOT FOUND"
                 status_icon = "❔"
            elif result.get('Is Success'):
                 # display_value = result['Extracted Value'] # Already set
                 status_icon = "✔️"
            else: # Default/unknown state
                 # display_value = result['Extracted Value'] # Already set
                 pass # Keep default icon and value

            # Use markdown to apply the CSS class and add tooltip
            st.markdown(f"""
            <div class="{card_class}" title="{tooltip_text}">
                <span class="prompt-name">{result['Prompt Name']}</span>
                <span class="extracted-value">{display_value}</span>
                <span style="font-size: 1.5em;">{status_icon}</span>
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
        st.markdown("#### Upload File(s)")
        uploaded_files = st.file_uploader(
            "Select PDF documents for extraction",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader_main",
            label_visibility="collapsed" # Hide label, use markdown header instead
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) selected.")
            # Store file info immediately only if needed before button press
            # st.session_state.processed_files_info = [(f.name, f.getvalue()) for f in uploaded_files]

    part_number_input = None
    with part_num_col:
        st.markdown("#### Reference Part")
        part_number_input = st.text_input(
            "Enter Part Number associated with the document(s)",
            key="part_number_input_main",
            value=st.session_state.current_part_number or "",
            placeholder="e.g., 12345678",
            label_visibility="collapsed" # Hide label
        )
        if part_number_input != st.session_state.current_part_number:
             # Update state immediately if user types
             st.session_state.current_part_number = part_number_input
             # Optionally trigger something or just store it


    st.divider()

    # --- Processing Trigger ---
    process_col1, process_col2, process_col3 = st.columns([1, 1.5, 1]) # Give button more space
    with process_col2: # Center the button
        process_disabled = not uploaded_files or not st.session_state.current_part_number
        if st.button("✨ Process Documents & Extract Info", key="process_button_main", type="primary", disabled=process_disabled, use_container_width=True):
            if not embedding_function or not llm:
                st.error("Core components (Embeddings or LLM) failed to initialize. Cannot process documents.")
                st.stop()

            # --- Start Processing ---
            # Store file info from uploader *now*
            st.session_state.processed_files_info = [(f.name, f.getvalue()) for f in uploaded_files]

            # Reset state before processing *new* files
            reset_processing_state()
            # Re-assign part number after reset
            st.session_state.current_part_number = part_number_input # Get latest value from input

            filenames = [f[0] for f in st.session_state.processed_files_info]
            logger.info(f"Starting processing for {len(filenames)} files: {', '.join(filenames)} associated with Part Number: '{st.session_state.current_part_number}'")
            st.info(f"Processing {len(filenames)} file(s) for Part Number: {st.session_state.current_part_number}...")

            # --- PDF Processing ---
            processed_docs = None
            pdf_processing_error = None
            try:
                with st.spinner("Step 1/4: Processing PDFs... Loading & Cleaning..."):
                    start_time = time.time()
                    file_like_objects = []
                    for name, content_bytes in st.session_state.processed_files_info:
                        file_obj = BytesIO(content_bytes)
                        file_obj.name = name # PyMuPDFLoader might need the name attribute
                        file_like_objects.append(file_obj)

                    # Create a temporary directory if pdf_processor needs file paths
                    # temp_dir = os.path.join(os.getcwd(), "temp_pdf_files_streamlit")
                    # os.makedirs(temp_dir, exist_ok=True)
                    # Note: If process_uploaded_pdfs handles BytesIO directly, temp_dir might not be needed
                    # Assuming it handles BytesIO or similar file-like objects
                    processed_docs = process_uploaded_pdfs(file_like_objects, None) # Pass None if temp dir not needed

                    processing_time = time.time() - start_time
                    logger.info(f"PDF processing took {processing_time:.2f} seconds.")
                    if not processed_docs:
                        pdf_processing_error = "No document content could be extracted from the PDF(s)."
                        logger.warning(pdf_processing_error)
                    # Clean up temp dir if created
                    # if os.path.exists(temp_dir): shutil.rmtree(temp_dir)

            except Exception as e:
                logger.error(f"Failed during PDF processing phase: {e}", exc_info=True)
                pdf_processing_error = f"Error processing PDFs: {e}"
                st.error(pdf_processing_error)

            # --- Vector Store Setup ---
            vector_store_error = None
            if processed_docs and not pdf_processing_error:
                logger.info(f"Generated {len(processed_docs)} document chunks.")
                try:
                    with st.spinner("Step 2/4: Indexing documents in vector store..."):
                        start_time = time.time()
                        st.session_state.retriever = setup_vector_store(processed_docs, embedding_function)
                        indexing_time = time.time() - start_time
                        logger.info(f"Vector store setup took {indexing_time:.2f} seconds.")
                        if not st.session_state.retriever:
                            vector_store_error = "Failed to create a retriever from documents."
                            logger.error(vector_store_error)

                except Exception as e:
                    logger.error(f"Failed during vector store setup: {e}", exc_info=True)
                    vector_store_error = f"Error setting up vector store: {e}"
                    st.error(vector_store_error)
                    st.session_state.retriever = None # Ensure it's None on failure

            # --- Create Extraction Chain ---
            extraction_chain_error = None
            if st.session_state.retriever and not vector_store_error:
                logger.success("Vector store setup complete. Retriever is ready.")
                try:
                    with st.spinner("Step 3/4: Preparing extraction engine..."):
                        st.session_state.extraction_chain = create_extraction_chain(st.session_state.retriever, llm)
                        if st.session_state.extraction_chain:
                            logger.success("Extraction chain created.")
                        else:
                            extraction_chain_error = "Failed to create extraction chain (returned None)."
                            st.error(extraction_chain_error)
                except Exception as e:
                    logger.error(f"Failed creating extraction chain: {e}", exc_info=True)
                    extraction_chain_error = f"Error creating extraction chain: {e}"
                    st.error(extraction_chain_error)
                    st.session_state.extraction_chain = None

            # --- Run Extraction ---
            extraction_error = None
            if st.session_state.extraction_chain and not extraction_chain_error:
                logger.info(f"Starting extraction for {len(prompts_to_run)} attributes...")
                extraction_results_list = []
                extraction_successful_overall = True # Flag if *any* extraction succeeds
                num_prompts = len(prompts_to_run)

                progress_bar = st.progress(0, text="Step 4/4: Running extraction...")
                SLEEP_INTERVAL_SECONDS = 0.5 # Adjust based on API rate limits

                for i, (prompt_name, prompt_text) in enumerate(prompts_to_run.items()):
                    json_result_str = '{"error": "Extraction not run."}'
                    run_time = 0.0; parse_error = None; is_rate_limit = False
                    final_answer_value = "Error" # Default value

                    progress_text = f"Step 4/4: Running extraction... ({i+1}/{num_prompts}: {prompt_name})"
                    progress_bar.progress((i + 1) / num_prompts, text=progress_text)

                    try:
                        start_time_ext = time.time()
                        logger.debug(f"Running extraction for: {prompt_name}")
                        json_result_str = run_extraction(
                            prompt_text, prompt_name, st.session_state.current_part_number, st.session_state.extraction_chain
                        )
                        run_time = time.time() - start_time_ext
                        logger.info(f"Extraction for '{prompt_name}' took {run_time:.2f} seconds. Raw output: {json_result_str[:200]}...") # Log snippet
                        time.sleep(SLEEP_INTERVAL_SECONDS) # Pause between calls
                    except Exception as e:
                        logger.error(f"Error during extraction call for '{prompt_name}' (PN: {st.session_state.current_part_number}): {e}", exc_info=True)
                        json_result_str = f'{{"error": "LLM API call failed: {e}"}}'
                        # Don't mark overall extraction as failed yet, just this field

                    # --- Simple Parsing Logic ---
                    raw_llm_output = json_result_str
                    string_to_search = raw_llm_output.strip() if isinstance(raw_llm_output, str) else '{}'
                    parsed_json = None
                    is_success = False; is_error = False; is_not_found = False; is_rate_limit = False; parse_error = None

                    # Basic Cleaning (Handle potential markdown code blocks)
                    if string_to_search.startswith("```json"):
                        string_to_search = re.sub(r"^```json\s*", "", string_to_search)
                        string_to_search = re.sub(r"\s*```$", "", string_to_search)
                    elif string_to_search.startswith("```"):
                         string_to_search = re.sub(r"^```\s*", "", string_to_search)
                         string_to_search = re.sub(r"\s*```$", "", string_to_search)

                    string_to_search = string_to_search.strip()

                    # Try Parsing JSON
                    try:
                        # Look for the first valid JSON object/array
                        match = re.search(r'\{.*\}|\[.*\]', string_to_search, re.DOTALL)
                        json_string_to_parse = match.group(0) if match else string_to_search

                        if not json_string_to_parse: # Handle empty string after cleaning
                            raise json.JSONDecodeError("Empty string cannot be parsed", "", 0)

                        parsed_json = json.loads(json_string_to_parse)

                        if isinstance(parsed_json, dict):
                            actual_keys = list(parsed_json.keys())
                            if len(actual_keys) == 1: # Expected format: {"Prompt Name": "Value"}
                                actual_key = actual_keys[0]
                                value = parsed_json[actual_key]
                                # Handle potential nested structures or lists slightly better
                                if isinstance(value, list):
                                    final_answer_value = ", ".join(map(str, value)) if value else "Empty List"
                                elif isinstance(value, dict):
                                    final_answer_value = json.dumps(value) # Represent dict as string
                                else:
                                    final_answer_value = str(value)

                                if actual_key != prompt_name:
                                     logger.warning(f"Key mismatch for '{prompt_name}'. Expected '{prompt_name}', got '{actual_key}'. Using value anyway.")
                                # Determine status based on value
                                value_lower = final_answer_value.lower()
                                is_not_found = "not found" in value_lower or "n/a" == value_lower or "none" == value_lower
                                is_error = "error" in value_lower # Check if LLM explicitly returned an error value
                                is_success = not is_not_found and not is_error

                            elif "error" in parsed_json: # Format: {"error": "message"}
                                error_msg = str(parsed_json['error'])
                                final_answer_value = f"Error: {error_msg}"
                                is_rate_limit = "rate limit" in error_msg.lower()
                                is_error = not is_rate_limit # It's an error unless specifically rate limit
                                parse_error = ValueError(f"LLM Error: {error_msg}") # Log LLM's error

                            else: # Unexpected dictionary structure
                                final_answer_value = f"Unexpected Format: {json.dumps(parsed_json)}"
                                is_error = True
                                parse_error = ValueError(f"Expected single key or 'error' key, got: {actual_keys}")
                        else: # Got a list or other JSON type unexpectedly
                            final_answer_value = f"Unexpected Type: {json.dumps(parsed_json)}"
                            is_error = True
                            parse_error = TypeError(f"Expected dict, got {type(parsed_json)}")

                    except json.JSONDecodeError as json_err:
                        final_answer_value = "Invalid JSON Response"
                        is_error = True
                        parse_error = json_err
                        logger.warning(f"JSON Decode Error for '{prompt_name}': {json_err}. Raw text: '{string_to_search[:200]}...'")
                    except Exception as parse_exc:
                        final_answer_value = "Parsing Error"
                        is_error = True
                        parse_error = parse_exc
                        logger.error(f"Unexpected Parsing Error for '{prompt_name}': {parse_exc}", exc_info=True)

                    # --- Store Result ---
                    result_data = {
                        'Prompt Name': prompt_name,
                        'Extracted Value': final_answer_value,
                        'Raw Output': raw_llm_output,
                        'Parse Error': str(parse_error) if parse_error else None,
                        'Is Success': is_success,
                        'Is Error': is_error and not is_rate_limit, # Explicitly separate rate limit status
                        'Is Not Found': is_not_found,
                        'Is Rate Limit': is_rate_limit,
                        'Latency (s)': round(run_time, 2)
                    }
                    extraction_results_list.append(result_data)
                    if is_success: extraction_successful_overall = True # Mark overall success if at least one field worked

                # --- End Extraction Loop ---
                progress_bar.empty() # Remove progress bar

                if not extraction_results_list:
                    extraction_error = "Extraction ran but produced no results."
                    st.error(extraction_error)
                else:
                    st.session_state.evaluation_results = extraction_results_list
                    st.session_state.extraction_performed = True
                    logger.success(f"Extraction loop complete for Part Number: {st.session_state.current_part_number}. Results generated: {len(extraction_results_list)}")
                    st.success("Extraction complete!")
                    st.rerun() # Rerun to switch view to results grid

            # --- Handle intermediate failures ---
            elif extraction_chain_error:
                st.error(f"Could not run extraction because the engine failed to prepare: {extraction_chain_error}")
            elif vector_store_error:
                st.error(f"Could not run extraction because the document indexing failed: {vector_store_error}")
            elif pdf_processing_error:
                 st.error(f"Could not run extraction because PDF processing failed: {pdf_processing_error}")
            elif not uploaded_files: # Should be caught by disabled button, but as a safeguard
                st.warning("Please upload PDF files to process.")


# --- Footer or other sections (Optional) ---
# st.divider()
# st.caption("LEONI LEOHAJJA Extraction Tool")