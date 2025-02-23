import os
import json
import boto3
from dotenv import load_dotenv
from typing import Generator, Dict, Any, List
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize the Bedrock Runtime client
client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY")
)

MODEL_ID = "us.amazon.nova-lite-v1:0"

def get_llm_output(user_input: str) -> Dict[str, Any]:
    """
    Calls the AWS Bedrock API to extract food items, microbes and metabolites from the user's input.
    """
    system_prompt = [
        {
            "text": """
            You are an advanced text analysis model specializing in biochemical analysis.
            
            Your task is to extract the following details from the user's input:
                1️⃣ **Food items mentioned**  
                2️⃣ **Dominating microbes linked to those foods**  
                3️⃣ **Metabolites produced by those microbes (including HMDB ID & concentration)**  
                
            Guidelines:
            - Include **only metabolites with valid HMDB IDs**
            - Each metabolite should have:
                ✅ Name  
                ✅ HMDB ID  
                ✅ Relative concentration (a float between 0 and 1)
                
            Your output must be in JSON format:
            {
                "food_items": ["food1", "food2", ...],
                "microbes": ["microbe1", "microbe2", ...],
                "metabolites": [
                    {
                        "name": "...",
                        "hmdb_id": "...",
                        "concentration": ...
                    },
                    ...
                ]
            }
            Respond only with valid JSON and nothing else.
            """
        }
    ]

    messages = [{"role": "user", "content": [{"text": user_input}]}]
    inference_config = {"maxTokens": 500, "temperature": 0.0, "topP": 0.9, "topK": 20}

    payload = {
        "schemaVersion": "messages-v1",
        "system": system_prompt,
        "messages": messages,
        "inferenceConfig": inference_config
    }

    try:
        # Invoke the model
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        content = json.loads(response["body"].read().decode("utf-8"))["output"]["message"]["content"][0]["text"]

        # Parse JSON response
        try:
            parsed = json.loads(content)
            
            # Convert concentration to amount in metabolites
            metabolites = parsed.get("metabolites", [])
            for m in metabolites:
                if "concentration" in m:
                    m["amount"] = m.pop("concentration")
                    
            # Return the complete structure
            return parsed

        except json.JSONDecodeError:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start != -1 and json_end != -1:
                parsed = json.loads(content[json_start:json_end])
                
                # Convert concentration to amount here too
                metabolites = parsed.get("metabolites", [])
                for m in metabolites:
                    if "concentration" in m:
                        m["amount"] = m.pop("concentration")
                        
                return parsed
            else:
                raise ValueError("Could not find valid JSON in response")

    except Exception as e:
        logger.error(f"Error calling Bedrock API or parsing response: {e}")
        return {"food_items": [], "microbes": [], "metabolites": []}

def generate_status_message(stage: str, data: Dict[str, Any]) -> str:
    """
    Generates a custom status update using Bedrock.

    Args:
        stage (str): The step name (e.g., 'llm_extraction', 'processing_metabolite', etc.)
        data (dict): Additional context for generating the message.

    Returns:
        str: Custom status message.
    """
    prompt_templates = {
        "llm_extraction": """
        The AI is analyzing the input to extract key food items, microbes, and metabolites.
        The identified metabolites are: {metabolites}.
        
        These metabolites will be validated against known biochemical pathways to assess 
        their impact. Provide an engaging update summarizing this process. This should be a fun compact one liner.
        """,
        "simulation_start": """
        We are about to launch the simulation! The pipeline will now:
        
        1️⃣ **Check if the metabolites can cross the blood-brain barrier (BBB)**  
        2️⃣ **Simulate their movement between blood, BBB interface, and brain tissue**  
        3️⃣ **Analyze their accumulation and interference effects on brain regions**  

        Generate an engaging status update that excites the user about the simulation. This should be a fun compact one liner.
        """,
        "processing_metabolite": """
        Currently processing **{name}** ({index}/{total}).

        1️⃣ **BBB Transport Model** - Determines if {name} can enter the brain by:
           - Predicting **crossing probability**, **transport rate**, and **destination (blood, BBB, or brain tissue)**.

        2️⃣ **Transport State Model** - Simulates the metabolite’s journey:
           - Tracks **how it moves over time** and predicts **steady-state distribution**.

        3️⃣ **Brain Effect Model** - Assesses the impact of {name} on the brain:
           - Predicts **concentration across 13 brain regions**, **accumulation rates**, and **potential interference patterns**.

        Provide an engaging user-friendly message explaining what’s happening in this step. This should be a fun compact one liner.
        """,
        "prediction_complete": """
        ✅ **Analysis of {name} is complete!**  

        - **BBB Transport Model Results**:  
          🔹 **Crossing Probability**: {prediction[bbb_prediction][crossing_probability]}  
          🔹 **Transport Rate**: {prediction[bbb_prediction][transport_rate]}  
          🔹 **Final Location**: {prediction[bbb_prediction][state_distribution]}  

        - **Transport State Model Predictions**:  
          🔄 **Time-Series Movement**: {prediction[transport_state_prediction][time_series]}  
          📈 **Transition Rates**: {prediction[transport_state_prediction][transition_rates]}  
          🏁 **Steady-State Location**: {prediction[transport_state_prediction][steady_state]}  

        - **Brain Effect Model Predictions**:  
          🧠 **Concentration Across Brain Regions**: {prediction[brain_effect_prediction][region_concentration]}  
          ⚡ **Interference with Brain Activity**: {prediction[brain_effect_prediction][interference_patterns]}  

        Generate a user-friendly message informing the user about the status. This should be a fun compact one liner.
        """,
        "aggregation_complete": """
        🧪 **Finalizing results for all metabolites!**  

        We are now analyzing how **different metabolites interact**, combining data to detect:
        - **Cumulative effects of multiple metabolites**
        - **How they may influence brain chemistry together**
        - **Potential synergies or conflicts in metabolic pathways**

        Generate an engaging message informing the user of this step. This should be a fun compact one liner.
        """,
        "chart_generation": """
        📊 **Generating interactive charts!**  

        The data is now being visualized in:
        - **Graphical representations of BBB transport rates**  
        - **Heatmaps of metabolite distributions across brain regions**  
        - **Line charts showing transport changes over time**  

        Generate an exciting message for the user about the upcoming visualizations. This should be a fun compact one liner.
        """,
        "final_summary": """
        🏁 **Finalizing the AI-generated report!**  

        This report will include:
        - **A deep dive into metabolite transport behavior**
        - **Predicted effects on key brain regions**
        - **Health insights based on these findings**  

        Provide an engaging message to let the user know their results are almost ready. This should be a fun compact one liner.
        """
    }

    system_prompt = [{"text": prompt_templates.get(stage, "Generate a relevant status message based on this data: {data}").format(**data)}]
    messages = [{"role": "user", "content": [{"text": "Generate the status message"}]}]
    inference_config = {"maxTokens": 500, "temperature": 0.7, "topP": 0.9, "topK": 20}

    payload = {
        "schemaVersion": "messages-v1",
        "system": system_prompt,
        "messages": messages,
        "inferenceConfig": inference_config
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )
        content = json.loads(response["body"].read().decode("utf-8"))["output"]["message"]["content"][0]["text"]
        return content.strip()

    except Exception as e:
        logger.error(f"Error generating status message: {e}")
        return f"Error generating status message: {str(e)}"

def generate_final_summary(user_input: str, extracted_data: Dict[str, Any] | List, pipeline_results: str) -> Generator[str, None, None]:
    """
    Generates a streaming final summary using Bedrock.

    Args:
        user_input (str): The original user-provided text.
        extracted_data (dict | list): Contains food items, detected microbes, and metabolites.
        pipeline_results (str): The processed and formatted simulation results.

    Returns:
        Generator[str, None, None]: Streams the final summary text.
    """
    logger.info(f"DEBUG - Input Parameters: user_input={repr(user_input)}, extracted_data={repr(extracted_data)}, pipeline_results={repr(pipeline_results)}")

    # Ensure extracted_data is a dictionary
    if isinstance(extracted_data, list):
        logger.info("DEBUG - extracted_data is a list, converting to proper structure...")
        converted_data = {
            "food_items": [],  # Will be populated by LLM
            "microbes": [],    # Will be populated by LLM
            "metabolites": extracted_data
        }
        extracted_data = converted_data
        logger.info(f"DEBUG - Converted to dict structure: {extracted_data}")

    food_items = extracted_data.get("food_items", [])
    microbes = extracted_data.get("microbes", [])
    metabolites = extracted_data.get("metabolites", [])

    logger.info(f"DEBUG - Extracted Details: food_items={food_items}, microbes={microbes}, metabolites={metabolites}")

    system_prompt = [{
        "text": f"""
        You are a friendly but highly knowledgeable assistant in biochemical analysis. 
        Your task is to generate a **clear and engaging summary** of our metabolite analysis.

        ---
        ## 🌟 **How We Processed Your Input**
        The user originally listed these **food items** in their input:
        🍽️ {', '.join(food_items) if food_items else "None provided"}.

        Based on this, we identified **these microbes** that interact with the food:
        🦠 {', '.join(microbes) if microbes else "None detected"}.

        These microbes are known to produce the following **key metabolites**:
        🧪 {', '.join([m['name'] for m in metabolites]) if metabolites else "None detected"}.

        ---
        ## 🔬 **How Our Simulation Works**
        After identifying the metabolites, we ran a **3-level biochemical simulation** to understand their effects:

        1️⃣ **Blood-Brain Barrier (BBB) Simulation**  
           - Determines if each metabolite can cross the **blood-brain barrier** and enter the brain.

        2️⃣ **Transport & Distribution Simulation**  
           - Analyzes **how the metabolites move through the brain** and which regions they interact with.

        3️⃣ **Brain Impact Analysis**  
           - Estimates the **effects of these metabolites** on different brain functions (e.g., mood, focus, energy).

        ---
        ## 📊 **What We Found**
        {pipeline_results}

        ---
        ## 📝 **Instructions for the Summary**
        - **Explain the findings in simple, engaging terms.**
        - **Make it easy to understand, avoiding excessive scientific jargon.**
        - **Use a friendly and positive tone, like explaining to a curious friend.**
        - **DO NOT ask any follow-up questions.** This is the final response.

        Now, generate a **user-friendly, informative summary** based on these results!
        """
    }]

    messages = [{"role": "user", "content": [{"text": "Generate the summary"}]}]
    inference_config = {"maxTokens": 1000, "temperature": 0.7, "topP": 0.9, "topK": 20}

    payload = {
        "schemaVersion": "messages-v1",
        "system": system_prompt,
        "messages": messages,
        "inferenceConfig": inference_config
    }

    try:
        response = client.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        chunk_count = 0
        logger.info("DEBUG - Starting to process response chunks...")
        for event in response["body"]:
            if "chunk" in event and "bytes" in event["chunk"]:
                chunk_count += 1
                chunk_data = json.loads(event["chunk"]["bytes"].decode("utf-8"))
                content_block_delta = chunk_data.get("contentBlockDelta")
                if content_block_delta and "delta" in content_block_delta and "text" in content_block_delta["delta"]:
                    logger.info(f"DEBUG - Yielding chunk {chunk_count}: {content_block_delta['delta']['text']}")
                    yield content_block_delta["delta"]["text"]
        logger.info(f"DEBUG - Finished processing chunks. Total chunks: {chunk_count}")

    except Exception as e:
        logger.error(f"DEBUG - Error during API call or chunk processing: {str(e)}")
        yield f"Error: {str(e)}"

if __name__ == "__main__":
    # Test the functions
    user_prompt = "I had coffee bagels this morning."

    # Test non-streaming mode
    print("Testing non-streaming function get_llm_output()...\n")
    metabolites_result = get_llm_output(user_prompt, stream=False)
    print(f"\nParsed Metabolites:\n{json.dumps(metabolites_result, indent=2)}\n")

    # Test streaming mode
    print("\nTesting streaming mode...\n")
    response_text = ""
    for result in get_llm_output(user_prompt, stream=True):
        print(result, end="", flush=True)
        response_text += result
    print(f"\nRaw Streaming Response:\n{response_text}\n")