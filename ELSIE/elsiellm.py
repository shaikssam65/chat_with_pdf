import pandas as pd
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

class CTransformers:
    def __init__(self, model, model_type, max_new_tokens, temperature):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def generate(self, prompt):
        # Tokenize the input prompt to format suitable for the model
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

        # Generate tokens from the model based on the input ids
        # Here, adjust max_length according to the number of tokens you want to generate
        outputs = self.model.generate(input_ids, max_length=self.max_new_tokens + len(input_ids[0]),
                                      temperature=self.temperature, num_return_sequences=1)

        # Decode the generated tokens to a readable string
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated_text

def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def extract_polymer_info(llm, text):
    prompt = f"""
    ### Task Description:
    Extract information about polymers and their melting points from the provided text.

    ### Provided Text:
    {text}

    ### Instructions:
    1. Identify any polymer names mentioned.
    2. Identify melting points associated with these polymers.

    ### Output Requirements:
    - Polymer and melting point found: Format as "Polymer: [Name], Melting Point: [Temperature]°C"
    - Only polymer name found: "Found Polymer Name: [Name]"
    - Only melting point found: "Found Melting Point: [Temperature]°C"
    - Neither found: "Did not find any information"
    """
    response = llm.generate(prompt)  # Call the model's generate method with the prepared prompt
    return response

def process_csv(input_file, output_file, llm):
    df = pd.read_csv(input_file)
    output_column_name = "LLM_Output"

    # Ensure output column does not already exist to avoid overwriting data
    if output_column_name in df.columns:
        output_column_name += "_updated"

    df[output_column_name] = df.apply(lambda row: extract_polymer_info(llm, row['combined_text']), axis=1)
    df.to_csv(output_file, index=False)

# Main execution part
if __name__ == "__main__":
    llm = load_llm()  # Load the LLM
    input_csv_path = "BLOB_CDE_LABELD_PLYMER_NER_F_poly_PROP.csv"  # Filename of the input CSV file in the current directory
    output_csv_path = "output_LLAMA_ELSIE.csv"  # Filename for the output CSV in the current directory
    process_csv(input_csv_path, output_csv_path, llm)  # Process the CSV file
