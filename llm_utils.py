from pydantic import BaseModel
from typing_extensions import Literal
from openai import OpenAI
import polars as pl
import instructor

# Set your key and a model from https://platform.openai.com/docs/models/continuous-model-upgrades
chosen_model = "gpt-4"
key = "" #insert key
client = OpenAI(api_key=key)
client = instructor.patch(OpenAI(api_key=key), mode=instructor.Mode.MD_JSON)

def prepare_df(df):
    """Transpose the columns with rows and rows with columns in a pandas DataFrame.
    Rename the first column name with 'instance' and number the annotators for each column.
    Remove the rows that have empty values.
    Return the clean DataFrame.

    Parameters:
    -df: pandas DataFrame containing instances as columns and annotations as rows.
    """
    
    df = transpose_dataframe(df)
    rename_columns_with_prefix(df)
    df = df.dropna().reset_index(drop = True)
    return df

# Here's how we interact with the model
def do_annotation(chosen_prompt):
    resp = client.chat.completions.create(
        model=chosen_model, 
        messages=[
            {"role": "user", 
            "content": chosen_prompt},
        ],
        temperature=1,
        )
    annotation = resp.choices[0].message.content
    return annotation


# Use pydantic to define the output schema
class Option(BaseModel):
    """
    Either yes or no.
    """
    option: Literal["yes", "no"]


def structured_annotation_output(chosen_prompt):
    resp = client.chat.completions.create(
        model=chosen_model, 
        messages=[
            {"role": "user", 
            "content": chosen_prompt},
        ],
        temperature=1,
        response_model=Option,
        )
    return resp


def annotate_synesthesia(data, my_instructions):
    """
    Annotate instances to identify linguistic synesthesia.
    Return a python dictionary containing the annotations for each instance.

    Parameters:
    - data (list): a list of instances to annotate.
    - my_instructions (str): instructions for the annotation task.
    """
    
    annotation_dict = {}

    for index in range(data.height):
        row = data[index]["instance"][0]  # take the text to annotate
        prompt = f"{my_instructions} {row}"

        annotation = structured_annotation_output(prompt)
        print(f" --- I HAVE ANNOTATED TEXT {index} ---")
        print(prompt)
        print(annotation)
        print("\n")

        # store the annotations in the dictionary
        annotation_dict[index] = annotation.option

    return annotation_dict

def do_translation(data, my_instructions, column_name="instance"):
    """
    Translate text in the 'instance' column of the DataFrame using the specified instructions.
    Return a dictionary containing the translated text for each row in the DataFrame. The keys are the row indices, and the values are the translated texts.

    Parameters:
    - data (DataFrame): A polars DataFrame containing the text to be translated.
    - my_instructions (str): Instructions for the LLM model to carry out the translation.
    - column_name (str): A string indicating the name of the column to retrieve the instances to translate. Default set to "instance"
    """

    translation_dict = {}

    for index in range(data.height):
        row = data[index][column_name][0]  # take the text to translate
        prompt = f"{my_instructions} {row}"
        
        resp = client.chat.completions.create(
        model=chosen_model, 
        messages=[
            {"role": "user", 
            "content": prompt},
        ],
        temperature=1,
        )
        translation = resp.choices[0].message.content
        print(f" --- I HAVE TRANSLATED TEXT {index} ---")
        print(prompt)
        print(translation)
        print("\n")

        # store the translations in the dictionary
        translation_dict[index] = translation
        
    return translation_dict

