import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from itertools import combinations
import os

def transpose_dataframe(df):
    # Transpose DataFrame
    df_transposed = df.T
    
    # Set the first row as the column headers
    df_transposed.columns = df_transposed.iloc[0]

    # Drop the first row (which is now the column names)
    df_transposed = df_transposed.drop(df_transposed.index[0])

    # Reset index
    df_transposed = df_transposed.reset_index()

    # Rename the 'index' column to 'Attributes'
    df_transposed = df_transposed.rename(columns={'index': 'instance'})
    
    return df_transposed


def count_annotated_labels(df, annotators=['annotator1', 'annotator2', 'annotator3', 'annotator4', 'annotator5', 'annotator6']):
    """
    Count cases of disagreement between annotators.

    Parameters:
    - df (DataFrame): DataFrame containing annotations.
    - annotators (list): list of column names containing annotations from different annotators.

    Return:
    - pandas Series: Series containing the number of cases of disagreement, categorized by the number of annotators disagreeing.
    """
    
    # Combine annotations into one column
    df_annotations = pd.melt(df, id_vars=['instance'], value_vars=annotators, value_name='annotation')

    # Find cases of disagreement
    annotated_labels = df_annotations.groupby('instance')['annotation'].nunique()
    
    # Count the number of cases where 1, 2, or all annotators disagree
    annotated_labels_counts = annotated_labels.value_counts()

    return annotated_labels_counts


def rename_columns_with_prefix(df, prefix='annotator'):
    """
    Rename columns in a DataFrame with a specified prefix followed by a number, 
    skipping the first column which is renamed as 'instance'.
    Return None: modify the input DataFrame in place.

    Parameters:
    - df (pandas.DataFrame): the DataFrame whose columns are to be renamed.
    - prefix (str, optional arg): the prefix to be added to each column name. Default is 'annotator'.
    """
    
    new_columns = {}
    for i, old_column in enumerate(df.columns):
        if i == 0:
            new_columns[old_column] = "instance"
        else:
            new_columns[old_column] = f"{prefix}{i}"
    df.rename(columns=new_columns, inplace=True)


def count_disagreements_in_corner_cases(df, instances):
    disagreement_counts = []

    for instance in instances:
        if instance in df.values:
            annotations = df[df.eq(instance).any(axis=1)].iloc[:, 1:].values.flatten()
            unique_annotations = len(set(annotations))
            disagreement_counts.append(unique_annotations)
        else:
            print(instance)

    return pd.Series(disagreement_counts).value_counts()


def get_lists_from_columns(df, columns):
    """
    Extract lists from specified columns of a DataFrame and convert the values to lowercase.
    Returns a list of lists, where each inner list corresponds to the values of a column, converted to lowercase.
    
    Parameters:
    - df (pandas.DataFrame): the DataFrame containing the columns.
    - columns (list): a list of column names from which to extract the lists.
    """
    
    lists = []
    for column in columns:
        lists.append(df[column].str.lower().tolist())
    return lists


def compute_cohen_kappa_all_combinations(list_of_lists):
    """
    Compute Cohen's kappa score for all combinations of lists.
    Return a list of tuples containing the names of the annotations being compared and their corresponding Cohen's kappa scores.

    This function computes the Cohen's kappa score for all combinations of lists
    in the input list_of_lists. Each list is compared with every other list except itself.

    Parameters:
    - list_of_lists (list): a list of lists containing the annotations to be compared.
    """
    results = []
    for ann1, ann2 in combinations(enumerate(list_of_lists), 2):  # Iterate over all combinations of lists
        i, ann1_list = ann1
        j, ann2_list = ann2
        kappa = cohen_kappa_score(ann1_list, ann2_list)
        results.append((f"ann{i+1}", f"ann{j+1}", round(kappa, 2)))
    return results


def compute_iaa_cohens_kappa(list_of_lists):
    """
    Compute the inter-annotator agreement using Cohen's kappa for multiple annotators.
    Each annotator is compared with every other annotator except itself.
    Return the average Cohen's kappa score across all annotator pairs.

    Parameters:
    - list_of_lists (list): a list of lists containing the annotations of multiple annotators.
    """
    
    total_kappa = 0
    num_pairs = 0

    for i, ann1 in enumerate(list_of_lists):
        for j, ann2 in enumerate(list_of_lists):
            if i != j:  # Avoid comparing a list with itself
                kappa = cohen_kappa_score(ann1, ann2)
                total_kappa += kappa
                num_pairs += 1

    if num_pairs == 0:
        return 0  # No pairs to compare

    average_kappa = total_kappa / num_pairs
    
    return round(average_kappa, 2)


def analyze_corner_cases(df, corner_cases,columns,author=None):
    """
    Analyzes corner cases in a DataFrame and computes Inter-Annotator Agreement (IAA) 
    with Cohen's kappa for each pair of annotators and the average Cohen's kappa.
    Print the names of the annotators compared and their corresponding Cohen's kappa scores 
    and the computed averaged Inter-Annotator Agreement (IAA).
    
    Parameters:
    - df (DataFrame): the DataFrame containing the data.
    - corner_cases (list): a list of corner cases.
    - columns (list): A list of column names from which to extract lists.
    - author (str, optional): The author's name. Defaults to None prints all instances.
    """
    
    # Extracting corner case annotations
    corner_case_annotations = [row for _, row in df.iterrows() if any(item in row.values for item in corner_cases)]
    corner_case_annotations = pd.DataFrame(corner_case_annotations, columns=df.columns).reset_index(drop=True)

    # Extracting lists from columns
    corner_list_of_annotations = get_lists_from_columns(corner_case_annotations, columns)

    # Computing Cohen's kappa for corner cases
    kappa_results = compute_cohen_kappa_all_combinations(corner_list_of_annotations)

    # Computing Inter-Annotator Agreement (IAA) for corner cases
    iaa = compute_iaa_cohens_kappa(corner_list_of_annotations)

    # Displaying the results
    for ann1, ann2, kappa in kappa_results:
        print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")

    print()
    
    if author != None:
        print(f"{author}'s corner cases Inter-Annotator Agreement (Cohen's kappa):", iaa)

    else:
        print(f"All instances' corner cases Inter-Annotator Agreement (Cohen's kappa):", iaa)


def analyze_regular_cases(df, reg_cases,columns,author=None):
    """
    Analyzes regular cases in a DataFrame and computes Inter-Annotator Agreement (IAA) 
    with Cohen's kappa for each pair of annotators and the average Cohen's kappa.
    Print the id of the annotators compared and their corresponding Cohen's kappa scores 
    and the computed averaged Inter-Annotator Agreement (IAA).
    
    Parameters:
    - df (DataFrame): the DataFrame containing the data.
    - reg_cases (list): a list of regular cases.
    - columns (list): a list of column names from which to extract lists.
    - author (str, optional): the author's name. Defaults to None prints all instances.
    """
    
    # Extracting regular case annotations
    reg_case_annotations = [row for _, row in df.iterrows() if not any(item in row.values for item in reg_cases)]
    reg_case_annotations = pd.DataFrame(reg_case_annotations, columns=df.columns).reset_index(drop=True)

    # Extracting lists from columns
    corner_list_of_annotations = get_lists_from_columns(reg_case_annotations, columns)

    # Computing Cohen's kappa for regular cases
    kappa_results = compute_cohen_kappa_all_combinations(corner_list_of_annotations)

    # Computing Inter-Annotator Agreement (IAA) for regualr cases
    iaa = compute_iaa_cohens_kappa(corner_list_of_annotations)

    # Displaying the results
    for ann1, ann2, kappa in kappa_results:
        print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")

    print()
    
    if author != None:
        print(f"{author}'s regular cases Inter-Annotator Agreement (Cohen's kappa):", iaa)

    else:
        print(f"All instances' regular cases Inter-Annotator Agreement (Cohen's kappa):", iaa)

def analyze_all_cases(df, columns, author=None):
    """
    Analyzes all cases in a DataFrame and computes Inter-Annotator Agreement (IAA) 
    with Cohen's kappa for each pair of annotators and the average Cohen's kappa.
    Prints the id of the annotators compared and their corresponding Cohen's kappa scores 
    and the computed averaged Inter-Annotator Agreement (IAA).
    
    Parameters:
    - df (DataFrame): the DataFrame containing the data.
    - columns (list): a list of column names from which to extract lists.
    - author (str, optional): the author's name. Defaults to None.
    """
    
    # Extracting lists from columns
    list_of_annotations = get_lists_from_columns(df, columns)
    
    # Computing Cohen's kappa for all cases
    kappa_results = compute_cohen_kappa_all_combinations(list_of_annotations)
    
    # Computing Inter-Annotator Agreement (IAA) for all cases
    iaa = compute_iaa_cohens_kappa(list_of_annotations)
    
    # Displaying the results
    if author:
        print(f"{author}'s Cohen's kappa and IAA (all cases):")
    else:
        print("All instances' Cohen's kappa and IAA (all cases):")
    print()
    for ann1, ann2, kappa in kappa_results:
        print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")
    print()
    print("Inter-Annotator Agreement (Cohen's kappa) - all cases:", iaa)
    print('----------------------------------------------')
    


def analyze_cases(df, corner_cases, columns, author=None):
    """
    Analyzes corner cases and regular cases in a DataFrame and computes Inter-Annotator Agreement (IAA) 
    with Cohen's kappa for each pair of annotators and the average Cohen's kappa.
    Prints the id of the annotators compared and their corresponding Cohen's kappa scores 
    and the computed averaged Inter-Annotator Agreement (IAA).
    
    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - corner_cases (list): A list of corner cases.
    - columns (list): A list of column names from which to extract lists.
    - author (str, optional): The author's name. Defaults to None.
    """
    
    if author != None:
        print(f"{author} - Corner Cases:")
        print()
        # Analyze corner cases
        analyze_corner_cases(df, corner_cases, columns, author)
        print('----------------------------------------------')
        print(f"{author} - Regular Cases:")
        print()
        # Analyze regular cases
        analyze_regular_cases(df, corner_cases, columns, author)
        print('----------------------------------------------')
    else:
        print(f"All instances - Corner Cases:")
        print()
        # Analyze corner cases
        analyze_corner_cases(df, corner_cases, columns, author)
        print('----------------------------------------------')
        print(f"All instances - Regular Cases:")
        print()
        # Analyze regular cases
        analyze_regular_cases(df, corner_cases, columns, author)
        print('----------------------------------------------')


def get_final_corpus(df, annotators):
    """
    Get instances and count the labels provided by the annotators.
    Select the label with the highest count as the final gold label. If there is a tie, the max() function will select the first element it encounters with the highest value.
    Return a DataFrame containing instances with their corresponding final labels.
    
    Parameters:
    - df (DataFrame): The DataFrame containing annotations.
    - annotators (list): A list of column names representing annotators.
    """

    # Initialize final corpus list
    final_corpus = []
    
    # Combine annotations into one column
    df_annotations = pd.melt(df, id_vars=['instance'], value_vars=annotators, value_name='annotation')
    df_annotations['annotation'] = df_annotations['annotation'].str.lower()
    
    instances = df_annotations.groupby('instance')['annotation'].nunique()
    
    # Add instances to final corpus
    for instance in instances.index:
        instance_data = df_annotations[df_annotations['instance'] == instance]['annotation']
        label_counts = instance_data.value_counts().to_dict()
        max_label = max(label_counts, key=label_counts.get)
        entry = {'instance': instance, 'label': max_label}
        final_corpus.append(entry)

    # Convert the final list of dictionaries to a DataFrame
    df_final_corpus = pd.DataFrame(final_corpus)

    return df_final_corpus


def visualize_label_distribution(df):
    """
    Visualizes the distribution of labels in a DataFrame.

    Parameters:
    - df (DataFrame): the DataFrame containing the data.
    """
    # Create figure and axis
    plt.figure(figsize=(5, 5))
    sns.countplot(x='label', data=df)

    # Show the plot
    plt.show()


def visualize_and_write_final_df(dataframes, file_names):
    """
    Visualizes the distribution of labels in each DataFrame in dataframes, 
    saves the DataFrames to CSV files with corresponding file names, 
    and prints the distribution of labels and file writing status.

    Parameters:
    - dataframes (list): a list of DataFrames containing the data to be written.
    - file_names (list): a list of file path names for saving the DataFrames.
    """

    if not os.path.exists('final_corpus'):
        os.mkdir('final_corpus')
    
    # Iterate through the dataframes and file names
    for df, file_name in zip(dataframes, file_names):
        # Get final corpus
        print(f"Distribution of the labels in file {file_name}")
        print(df['label'].value_counts())

        # Visualize label distribution
        visualize_label_distribution(df)
        
        # Write DataFrame to CSV file
        df.to_csv(file_name, index=False)
        
        print(f"Final corpus for {file_name} successfully written to csv file.")
        print('----------------------------------------------')
        print()


