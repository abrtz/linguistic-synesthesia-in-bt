import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa
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

    # Rename the 'index' column to 'instance'
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
        elif '2024' in old_column:
            new_columns[old_column] = f"{prefix}{i}"
        else:
            new_columns[old_column] = old_column
    df.rename(columns=new_columns, inplace=True)

def prepare_df(df):
    """
    Prepare the DataFrame by transposing it, renaming columns with a prefix, and dropping NaN values.
    Return the prepared DataFrame.
    
    Parameters:
    - df (pandas.DataFrame): The DataFrame to be prepared with instances as columns and annotations as rows.
    """
    
    df = transpose_dataframe(df)
    rename_columns_with_prefix(df)
    df = df.dropna().reset_index(drop=True)
    return df

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


def compute_cohen_kappa_all_combinations(list_of_lists, column_names):
    """
    Compute Cohen's kappa score for all combinations of lists.
    Return a list of tuples containing the names of the annotations being compared and their corresponding Cohen's kappa scores.

    This function computes the Cohen's kappa score for all combinations of lists
    in the input list_of_lists. Each list is compared with every other list except itself.

    Parameters:
    - list_of_lists (list): A list of lists containing the annotations to be compared.
    - column_names (list): A list of column names corresponding to the annotations.
    """
    results = []
    for (ann1_name, ann1_list), (ann2_name, ann2_list) in combinations(zip(column_names, list_of_lists), 2):
        kappa = cohen_kappa_pairs([ann1_list, ann2_list])
        results.append((ann1_name, ann2_name, round(kappa, 2)))
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

def cohen_kappa_pairs(lists_annotations):
    """Computes Cohen kappa for pair-wise annotators.
    Return Cohen's Kappa statistic.
    
    Parameters:
    - list_of_lists (list): a list of lists containing the annotations of multiple annotators.
    """

    ann1 = lists_annotations[0]
    ann2 = lists_annotations[1]
    
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = (cnt1 / len(ann1)) * (cnt2 / len(ann2))
        E += count

    # Handle the case where E is 1 to avoid division by zero
    if E == 1:
        return 0

    return round((A - E) / (1 - E), 2)

def df_to_matrix(df):
    """
    Convert a DataFrame to a Numpy matrix.
    Return a Numpy matrix containing the counts of annotators for each instance and label.
    
    Parameters:
    - df: Pandas DataFrame with instance as index and annotator as columns.
    """

    # Reshape the DataFrame
    df_fleiss = df.melt(id_vars='instance', var_name='annotator', value_name='label')
    
    # Pivot the DataFrame to get counts
    df_fleiss = df_fleiss.pivot_table(index='instance', columns='label', values='annotator', aggfunc='count', fill_value=0)
    df_fleiss = df_fleiss.rename_axis(columns=None, index=None)
    
    matrix = df_fleiss.values  # convert Pandas DataFrame to Numpy matrix

    return matrix

def calculate_fleiss_kappa(df):
    """
    Calculate Fleiss' Kappa score from a DataFrame.
    Return Fleiss' Kappa score.

    Parameters:
    - df: Pandas DataFrame with instance as rows and annotators as columns.
    """
    
    # Convert DataFrame to matrix
    matrix = df_to_matrix(df)

    # Calculate Fleiss' Kappa
    fleiss_kappa_score = fleiss_kappa(matrix)

    return round(fleiss_kappa_score, 2)

def observed_agreement(ann1, ann2):
    """Computes observed agreement for pair-wise annotators.

    :param ann1: annotations provided by first annotator
    :type ann1: list
    :param ann2: annotations provided by second annotator
    :type ann2: list

    :rtype: float
    :return: Cohen kappa statistic
    """
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)

    return A

def compute_observed_agreement_combinations(list_of_lists,column_names):
    """
    Compute the observed agreement for all combinations of lists of annotations.
    Return a list of tuples containing the names of the annotations being compared and their corresponding observed agreement scores.

    This function computes the observed agreement score for all combinations of lists
    in the input list_of_lists. Each list is compared with every other list except itself.
    
    Parameters:
    - list_of_lists (list): A list of lists containing the annotations to be compared.
    - column_names (list): A list of column names corresponding to the annotations.
    """
    results = []
    
    agreement = 0
    count = 0

    for (ann1_name, ann1_list), (ann2_name, ann2_list) in combinations(zip(column_names, list_of_lists), 2):
        oa = observed_agreement(ann1_list, ann2_list)
        results.append((ann1_name, ann2_name, round(oa, 2)))

    return results

def compute_oa(list_of_lists):
    """
    Compute the inter-annotator agreement using observed agreement for multiple annotators.
    Each annotator is compared with every other annotator except itself.
    Return the average observed agreement score across all annotator pairs.

    Parameters:
    - list_of_lists (list): a list of lists containing the annotations of multiple annotators.
    """
    
    total_oa = 0
    num_pairs = 0

    for i, ann1 in enumerate(list_of_lists):
        for j, ann2 in enumerate(list_of_lists):
            if i != j:  # Avoid comparing a list with itself
                oa = observed_agreement(ann1, ann2)
                total_oa += oa
                num_pairs += 1
    
    if num_pairs == 0:
        return 0  # No pairs to compare

    average_oa = total_oa / num_pairs
    
    return round(average_oa, 2)

def analyze_corner_cases(df, corner_cases,columns,author=None, iaa='cohens'):
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
    - iaa (str, optional): the method to compute IAA. 'cohens' for Cohen's kappa, 'fleiss' for Fleiss Kappa, and 'oa' for observed agreement. Defaults to 'cohens'.
    """
    
    # Extracting corner case annotations
    corner_case_annotations = [row for _, row in df.iterrows() if any(item in row.values for item in corner_cases)]
    corner_case_annotations = pd.DataFrame(corner_case_annotations, columns=df.columns).reset_index(drop=True)

    if iaa == 'cohens':
        # Extracting lists from columns
        corner_list_of_annotations = get_lists_from_columns(corner_case_annotations, columns)
    
        # Computing Cohen's kappa for corner cases
        cohen_kappa_results = compute_cohen_kappa_all_combinations(corner_list_of_annotations, columns)
    
        # Computing Inter-Annotator Agreement (IAA) for corner cases
        iaa_cohen_score = compute_iaa_cohens_kappa(corner_list_of_annotations)
    
        # Displaying the results
        for ann1, ann2, kappa in cohen_kappa_results:
            print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")
    
        print()
        
        if author != None:
            print(f"{author}'s corner cases Inter-Annotator Agreement (Cohen's kappa):", iaa_cohen_score)
    
        else:
            print(f"All instances' corner cases Inter-Annotator Agreement (Cohen's kappa):", iaa_cohen_score)

    elif iaa == 'fleiss':
        iaa_fleiss_score = calculate_fleiss_kappa(corner_case_annotations)
        
        if author != None:
            print(f"{author}'s corner cases Inter-Annotator Agreement (Fleiss' kappa):", iaa_fleiss_score)
    
        else:
            print(f"All instances' corner cases Inter-Annotator Agreement (Fleiss' kappa):", iaa_fleiss_score)

    elif iaa == 'oa':
        observed_agreement_results = compute_observed_agreement_combinations(corner_case_annotations, columns)

        iaa_observed_agreement = compute_oa(corner_list_of_annotations)

        # Displaying the results
        for ann1, ann2, oa in observed_agreement_results:
            print(f"Observed agreement between {ann1} and {ann2}: {oa}")

        if author != None:
            print(f"{author}'s corner cases Inter-Annotator Agreement (Observed Agreement):", iaa_observed_agreement)
    
        else:
            print(f"All instances' corner cases Inter-Annotator Agreement (Observed Agreement):", iaa_observed_agreement)


def analyze_regular_cases(df, reg_cases,columns,author=None, iaa='cohens'):
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
    - iaa (str, optional): the method to compute IAA. 'cohens' for Cohen's kappa, 'fleiss' for Fleiss Kappa, and 'oa' for observed agreement. Defaults to 'cohens'.
    """
    
    # Extracting regular case annotations
    reg_case_annotations = [row for _, row in df.iterrows() if not any(item in row.values for item in reg_cases)]
    reg_case_annotations = pd.DataFrame(reg_case_annotations, columns=df.columns).reset_index(drop=True)

    if iaa == 'cohens':
        # Extracting lists from columns
        corner_list_of_annotations = get_lists_from_columns(reg_case_annotations, columns)
    
        # Computing Cohen's kappa for regular cases
        cohen_kappa_results = compute_cohen_kappa_all_combinations(corner_list_of_annotations, columns)
    
        # Computing Inter-Annotator Agreement (IAA) for regualr cases
        iaa_cohen_score = compute_iaa_cohens_kappa(corner_list_of_annotations)
    
        # Displaying the results
        for ann1, ann2, kappa in cohen_kappa_results:
            print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")
    
        print()
        
        if author != None:
            print(f"{author}'s regular cases Inter-Annotator Agreement (Cohen's kappa):", iaa_cohen_score)
    
        else:
            print(f"All instances' regular cases Inter-Annotator Agreement (Cohen's kappa):", iaa_cohen_score)
        
    elif iaa == 'fleiss':
        iaa_fleiss_score = calculate_fleiss_kappa(reg_case_annotations)

        if author != None:
            print(f"{author}'s regular cases Inter-Annotator Agreement (Fleiss' kappa):", iaa_fleiss_score)
    
        else:
            print(f"All instances' regular cases Inter-Annotator Agreement (Fleiss' kappa):", iaa_fleiss_score)

    elif iaa == 'oa':
        observed_agreement_results = compute_observed_agreement_combinations(reg_case_annotations, columns)
        iaa_observed_agreement = compute_oa(reg_case_annotations)

        # Displaying the results
        for ann1, ann2, oa in observed_agreement_results:
            print(f"Observed agreement between {ann1} and {ann2}: {oa}")

        if author != None:
            print(f"{author}'s regular cases Inter-Annotator Agreement (Observed Agreement):", iaa_observed_agreement)
    
        else:
            print(f"All instances' regular cases Inter-Annotator Agreement (Observed Agreement):", iaa_observed_agreement)


def analyze_cases(df, corner_cases, columns, author=None, iaa='cohens'):
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
    - iaa (str, optional): The method to compute IAA. 'cohens' for Cohen's kappa, 'fleiss' for Fleiss Kappa, and 'oa' for observed agreement. Defaults to 'cohens'.
    """
    
    if author != None:
        print(f"{author} - Corner Cases:")
        print()
        # Analyze corner cases
        analyze_corner_cases(df, corner_cases, columns, author, iaa)
        print('----------------------------------------------')
        print(f"{author} - Regular Cases:")
        print()
        # Analyze regular cases
        analyze_regular_cases(df, corner_cases, columns, author, iaa)
        print('----------------------------------------------')
    else:
        print(f"All instances - Corner Cases:")
        print()
        # Analyze corner cases
        analyze_corner_cases(df, corner_cases, columns, author, iaa)
        print('----------------------------------------------')
        print(f"All instances - Regular Cases:")
        print()
        # Analyze regular cases
        analyze_regular_cases(df, corner_cases, columns, author, iaa)
        print('----------------------------------------------')


def analyze_all_cases(df, columns, author=None, iaa='cohens'):
    """
    Analyzes all cases in a DataFrame and computes Inter-Annotator Agreement (IAA) 
    with Cohen's kappa for each pair of annotators and the average Cohen's kappa.
    Prints the id of the annotators compared and their corresponding Cohen's kappa scores 
    and the computed averaged Inter-Annotator Agreement (IAA).
    
    Parameters:
    - df (DataFrame): the DataFrame containing the data.
    - columns (list): a list of column names from which to extract lists.
    - author (str, optional): the author's name. Defaults to None.
    - iaa (str, optional): the method to compute IAA. 'cohens' for Cohen's kappa, 'fleiss' for Fleiss Kappa, and 'oa' for observed agreement. Defaults to 'cohens'.
    """

    if iaa == 'cohens':
        # Extracting lists from columns
        list_of_annotations = get_lists_from_columns(df, columns)
        
        # Computing Cohen's kappa for all cases
        cohen_kappa_results = compute_cohen_kappa_all_combinations(list_of_annotations, columns)
        
        # Computing Inter-Annotator Agreement (IAA) for all cases
        iaa_cohen_score = compute_iaa_cohens_kappa(list_of_annotations)

        # Displaying the results
        if author:
            print(f"{author}'s Cohen's kappa and IAA (all cases):")
        else:
            print("All instances' Cohen's kappa and IAA (all cases):")
        print()
        for ann1, ann2, kappa in cohen_kappa_results:
            print(f"Cohen's kappa between {ann1} and {ann2}: {kappa}")
        print()
        print("Inter-Annotator Agreement (Cohen's kappa) score - all cases:", iaa_cohen_score)
        print('----------------------------------------------')
    
    elif iaa == 'fleiss':
        iaa_fleiss_score = calculate_fleiss_kappa(df)
        print(f"Inter-Annotator Agreement (Fleiss Kappa) score - all cases: {iaa_fleiss_score}")
        print('----------------------------------------------')

    elif iaa == 'oa':
        # Extracting lists from columns
        list_of_annotations = get_lists_from_columns(df, columns)
        
        observed_agreement_results = compute_observed_agreement_combinations(list_of_annotations, columns)
        iaa_observed_agreement = compute_oa(list_of_annotations)

        # Displaying the results
        if author:
            print(f"{author}'s Observed Agreement and IAA (all cases):")
        else:
            print("All instances' Observed Agreement and IAA (all cases):")
        print()
        for ann1, ann2, oa in observed_agreement_results:
            print(f"Observed agreement between {ann1} and {ann2}: {oa}")
        print()
        print("Inter-Annotator Agreement (Observed Agreement) score - all cases:", iaa_observed_agreement)
        print('----------------------------------------------')


def get_final_corpus_and_ties(df, annotators):
    """
    Process annotations from a DataFrame.
    Select the label with the highest count as the final gold label. 
    Ties are handled by disregarding the last annotator's annotation (done by LLM).
    Return a DataFrame containing instances with their corresponding final labels and a list of instances were ties occurred regarding four annotators.
    
    Parameters:
    - df (DataFrame): the DataFrame containing annotations.
    - annotators (list): a list of column names representing annotators.
    """
    
    # Initialize final corpus list and tie instances list
    final_corpus = []
    tie_instances = []

    # Extracting annotations into one column
    df_annotations = pd.melt(df, id_vars=['instance'], value_vars=annotators, value_name='annotation')
    df_annotations['annotation'] = df_annotations['annotation'].str.lower()

    # Group instances and count unique annotations
    instances = df_annotations.groupby('instance')['annotation'].nunique()

    # Add instances to final corpus
    for instance in instances.index:
        instance_data = df_annotations[df_annotations['instance'] == instance]['annotation']
        label_counts = instance_data.value_counts().to_dict()
        max_count = max(label_counts.values())
        if max_count == 2:  #if there is a tie, disregard the last annotator and calculate max label from the first three human annotators
            instance_data = instance_data.iloc[:-1]  # Exclude the last annotation done by LLM
            label_counts = instance_data.value_counts().to_dict()
            max_label = max(label_counts, key=label_counts.get)
            tie_instances.append(instance)  # Store instances with ties
        else:
            max_label = max(label_counts, key=label_counts.get)  # Only one label with maximum count
        entry = {'instance': instance, 'label': max_label}
        final_corpus.append(entry)

    # Convert the final list of dictionaries to a DataFrame
    df_final_corpus = pd.DataFrame(final_corpus)

    return df_final_corpus, tie_instances


def process_annotations_and_disagreements(df, annotators):
    """
    Process annotations from a DataFrame.
    Keep the instances in which there is common agreement among all annotators.
    If there is no common agreement, keep the instances in which all human annotators agreed.

    Return a DataFrame containing instances with their corresponding final labels and a list of instances where there was disagreement from LLM.
    
    Parameters:
    - df (DataFrame): The DataFrame containing annotations.
    - annotators (list): a list of column names representing annotators.
    
    Returns:
    - df_final_corpus (DataFrame): A DataFrame containing instances with their corresponding final labels.
    - instances_with_llm_disagreement (list): A list of instances where there was disagreement from LLM.
    """
    
    # Initialize final corpus list and list for instances with LLM disagreement
    final_corpus = []
    instances_with_llm_disagreement = []

    # Extract annotations into one column
    df_annotations = pd.melt(df, id_vars=['instance'], value_vars=annotators, value_name='annotation')
    df_annotations['annotation'] = df_annotations['annotation'].str.lower()

    # Group instances and count unique annotations
    instances = df_annotations.groupby('instance')['annotation'].nunique()

    # Add instances to final corpus
    for instance in instances.index:
        instance_data = df_annotations[df_annotations['instance'] == instance]['annotation']
        label_counts = instance_data.value_counts().to_dict()
        max_count = max(label_counts.values())

        if max_count == 4:  # All labels are the same
            label = list(label_counts.keys())[0]  # Get the key from label_counts
            entry = {'instance': instance, 'label': label}
            final_corpus.append(entry)

        elif max_count == 3:
            # Check if the annotations from the first three annotators are the same
            human_annotators = instance_data.iloc[:3]
            if len(human_annotators.unique()) == 1:
                label = human_annotators.iloc[0]  # Use the label from the first three annotators
                entry = {'instance': instance, 'label': label}
                final_corpus.append(entry)
                instances_with_llm_disagreement.append(instance)  # Store instances with disagreement from LLM

    # Convert the final list of dictionaries to a DataFrame
    df_final_corpus = pd.DataFrame(final_corpus)

    return df_final_corpus, instances_with_llm_disagreement


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


