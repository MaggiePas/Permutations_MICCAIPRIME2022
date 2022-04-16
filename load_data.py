from dataframe_utils import *


def load_longitudinal_tabular_data(input_path, write_path, quick=True, write_csv=True):

    # We can save a 'processed' version of the input data to load them quickly each time
    if quick:
        input_feats = read_csv(input_path, separator=',')
    else:
        input_data = read_csv(input_path, separator=',')

        input_feats = tabular_data_processing(input_data)

        if write_csv:
            input_feats.to_csv(os.path.join(write_path, 'processed.csv'), index=False)

    return input_feats


def tabular_data_processing(input_data):

    # Do not use confounders as input variables to the model
    input_data = drop_confounders(input_data, age=False)

    # Some specific processing required for the stroop input variables
    # add the features and then drop the columns we summed
    input_data = sum_stroop(input_data)

    # drop the features we don't need
    input_data = drop_features(input_data, age=False)

    # Rename the new constructs that we will predict to make it easier to remember which is which
    input_data = rename_constructs(input_data)

    # Fill out missing values per subject with nearest visit values
    input_feats = fill_out_missing_values_nearest(input_data)

    # For some subjects we only have one visit therefore we still need the columnn mean/mode
    # to fill out the missing values
    input_feats = fill_out_missing_values(input_feats)

    # Turn variables such as sex from 'F' and 'M' to 0 and 1
    input_feats = text_to_codes(input_feats)

    # An idea was to turn the variables with hours into codes
    # but the results were not improved by that
    # input_feats = hours_to_codes(input_feats)

    # I don't use one-hot encoding anymore since the results were slightly better without it
    # input_feats = categorical_to_onehot(input_feats)

    # We only want to consider visits below 18 yo
    input_feats = remove_18(input_feats)

    return input_feats


# Find common subjects among imaging and tabular input variables longitudinal
def common_subjects_longitudinal(avg_visit_data, avg_visit_imaging_scores):

    common = set(avg_visit_imaging_scores['subject']).intersection(set(avg_visit_data['subject']))

    common_imaging_scores = avg_visit_imaging_scores.loc[avg_visit_imaging_scores['subject'].isin(common)]

    common_scores = pd.merge(avg_visit_data, common_imaging_scores, on=['subject', 'visit'])

    return common_scores


def load_longitudinal_with_imaging(input_path, input_imaging_path_fa, input_imaging_path_t1,write_path, input_imaging_path_rsf=None, quick=True, write_csv=False):

    # Load imaging scores
    imaging_scores_fa = read_csv(input_imaging_path_fa, separator=',')

    imaging_scores_t1 = read_csv(input_imaging_path_t1, separator=',')

    imaging_scores = pd.merge(imaging_scores_fa, imaging_scores_t1, on=['subject', 'visit'])

    if input_imaging_path_rsf != None:
        imaging_scores_rsf = read_csv(input_imaging_path_rsf, separator=',')
        imaging_scores = pd.merge(imaging_scores, imaging_scores_rsf, on=['subject', 'visit'])

    # We can save a 'processed' version of the input data to load them quickly each time
    if quick:
        input_feats = read_csv(input_path, separator=',')
    else:
        input_data = read_csv(input_path, separator=',')

        input_feats = tabular_data_processing(input_data)

    # Merge the two sets of information to the overall data
    input_feats = common_subjects_longitudinal(input_feats, imaging_scores)

    if write_csv:
        input_feats.to_csv(os.path.join(write_path, 'processed_with_imaging.csv'), index=False)

    return input_feats


