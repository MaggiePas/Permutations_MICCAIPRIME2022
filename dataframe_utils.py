import pandas as pd
import os
import numpy as np


def read_csv(in_path='', separator=';'):

    in_data = pd.read_csv(in_path, sep=separator)

    return in_data


def load_predicted_values(predicted_value='negative_valence', in_data=None):

    predict = in_data[["subject", predicted_value]]

    return predict


def load_input_features(features=[], in_data=None):

    feature = in_data.loc[:, features]

    return feature


def fill_out_missing_values_nearest(in_data=None, write_csv=False, path=None):

    features = list(in_data.columns)
    appended_data = []

    # Select the visits for each subject
    for subject in in_data.subject.unique():
        subj_visits = in_data[in_data['subject'] == subject]
        # For each column fill out the missing values
        for feature in features:
            # If the column has missing values proceed (we avoid running it on string columns)
            if subj_visits[feature].isnull().any():
                # Fist try to fill out the missing values of the baseline visits from the followup ones
                subj_visits[feature] = subj_visits[feature].bfill()
                # Then fill out the followup missing values based on the previous ones
                subj_visits[feature] = subj_visits[feature].interpolate('pad')

        appended_data.append(subj_visits)

    filled = pd.concat(appended_data)

    return filled


def fill_out_missing_values(in_data=None, write_csv=False, path=None):

    # For columns with categorical values replace missing values with mode instead of mean
    dropdown = ['site',	'sex',	'hispanic',	'race',
                'ses_parent_yoe',	'cahalan_score',	'exceeds_bl_drinking_2',
                'lssaga_dsm4_youth_d04_diag',	'lssaga_dsm4_youth_d05_diag',
                'youthreport1_yfhi4',	'youthreport1_yfhi3',	'youthreport1_yfhi5',	'youthreport2_chks_set2_chks3',	'youthreport2_chks_set2_chks4',
                'youthreport2_chks_set4_chks5',	'youthreport2_chks_set4_chks6',	'youthreport2_chks_set5_chks7',
                'youthreport2_chks_set5_chks8',	'youthreport2_chks_set5_chks9',	'youthreport2_pwmkcr_involvement_pwmkcr3',
                'youthreport2_shq1',	'youthreport2_shq2',	'youthreport2_shq3',
                'youthreport2_shq4',	'youthreport2_shq5']

    for column_name in dropdown:
        in_data[column_name] = in_data[column_name].fillna(in_data[column_name].mode()[0])

    # For the rest of the columns that are numerical values replace missing values with column mean
    filled = in_data.where(pd.notna(in_data), in_data.mean().round(0), axis='columns')

    if write_csv:
        filled.to_csv(os.join(path, 'filled_missing_data.csv', index=False))

    return filled


def average_visit_data(in_data=None, write_csv=False, path=None):

    appended_data = []
    boolean_to_predict = ['negative_valence', 'arousal', 'cognitive', 'positive_valence', 'no_construct_at_all']

    for subject in in_data.subject.unique():
        subj_visits = in_data[in_data['subject'] == subject]
        subj_visits_mean = subj_visits.groupby('subject', as_index=False).mean()
        for feat in boolean_to_predict:
            # If one of the visits is True then the 'mean' label is also True
            subj_visits_mean[feat] = subj_visits_mean[feat].apply(np.ceil)
        appended_data.append(subj_visits_mean)

    all_subjects = pd.concat(appended_data)

    if write_csv:
        all_subjects.to_csv(os.join(path, 'average_data.csv', index=False))

    return all_subjects


def average_imaging_visit_data(in_data=None):

    appended_data = []

    for subject in in_data.subject.unique():
        subj_visits = in_data[in_data['subject'] == subject]
        subj_visits_mean = subj_visits.groupby('subject', as_index=False).mean()

        appended_data.append(subj_visits_mean)

    all_subjects = pd.concat(appended_data)

    return all_subjects


def text_to_codes(in_data=None, path=None, write_csv=False):

    # Variables in categorical format that need to be converted
    # categorical = ['site', 'sex', 'cahalan_score', 'scanner', 'scanner_model', 'hispanic']
    categorical = ['site', 'sex', 'cahalan_score', 'hispanic']

    for category in categorical:
        in_data[category] = in_data[category].astype('category')

    cat_columns = in_data.select_dtypes(['category']).columns
    in_data[cat_columns] = in_data[cat_columns].apply(lambda x: x.cat.codes)

    if write_csv:
        in_data.to_csv(os.join(path, 'categorical_data.csv', index = False))
    return in_data


def hours_to_codes(in_data=None):
    hours_categories = {1500:0, 1530:1, 1600:2, 1630:3, 1700:4, 1730:5, 1800:6, 1830:7, 1900:8, 1930:9,
                        2000:10, 2030:11, 2100:12, 2130:13, 2200:14, 2230:15, 2300:16, 2330:17, 2400:18,
                        0:19, 30:20, 100:21, 130:22, 200:23, 230:24, 300:25, 330:26, 400:27, 430:28,
                        500:29, 530:30, 600:31, 630:32, 700:33, 730:34, 800:35, 830:36, 900:37, 930:38,
                        1000:39, 1030:40, 1100:41, 1130:42, 1200:43, 1230:44, 1300:45, 1330:46, 1400:47, 1430:48}

    in_data['youthreport2_shq1'] = in_data.youthreport2_shq1.replace(hours_categories)
    in_data['youthreport2_shq2'] = in_data.youthreport2_shq2.replace(hours_categories)
    in_data['youthreport2_shq3'] = in_data.youthreport2_shq3.replace(hours_categories)
    in_data['youthreport2_shq4'] = in_data.youthreport2_shq4.replace(hours_categories)

    return in_data

# Split features in categories based on 'Prediction Variables_Depression_NCANDAV5_1510'
def get_feature_categories(input_feats=None, data=None):

    demographics_f = input_feats['demographics'][input_feats['demographics'].notna()]
    demographics = data.loc[:, data.columns.intersection(demographics_f)]

    drug_use_f = input_feats['drug_use'][input_feats['drug_use'].notna()]
    drug_use = data.loc[:, data.columns.intersection(drug_use_f)]

    physical_f = input_feats['physical'][input_feats['physical'].notna()]
    physical = data.loc[:, data.columns.intersection(physical_f)]

    life_f = input_feats['life'][input_feats['life'].notna()]
    life = data.loc[:, data.columns.intersection(life_f)]

    personality_f = input_feats['personality'][input_feats['personality'].notna()]
    personality = data.loc[:, data.columns.intersection(personality_f)]

    brief_f = input_feats['brief'][input_feats['brief'].notna()]
    brief = data.loc[:, data.columns.intersection(brief_f)]

    neuropsych_f = input_feats['neuropsych'][input_feats['neuropsych'].notna()]
    neuropsych = data.loc[:, data.columns.intersection(neuropsych_f)]

    sleep_f = input_feats['sleep'][input_feats['sleep'].notna()]
    sleep = data.loc[:, data.columns.intersection(sleep_f)]

    possible_categories = ['demographics', 'drug_use', 'physical', 'life',
                           'personality', 'brief', 'neuropsych', 'sleep']

    categories = [demographics, drug_use, physical, life,
                  personality, brief, neuropsych, sleep]

    categories_dictionary = dict(zip(possible_categories, categories))

    return categories_dictionary


def categorical_to_onehot(in_data=None, path=None, write_csv=False):

    # Variables in categorical format that need to be converted
    categorical = ['site', 'sex', 'cahalan_score', 'hispanic', 'race']
    # categorical = ['sex', 'cahalan_score']

    for category in categorical:
        # using the same example as above
        df = pd.DataFrame({category: in_data.loc[:, category]})

        one_hot = pd.get_dummies(df[category], prefix=category, drop_first=False)

        # use pd.concat to join the new columns with your original dataframe
        in_data = pd.concat([in_data, one_hot], axis=1)

        # now drop the original 'country' column (you don't need it anymore)
        in_data.drop([category], axis=1, inplace=True)

    if write_csv:
        in_data.to_csv(os.join(path, 'onehot_data.csv', index = False))
    return in_data


def drop_confounders(in_data=None, age=False):

    #confounders = ['scanner', 'site', 'scanner_model', 'race', 'hispanic']
    confounders = ['scanner', 'scanner_model']

    if age:
        confounders.append('visit_age')
    in_data = in_data.drop(columns=confounders)

    return in_data


def rename_constructs(in_data=None):
    in_data = in_data.rename(columns={'ysr_anxdep_t':'negative_valence_anxiety'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section16_ysr103':'negative_valence_sad'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section2_ysr08':'cognitive_concentration'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section13_ysr78':'cognitive_distracted'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section16_ysr100':'arousal_trouble_sleeping'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section8_ysr54':'arousal_overtired'})
    in_data = in_data.rename(columns={'youthreport1_ysr_section1_ysr05':'positive_valence_v2'})

    return in_data


def sum_stroop(in_data=None):

    in_data['stroop_error_sum'] = in_data['stroop_conm_rr_error'] + in_data['stroop_conm_rs_error'] + in_data['stroop_connm_rr_error'] + \
                   in_data['stroop_connm_rr_error'] + in_data['stroop_incm_rr_error'] + in_data['stroop_incm_rs_error'] + \
                   in_data['stroop_incm_rs_error'] + in_data['stroop_incm_rs_error']

    in_data['stroop_miss_sum'] = in_data['stroop_conm_rr_miss'] + in_data['stroop_conm_rs_miss'] + in_data['stroop_connm_rr_miss'] + \
                   in_data['stroop_connm_rr_miss'] + in_data['stroop_incm_rr_miss'] + in_data['stroop_incm_rs_miss'] + \
                   in_data['stroop_incm_rs_miss'] + in_data['stroop_incm_rs_miss']

    return in_data


def drop_features(in_data=None, age=False):

    reduntant = ['youthreport2_rsq_sec1_rsq1', 'youthreport2_rsq_sec1_rsq5',
                 'youthreport2_rsq_sec2_rsq9', 'youthreport2_rsq_sec1_rsq2', 'youthreport2_rsq_sec2_rsq12',
                 'youthreport2_rsq_sec2_rsq7', 'youthreport2_rsq_sec1_rsq3', 'youthreport2_rsq_sec1_rsq4',
                 'youthreport2_rsq_sec2_rsq10', 'youthreport2_rsq_sec1_rsq6', 'youthreport2_rsq_sec4_rsq19',
                 'youthreport2_rsq_sec4_rsq20', 'youthreport2_rsq_sec2_rsq8', 'youthreport2_rsq_sec3_rsq17',
                 'youthreport2_rsq_sec3_rsq18', 'youthreport2_rsq_sec3_rsq13', 'youthreport2_rsq_sec3_rsq14',
                 'youthreport2_rsq_sec3_rsq15', 'np_ehi_lh', 'np_ehi_rh', 'stroop_conm_rr_error', 'stroop_conm_rs_error',
                 'stroop_connm_rr_error', 'stroop_connm_rs_error', 'stroop_incm_rr_error', 'stroop_incm_rs_error',
                 'stroop_incnm_rr_error', 'stroop_incnm_rs_error', 'stroop_conm_rr_miss', 'stroop_conm_rs_miss',
                 'stroop_connm_rr_miss', 'stroop_connm_rs_miss', 'stroop_incm_rr_miss', 'stroop_incm_rs_miss',
                 'stroop_incnm_rr_miss', 'stroop_incnm_rs_miss', 'stroop_mean_3stdl', 'stroop_mean_3stdu', 'stroop_total_std', 'stroop_total_median']

    in_data = in_data.drop(columns=reduntant)

    return in_data


def remove_18(in_data=None):

    in_data = in_data.loc[in_data['visit_age'] < 18]

    return in_data


def bootstrap(in_data=None):
    subj = np.unique(in_data['subject'])
    bootstrapped_subj = np.random.choice(subj, size=len(subj), replace=True)
    new_in_data = pd.DataFrame(columns=list(in_data.columns))
    for subj_i in bootstrapped_subj:
        subj_vis = in_data[in_data['subject'] == subj_i]
        new_in_data = new_in_data.append(subj_vis, ignore_index=True)

    return new_in_data
