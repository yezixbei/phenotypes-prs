import pandas as pd
from main import create_wide_table, get_stats, get_associations

def test_create_wide_table():
    test_phenotypes,test_demographics,test_prs=[],[],[]

    test_phenotypes.append({
        'uuid':"1",
        'phenotype_id':"a",
        'value':1.0
    })
    test_phenotypes.append({
        'uuid':"2",
        'phenotype_id':"a",
        'value':2.0
    })
    test_phenotypes.append({
        'uuid':"3",
        'phenotype_id':"a",
        'value':3.0
    })
    test_phenotypes.append({
        'uuid':"4",
        'phenotype_id':"a",
        'value':None
    })
    test_phenotypes.append({
        'uuid':"5",
        'phenotype_id':"b",
        'value':1.0
    })
    test_phenotypes.append({
        'uuid':"6",
        'phenotype_id':"b",
        'value':2.0
    })

    test_prs.append({
        'uuid':"1",
        'prs':1.0,
    })
    test_prs.append({
        'uuid':"2",
        'prs':2.0,
    })
    test_prs.append({
        'uuid':"3",
        'prs':3.0,
    })
    test_prs.append({
        'uuid':"4",
        'prs':-1.0,
    })
    test_prs.append({
        'uuid':"5",
        'prs':-1.0,
    })
    test_prs.append({
        'uuid':"6",
        'prs':-2.0,
    })

    test_demographics.append({
        'uuid':"1",
        'age_at_progression_enrollment':50,
        'sexM':0,
        'smoking_status':"past_5_to_10_years"
    })
    test_demographics.append({
        'uuid':"2",
        'age_at_progression_enrollment':50,
        'sexM':0,
        'smoking_status':"past_5_to_10_years"
    })
    test_demographics.append({
        'uuid':"3",
        'age_at_progression_enrollment':50,
        'sexM':0,
        'smoking_status':"past_5_to_10_years"
    })
    test_demographics.append({
        'uuid':"4",
        'age_at_progression_enrollment':50,
        'sexM':0,
        'smoking_status':"past_5_to_10_years"
    })
    test_demographics.append({
        'uuid':"5",
        'age_at_progression_enrollment':30,
        'sexM':1,
        'smoking_status':"never"
    })
    test_demographics.append({
        'uuid':"6",
        'age_at_progression_enrollment':30,
        'sexM':1,
        'smoking_status':"never"
    })

    return create_wide_table(pd.DataFrame(test_phenotypes), pd.DataFrame(test_demographics), pd.DataFrame(test_prs))


def test_get_stats():
    stats=[]
    stats.append({
        'phenotype_id':"a",
        'not_missing_count': 3,
        'missingness_rate': 0.25,
        'avg_value': 2.0,
        'median_value': 2.0,
        'std_value': 1.0,
        'avg_age': 50.0
    })
    stats.append({
        'phenotype_id':"b",
        'not_missing_count': 2,
        'missingness_rate': 0.0,
        'avg_value': 1.5,
        'median_value': 1.5,
        'std_value': 0.707107,
        'avg_age': 30.0
    })
    wide_table = test_create_wide_table()
    assert pd.DataFrame(stats).equals(get_stats(wide_table))

def test_get_associations():
    associations=[]
    associations.append({
        'phenotype_id': "a",
        'coef': 1.0,
        'intercept': 0.0,
        'r_squared': 1.0
    })
    associations.append({
        'phenotype_id': "b",
        'coef': -1.0,
        'intercept': 0.0,
        'r_squared': 1.0
    })
    max_score = 1

    wide_table = test_create_wide_table()
    test_associations, test_max_score = get_associations(wide_table)
    assert max_score==test_max_score[1]
    assert pd.DataFrame(associations).equals(test_associations)

if __name__ == '__main__':
    test_get_stats()
    test_get_associations()