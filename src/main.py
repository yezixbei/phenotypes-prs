import pandas as pd
import pandasql as ps
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from pandera import DataFrameSchema, Column, errors
import matplotlib.pyplot as plt


__version__ = "1.0.0"

# file paths
phenotypes_path, demographics_path, prs_path = 'in/phenotypes.parquet','in/demographics.tsv','in/prs.tsv'
stats_path, associations_path, associations_png_path = 'out/stats.parquet', 'out/associations.parquet', 'out/associations.png'

# output schemas
stats_schema = DataFrameSchema({
    "phenotype_id": Column(str),
    "not_missing_count": Column(int),
    "missingness_rate": Column(float),
    "avg_value": Column(float),
    "median_value": Column(float),
    "std_value": Column(float),
    "avg_age": Column(float),
})
associations_schema = DataFrameSchema({
    "phenotype_id": Column(str),
    'coef': Column(float),
    'intercept': Column(float),
    "r_squared": Column(float),
})


# pipeline starts here
# ingest data
def load_files(p_path, demo_path, prs_path):
    phenotypes = pd.read_parquet(p_path)
    demographics = pd.read_csv(demo_path,sep='\t')
    prs = pd.read_csv(prs_path,sep='\t')

    return phenotypes, demographics, prs

# improve data quality
def create_wide_table(phenotypes, demographics, prs):
    query='''
    SELECT 
        p.uuid as uuid,
        phenotype_id,
        value,
        prs,
        age_at_progression_enrollment,
        sexM,
        smoking_status
    FROM phenotypes p
    LEFT JOIN prs r
    ON p.uuid = r.uuid
    LEFT JOIN demographics d
    ON p.uuid = d.uuid
    '''
    df = ps.sqldf(query)

    return df

# transformations
def get_stats(df):
    stats=[]
    grouped = df.groupby('phenotype_id')
    for name, group in grouped: 
        stats.append({
            'phenotype_id': name,
            'not_missing_count':group['value'].notna().sum(),
            'missingness_rate': group['value'].isna().sum() *1.0 / (group['value'].notna().sum()+group['value'].isna().sum()),
            'avg_value': group['value'].mean(),
            'median_value': group['value'].median(),
            'std_value': round(group['value'].std(),6),
            'avg_age': group['age_at_progression_enrollment'].mean(),
        })

    out=pd.DataFrame(stats)
    return out

def get_associations(df):
    associations, max_score = [], [-1,-1]
    smoking_order = ['never', 'past_more_than_10_years', 'past_5_to_10_years', 'past_less_than_5_years', 'current']
    encoder = OrdinalEncoder(categories=[smoking_order])

    df = df.dropna()
    df['smoking_status'] = encoder.fit_transform(df[['smoking_status']])
    grouped = df.groupby('phenotype_id')

    for phenotype, group in grouped:

        X = group[['prs', 'age_at_progression_enrollment', 'sexM', 'smoking_status']]
        y = group['value']
        model = LinearRegression().fit(X, y)
        score = round(model.score(X, y), 4)

        if score > max_score[1]:
            max_score[0] = phenotype
            max_score[1] = score

        associations.append({
            'phenotype_id': phenotype,
            'coef': round(model.coef_[0], 4),
            'intercept': round(model.intercept_, 4),
            'r_squared': score
        })

    out=pd.DataFrame(associations)
    print( max_score[0]+": "+str(max_score[1]) )
    return out,max_score


# loading & publishing
def validate_schemas(stats, associations):
    try:
        stats_schema.validate(stats)
        print("stats_schema is valid!")
    except errors.SchemaError as e:
        print("Validation error:", e)

    try:
        associations_schema.validate(associations)
        print("associations_schema is valid!")
    except errors.SchemaError as e:
        print("Validation error:", e)

def visualize(associations):
    plt.bar(associations['phenotype_id'], associations['r_squared'])

    plt.xlabel('phenotype_id')
    plt.ylabel('r_squared')
    plt.title('Phenotype vs PRS')

    plt.savefig(associations_png_path, format="png")



if __name__ == '__main__':
    # ingest data
    phenotypes, demographics, prs=load_files(phenotypes_path, demographics_path, prs_path)

    # improve data quality
    merged=create_wide_table(phenotypes, demographics, prs)

    # transformations
    stats = get_stats(merged)
    associations,_ = get_associations(merged)

    # loading & publishing
    validate_schemas(stats, associations)
    visualize(associations)
    stats.to_parquet(stats_path)
    associations.to_parquet(associations_path)