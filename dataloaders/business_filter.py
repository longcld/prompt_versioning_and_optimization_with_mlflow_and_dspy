import json
import pandas as pd
import random
from dspy import Example
from setup import CONFIG

print("📂 Loading and preparing dataset...")


def load_and_prepare_data():
    print(f"🔍 Filtering dataset from {CONFIG['DATA_PATH']}")
    with open(CONFIG["DATA_PATH"], "r") as f:
        raw_data = json.load(f)

    dataset = [
        item for item in raw_data if item["Category"] == CONFIG["CATEGORY"]
    ]

    dspy_ds = [
        Example(
            question=item["Question"],
            in_scope=item["Scope"] == "in scope"
        ).with_inputs("question") for item in dataset
    ]

    random.Random(0).shuffle(dspy_ds)

    eval_samples = int(len(dspy_ds) * CONFIG["EVAL_SAMPLE_RATE"])
    train_set = dspy_ds[:-eval_samples]
    eval_set = dspy_ds[-eval_samples:]

    print(f"✅ Dataset prepared: {len(train_set)} train, {len(eval_set)} eval")

    return train_set, eval_set


train_set, eval_set = load_and_prepare_data()

print("=" * 60)
print()

print("📖 Preparing scope descriptions from Excel...")

business_filter_file_path = "data/business_filter.xlsx"


def get_scope_description(business_name):
    """Get scope description for a specific business."""
    df = pd.read_excel(business_filter_file_path, sheet_name=business_name)
    df = df[[
        "category",
        "intent",
        "detail",
        "scope"
    ]]

    mapping_scope = {
        "Inscope": "Inscope",
        "Outscope": "Outscope",
        "Ủy quyền": "Inscope",
        "Không ủy quyền": "Outscope"
    }

    df['scope'] = df['scope'].map(mapping_scope)
    df['scope'] = df['scope'].fillna("OOS")

    df = df[df['scope'] != "OOS"]

    category = df['category'].unique().tolist()
    inscope_df = df[df['scope'] == "Inscope"]
    outscope_df = df[df['scope'] == "Outscope"]

    inscope_df = inscope_df.drop(columns=['scope'])
    outscope_df = outscope_df.drop(columns=['scope'])

    inscope = json.dumps(
        inscope_df.to_dict(orient='records'),
        indent=2,
        ensure_ascii=False
    )
    outscope = json.dumps(
        outscope_df.to_dict(orient='records'),
        indent=2,
        ensure_ascii=False
    )

    return category, f"<Inscope>\n{inscope}\n</Inscope>", f"<Outscope>\n{outscope}\n</Outscope>"


category, inscope, outscope = get_scope_description(CONFIG["CATEGORY"].lower())

print("✅ Scope descriptions prepared:")
print(f"Categories: {category}")
print(f"Inscope: {inscope}")
print(f"Outscope: {outscope}")
print("=" * 60)
print()
