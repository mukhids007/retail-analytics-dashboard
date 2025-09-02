import pandas as pd
import numpy as np
from datetime import datetime
import re

"""
Retail_Live_Project_Dataset – Cleaning Utility
----------------------------------------------
• Reads either **Retail_Live_Project_Dataset.xlsx** (default) or a user‑supplied path.
• Applies column‑name normalisation, type coercion, category harmonisation and
  data‑quality flagging tailored to the schema that ships with the live project
  workbook (≈ 16 columns).
• Writes a side‑by‑side file with **_cleaned** suffix, preserving the original
  file format (CSV ⇆ Excel).

Run from shell:
    python clean_retail_data.py  # uses default Excel file in cwd
    python clean_retail_data.py my_raw_file.csv
"""


CATEGORY_MAPPINGS: dict[str, str] = {
    "laptop": "Electronics",
    "sofa": "Home",
    "shoes": "Fashion",
    "shampoo": "Beauty",
    "novel": "Books",
}

TEXT_COLUMNS = [
    "City",
    "Product_Category",
    "Product_Name",
    "Payment_Mode",
    "Delivery_Status",
    "Customer_Name",
    "Email",
    "Gender",
    "Country",
]

NUMERIC_COLUMNS = [
    "Purchase_Amount",
    "Discount_Offered",
    "Customer_Satisfaction",
    "Age",
]

DATE_COLUMNS = ["Purchase_Date"]

PAYMENT_MODE_MAP = {
    "upi": "UPI",
    "debit card": "Debit Card",
    "credit card": "Credit Card",
    "cash": "Cash",
    "net banking": "Net Banking",
}

DELIVERY_STATUS_MAP = {
    "delivered": "Delivered",
    "pending": "Pending",
    "cancelled": "Cancelled",
    "returned": "Returned",
}

GENDER_MAP = {"male": "Male", "female": "Female", "m": "Male", "f": "Female"}

EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _read_file(path: str) -> pd.DataFrame:
    """Auto‑detect loader based on extension."""
    if path.lower().endswith(".csv"):
        return pd.read_csv(path)
    if path.lower().endswith((".xls", ".xlsx")):
        return pd.read_excel(path)
    raise ValueError("Unsupported file format → use CSV or Excel.")


def _write_file(df: pd.DataFrame, original_path: str) -> str:
    """Save cleaned data next to original and return the new path."""
    cleaned_path = re.sub(r"\.(csv|xlsx?|xls)$", r"_cleaned.\1", original_path, flags=re.I)
    if cleaned_path.lower().endswith(".csv"):
        df.to_csv(cleaned_path, index=False)
    else:
        df.to_excel(cleaned_path, index=False)
    return cleaned_path


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Primary cleaning pipeline aligned with Retail_Live_Project_Dataset schema."""

    df = df.copy()

    # 1️⃣ column name hygiene
    df.columns = df.columns.str.strip()

    # 2️⃣ text normalisation
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # 3️⃣ product category correction (vectorised for speed)
    mask_any = pd.Series(False, index=df.index)
    for key, cat in CATEGORY_MAPPINGS.items():
        key_mask = df["Product_Name"].str.lower().str.contains(key, na=False)
        df.loc[key_mask, "Product_Category"] = cat
        mask_any |= key_mask
    # the rest remain as‑is

    # 4️⃣ numeric coercion
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 5️⃣ date parsing
    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=False)

    # 6️⃣ categorical harmonisation
    if "Payment_Mode" in df.columns:
        df["Payment_Mode"] = (
            df["Payment_Mode"].str.lower().map(PAYMENT_MODE_MAP).fillna(df["Payment_Mode"])
        )
    if "Delivery_Status" in df.columns:
        df["Delivery_Status"] = (
            df["Delivery_Status"].str.lower().map(DELIVERY_STATUS_MAP).fillna(df["Delivery_Status"])
        )
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].str.lower().map(GENDER_MAP).fillna(df["Gender"])

    # 7️⃣ email validation & formatting
    if "Email" in df.columns:
        df["Email"] = df["Email"].str.lower().str.strip()
        df["Email_Valid"] = df["Email"].str.match(EMAIL_REGEX, na=False)

    # 8️⃣ satisfaction clipping (1–5)
    if "Customer_Satisfaction" in df.columns:
        df["Customer_Satisfaction"] = df["Customer_Satisfaction"].clip(1, 5)

    # 9️⃣ proper‑case cities / countries
    for loc_col in ("City", "Country"):
        if loc_col in df.columns:
            df[loc_col] = df[loc_col].str.title()

    # 🔟 quality flags
    df["Data_Quality_Issues"] = ""
    if "Customer_Name" in df.columns:
        miss_name = df["Customer_Name"].eq("") | df["Customer_Name"].isna()
        df.loc[miss_name, "Data_Quality_Issues"] += "Missing Customer Name; "
    if "Email" in df.columns:
        df.loc[~df["Email_Valid"], "Data_Quality_Issues"] += "Invalid Email; "
    if "Purchase_Amount" in df.columns:
        bad_amt = df["Purchase_Amount"].le(0) | df["Purchase_Amount"].isna()
        df.loc[bad_amt, "Data_Quality_Issues"] += "Invalid Purchase Amount; "

    # optional summary to stdout (can be removed in production)
    print("Cleaning summary →", {
        "rows": len(df),
        "issues": (df["Data_Quality_Issues"] != "").sum(),
    })

    return df


def load_clean_validate(file_path: str) -> pd.DataFrame | None:
    """Convenience wrapper used by __main__."""
    try:
        df_raw = _read_file(file_path)
        df_clean = clean_dataset(df_raw)
        out_path = _write_file(df_clean, file_path)
        print(f"✔ Saved cleaned data → {out_path}\n")

        # simple validation report
        print("Validation snapshot:\n",
              df_clean.isnull().sum().to_frame("nulls").T)
        return df_clean
    except Exception as exc:
        print("❌", exc)
        return None


if __name__ == "__main__":
    import sys, pathlib

    # default to the live‑project workbook in current directory
    default_file = "Retail_Live_Project_Dataset.xlsx"
    target = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(default_file)

    if not target.exists():
        raise FileNotFoundError(f"File not found → {target.resolve()}")

    load_clean_validate(str(target))
