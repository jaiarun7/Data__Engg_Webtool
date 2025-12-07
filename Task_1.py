import math
import re
import hashlib
from pathlib import Path
from datetime import datetime, date

import pandas as pd
import numpy as np
import psycopg2
from psycopg2.extras import execute_values


class IncrementalETL:
    def __init__(self, db_config):
        """
        Initialize ETL pipeline with database configuration

        Example db_config:
        {
            'host': 'localhost',
            'database': 'postgres',
            'user': 'postgres',
            'password': '1234',
            'port': 5432
        }
        """
        self.db_config = db_config
        self.conn = None
        self.cursor = None

    def connect_db(self):
        """Establish database connection"""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cursor = self.conn.cursor()
            print("Database connection established")
        except Exception as e:
            print(f"Database connection failed: {e}")
            raise

    def close_db(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("Database connection closed")

    def extract(self, csv_path):
        """Extract data from CSV file"""
        try:
            df = pd.read_csv(csv_path)
            print(f"Extracted {len(df)} records from {csv_path}")
            return df
        except Exception as e:
            print(f"Extraction failed: {e}")
            raise

    def transform(self, df):
        """Apply transformations to the dataframe - kept consistent with original logic"""
        print("Starting transformations...")

        # Normalize literal tokens to pandas NA so coercions are consistent
        df.replace({'NaN': pd.NA, 'nan': pd.NA, 'NULL': pd.NA, 'null': pd.NA, '': pd.NA}, inplace=True)

        # 1. Trim whitespace from all string columns (same as original)
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].str.strip()
        print("  Trimmed whitespace")

        # 2. Standardize date formats (assuming 'created_date' column exists)
        if 'created_date' in df.columns:
            parsed = pd.to_datetime(df['created_date'], errors='coerce')
            # keep a string version if desirable (mirrors earlier behavior)
            df['created_date_str'] = parsed.dt.strftime('%Y-%m-%d')
            # for DB use python.date objects
            df['created_date'] = parsed.dt.date
        print("  Standardized date formats")

        # 3. Normalize phone numbers (remove special chars, keep digits only)
        if 'phone' in df.columns:
            df['phone'] = df['phone'].apply(self._normalize_phone)
        print("  Normalized phone numbers")

        # 4. Add derived column: lead_age_days (safe)
        if 'created_date' in df.columns:
            now = datetime.now().date()

            def safe_age(x):
                if pd.isna(x) or x is None:
                    return None
                if isinstance(x, date) and not isinstance(x, datetime):
                    return (now - x).days
                try:
                    parsed = pd.to_datetime(x, errors='coerce')
                    if pd.isna(parsed):
                        return None
                    return (now - parsed.date()).days
                except Exception:
                    return None

            df['lead_age_days'] = df['created_date'].apply(safe_age)
        print("  Added lead_age_days column")

        # 5. Create row_hash for incremental logic (composite of key fields)
        df['row_hash'] = df.apply(self._generate_row_hash, axis=1)
        print("  Generated row hashes")

        # 6. Add ETL timestamp
        df['etl_loaded_at'] = datetime.now()

        # Final: convert pandas NA / NaT / np.nan to Python None for safe DB insertion
        df = df.where(pd.notnull(df), None)

        print(f"Transformation complete: {len(df)} records processed")
        return df

    def _normalize_phone(self, phone):
        """Normalize phone number to digits only"""
        if phone is None or (isinstance(phone, float) and math.isnan(phone)):
            return None
        digits_only = re.sub(r'\D', '', str(phone))
        return digits_only if digits_only else None

    def _generate_row_hash(self, row):
        """Generate hash from key fields for change detection"""
        exclude = {'row_hash', 'etl_loaded_at', 'lead_age_days'}
        parts = []
        for col in row.index:
            if col in exclude:
                continue
            val = row[col]
            if pd.isna(val) or val is None:
                parts.append('')
            else:
                if isinstance(val, date) and not isinstance(val, datetime):
                    parts.append(val.isoformat())
                elif isinstance(val, datetime):
                    parts.append(val.isoformat())
                else:
                    parts.append(str(val))
        combined = '|'.join(parts)
        return hashlib.md5(combined.encode('utf-8')).hexdigest()

    def get_existing_hashes(self, table_name, id_column='id'):
        """Retrieve existing row hashes from database"""
        try:
            query = f"SELECT {id_column}, row_hash FROM {table_name}"
            self.cursor.execute(query)
            existing = {row[0]: row[1] for row in self.cursor.fetchall()}
            print(f"Retrieved {len(existing)} existing records")
            return existing
        except Exception as e:
            print(f"Failed to retrieve existing hashes: {e}")
            return {}

    def _normalize_missing_token(self, x):
        """Normalize many missing token types to Python None"""
        if x is None:
            return None
        if x is pd.NA:
            return None
        if isinstance(x, (np.generic,)):
            try:
                x = x.item()
            except Exception:
                pass
        try:
            if isinstance(x, float) and math.isnan(x):
                return None
        except Exception:
            pass
        if isinstance(x, str):
            s = x.strip()
            if s == '':
                return None
            if s.lower() in ('nan', 'nat', 'null', 'none'):
                return None
            return s
        return x

    def _get_table_columns(self, table_name):
        """Return list of column names for the target table (public schema)"""
        # Ensure cursor exists
        if self.cursor is None:
            raise RuntimeError("DB cursor not initialized. Call connect_db() first.")
        q = """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = %s
              AND table_schema = 'public'
            ORDER BY ordinal_position
        """
        self.cursor.execute(q, (table_name,))
        cols = [r[0] for r in self.cursor.fetchall()]
        return cols

    def load_incremental(self, df, table_name, id_column='id'):
        """Load data incrementally using upsert logic"""
        print("Starting incremental load...")

        # Get existing records
        existing_hashes = self.get_existing_hashes(table_name, id_column)

        # Identify new and updated records
        if id_column in df.columns:
            df['is_new'] = ~df[id_column].isin(existing_hashes.keys())
            df['is_updated'] = df.apply(
                lambda row: (row[id_column] in existing_hashes) and
                           (existing_hashes.get(row[id_column]) != row['row_hash']),
                axis=1
            )
        else:
            df['is_new'] = True
            df['is_updated'] = False

        new_records = df[df['is_new'] == True]
        updated_records = df[df['is_updated'] == True]

        print(f"  New records: {len(new_records)}")
        print(f"  Updated records: {len(updated_records)}")
        print(f"  Unchanged records: {len(df) - len(new_records) - len(updated_records)}")

        # Prepare load_df (only new/updated)
        load_df = df[(df['is_new'] == True) | (df['is_updated'] == True)].copy()
        load_df = load_df.drop(columns=['is_new', 'is_updated'], errors='ignore')

        if len(load_df) == 0:
            print("No new or updated records to load")
            return

        # Strict per-cell cleaning to prevent np.nan slipping into DB
        load_df = load_df.astype(object).copy()
        load_df = load_df.applymap(self._normalize_missing_token)

        # Enforce created_date is python.date or None
        if 'created_date' in load_df.columns:
            def enforce_date(v):
                if v is None:
                    return None
                if isinstance(v, date) and not isinstance(v, datetime):
                    return v
                if isinstance(v, datetime):
                    return v.date()
                try:
                    parsed = pd.to_datetime(v, errors='coerce')
                    if pd.isna(parsed):
                        return None
                    return parsed.date()
                except Exception:
                    return None
            load_df['created_date'] = load_df['created_date'].apply(enforce_date)

        # Final cleanup: numpy scalars -> python, float nan -> None
        def final_cleanup(x):
            if isinstance(x, (np.generic,)):
                try:
                    x = x.item()
                except Exception:
                    pass
            try:
                if isinstance(x, float) and math.isnan(x):
                    return None
            except Exception:
                pass
            return x
        load_df = load_df.applymap(final_cleanup)

        # Validate created_date column only has date or None
        if 'created_date' in load_df.columns:
            invalid_mask = load_df['created_date'].apply(lambda v: (v is not None) and (not isinstance(v, date)))
            if invalid_mask.any():
                bad_sample = load_df[invalid_mask].head(20).to_dict('records')
                print("ERROR: created_date contains invalid types after cleaning. Sample:")
                print(bad_sample)
                raise ValueError("created_date must be python datetime.date or None for every row.")

        # Drop helper-only column if present (we created this earlier)
        if 'created_date_str' in load_df.columns:
            load_df = load_df.drop(columns=['created_date_str'])

        # --- IMPORTANT: filter DataFrame columns to only those that exist in the DB table ---
        # This avoids trying to insert non-existent columns like created_date_str
        table_columns = self._get_table_columns(table_name)
        # Keep only intersection and preserve order according to load_df
        cols_to_insert = [c for c in load_df.columns if c in table_columns]
        dropped = [c for c in load_df.columns if c not in cols_to_insert]
        if dropped:
            print(f"Note: Dropping {len(dropped)} DataFrame columns not present in table: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
        load_df = load_df[cols_to_insert]

        # Handle lead_id SERIAL behavior:
        use_conflict = False
        if id_column in load_df.columns:
            all_none = all(val is None for val in load_df[id_column].tolist())
            if all_none:
                load_df = load_df.drop(columns=[id_column])
                use_conflict = False
            else:
                use_conflict = True
        else:
            use_conflict = False

        # Build upsert SQL
        columns = load_df.columns.tolist()
        if len(columns) == 0:
            print("No columns left to insert after filtering to DB table columns. Aborting load.")
            return

        if use_conflict and id_column in columns:
            upsert_sql = f"""
INSERT INTO {table_name} ({', '.join(columns)})
VALUES %s
ON CONFLICT ({id_column})
DO UPDATE SET
    {', '.join(f"{col} = EXCLUDED.{col}" for col in columns if col != id_column)}
"""
        else:
            upsert_sql = f"""
INSERT INTO {table_name} ({', '.join(columns)})
VALUES %s
"""

        # Build clean values list
        records = load_df.to_dict('records')

        def clean_cell_for_db(col_name, val):
            if isinstance(val, (np.generic,)):
                try:
                    val = val.item()
                except Exception:
                    pass
            if val is None:
                return None
            try:
                if isinstance(val, float) and math.isnan(val):
                    return None
            except Exception:
                pass
            if col_name == 'created_date':
                if isinstance(val, date) and not isinstance(val, datetime):
                    return val
                if isinstance(val, datetime):
                    return val.date()
                try:
                    parsed = pd.to_datetime(val, errors='coerce')
                    if pd.isna(parsed):
                        return None
                    return parsed.date()
                except Exception:
                    return None
            return val

        values = [tuple(clean_cell_for_db(col, rec.get(col)) for col in columns) for rec in records]

        # Sanity check sample
        sample_problems = []
        for i, rec in enumerate(values[:50]):
            for j, val in enumerate(rec):
                if isinstance(val, float) and math.isnan(val):
                    sample_problems.append({'row': i, 'col': columns[j], 'val': val})
        if sample_problems:
            print("DEBUG: Found float('nan') in sample values:", sample_problems)
            raise ValueError("float('nan') still present in values sample; aborting to avoid DB type errors.")

        # Execute upsert
        try:
            execute_values(self.cursor, upsert_sql, values)
            self.conn.commit()
            print(f"Successfully loaded {len(values)} records")
        except Exception as e:
            self.conn.rollback()
            print(f"Load failed: {e}")
            raise

    def run_pipeline(self, csv_path, table_name, id_column='id'):
        """Execute complete ETL pipeline"""
        print("\n" + "="*60)
        print("STARTING ETL PIPELINE")
        print("="*60 + "\n")
        try:
            df = self.extract(csv_path)
            df = self.transform(df)
            self.connect_db()
            self.load_incremental(df, table_name, id_column)
            print("\n" + "="*60)
            print("ETL PIPELINE COMPLETED SUCCESSFULLY")
            print("="*60 + "\n")
        except Exception as e:
            print(f"\nPipeline failed: {e}")
            raise
        finally:
            self.close_db()


# Example usage (same as your original)
if __name__ == "__main__":
    db_config = {
        'host': 'localhost',
        'database': 'postgres',
        'user': 'postgres',
        'password': '1234',
        'port': 5432
    }

    etl = IncrementalETL(db_config)

    etl.run_pipeline(
        csv_path=Path(r"C:\Task 1\leads_data.csv"),
        table_name='leads',
        id_column='lead_id'
    )
