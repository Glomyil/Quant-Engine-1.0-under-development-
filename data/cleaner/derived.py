import pandas as pd
import numpy as np
from data.cleaner import base


def run(df: pd.DataFrame, params: dict, ctx: dict) -> pd.DataFrame:
    base.check_not_empty(df, context="derived")
    base.check_required_columns(df)
    base.check_required_columns(df, ["hfq_close"])

    out = df.copy()

    px = pd.to_numeric(out["hfq_close"], errors="coerce").ffill()
    out["ret_1d_hfq"] = px.pct_change(fill_method=None)
    r = out["ret_1d_hfq"].to_numpy(dtype="float")

    out.attrs["derived"] = {
        "ret_nan_rows": int(np.isnan(r).sum()),
        "ret_inf_rows": int(np.isinf(r).sum()),
    }
    return out
