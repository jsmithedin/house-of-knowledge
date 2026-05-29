import sys
from datetime import datetime, timezone
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import streamlit as st

from app.config import Settings
from app.usage import UsageStore

st.set_page_config(page_title="Token usage", page_icon="📊", layout="wide")

settings = Settings()
store = UsageStore(settings.usage_db_path)
store.rollup_stale_months()

now = datetime.now(timezone.utc)
month_label = now.strftime("%B %Y")

st.title("📊 Token usage")
st.caption(f"{month_label} (UTC)")

summary = store.get_current_month_summary()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Requests", summary["request_count"])
c2.metric("Input tokens", f"{summary['input_tokens']:,}")
c3.metric("Output tokens", f"{summary['output_tokens']:,}")
c4.metric("Est. cost (USD)", f"${summary['estimated_usd']:.4f}")

if store.has_unknown_model_events():
    st.caption("Cost unknown for one or more models (no pricing configured).")

st.subheader("Request log")
events = store.get_current_month_events()
if events:
    st.dataframe(
        [
            {
                "Time (UTC)": e["created_at"],
                "Model": e["model_id"],
                "Input": e["input_tokens"],
                "Output": e["output_tokens"],
                "Est. USD": f"${e['estimated_usd']:.6f}",
            }
            for e in events
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No requests recorded this month yet.")

st.subheader("Previous months")
history = store.get_monthly_history()
if history:
    st.dataframe(
        [
            {
                "Month": h["year_month"],
                "Requests": h["request_count"],
                "Input": h["input_tokens"],
                "Output": h["output_tokens"],
                "Est. USD": f"${h['estimated_usd']:.4f}",
            }
            for h in history
        ],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No completed months yet.")
