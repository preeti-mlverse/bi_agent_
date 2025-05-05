# collaboration.py
import streamlit as st
import os
import json
from utils.integrations import (
    send_slack_alert,
    send_email_gmail,
    write_to_google_sheet,
    send_teams_alert,
    create_hubspot_contact
)

CONFIG_PATH = "configs/collab_config.json"

AVAILABLE_TOOLS = {
    "Slack": ["Webhook URL"],
    "Gmail": ["Client ID", "Client Secret", "Redirect URI"],
    "Google Sheets": ["Service Account JSON"],
    "Microsoft Teams": ["Webhook URL"],
    "Outlook": ["Client ID", "Tenant ID", "Secret"],
    "Hubspot CRM": ["API Key"],
    "Zoho CRM": ["Client ID", "Client Secret", "Refresh Token"]
}

def load_configs():
    config_path = "configs/collab_config.json"
    if not os.path.exists(config_path):
        return {}  # return empty config if file doesn't exist

    with open(config_path, "r") as f:
        try:
            content = f.read().strip()
            if not content:
                return {}  # empty file case
            return json.loads(content)
        except json.JSONDecodeError:
            st.error("⚠️ Could not parse collab_config.json. The file may be corrupted.")
            return {}  # fallback to empty config


def save_configs(configs):
    os.makedirs("configs", exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(configs, f, indent=2)

def collaboration_tab():
    st.title("🔗 App & Tool Integrations")

    configs = load_configs()

    st.subheader("📥 Connect New Tool")
    tool = st.selectbox("Choose an app or tool", list(AVAILABLE_TOOLS.keys()))

    with st.form(f"form_{tool}"):
        tool_config = {}
        for field in AVAILABLE_TOOLS[tool]:
            tool_config[field] = st.text_input(field)

        submitted = st.form_submit_button("🔐 Save Connection")
        if submitted:
            configs[tool] = tool_config
            save_configs(configs)
            st.success(f"{tool} connected successfully.")

    st.divider()

    st.subheader("✅ Connected Tools")
    if not configs:
        st.info("No tools connected yet.")
    else:
        for k, v in configs.items():
            st.markdown(f"**{k}**")
            for key in v:
                st.markdown(f"• `{key}`: {v[key][:10]}...")

            if st.button(f"❌ Disconnect {k}", key=f"disconnect_{k}"):
                del configs[k]
                save_configs(configs)
                st.success(f"{k} disconnected.")