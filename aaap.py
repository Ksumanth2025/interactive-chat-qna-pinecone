import streamlit as st
import streamlit.components.v1 as components

st.title("🧑‍💻 D-ID Web Agent Avatar")

# Embed D-ID Web Agent
html_code = """
<div id="did-container"></div>
<script
    type="module"
    src="https://agent.d-id.com/v1/index.js"
    data-name="did-agent"
    data-mode="fabio"
    data-client-key="YXV0aDB8NjgyYzc1ZGU4M2Y5YWU0YjM3YWJiNGNkOlVOWDF5UnZ3Mk9hZVg2bjJBbDViZw=="
    data-agent-id="agt_gsuYM9f5"
    data-monitor="true">
</script>
"""

components.html(html_code, height=600)
