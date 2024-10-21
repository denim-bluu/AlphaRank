import streamlit as st


def update_preprocessor_steps():
    st.session_state.preprocessor_steps = st.session_state.preprocessor_steps_select


def update_selected_metrics():
    st.session_state.selected_metrics = st.session_state.metrics_select
    # Reset weights for unselected metrics
    st.session_state.metric_weights = {
        metric: st.session_state.metric_weights.get(metric, 0.5)
        for metric in st.session_state.selected_metrics
    }


def render():
    st.header("Configuration")

    # Preprocessor steps
    st.subheader("Preprocessor Steps")
    all_steps = [
        "SortStep",
        "RollingMeanReturnStep",
        "RollingMeanBenchmarkReturnStep",
        "RollingStdReturnStep",
        "RollingStdBenchmarkReturnStep",
    ]
    st.multiselect(
        "Select preprocessor steps",
        all_steps,
        default=st.session_state.preprocessor_steps,
        key="preprocessor_steps_select",
        on_change=update_preprocessor_steps,
    )

    # Metrics selection and weights
    st.subheader("Metrics")
    all_metrics = [
        "Excess_Return",
        "Beta",
        "Volatility",
        "Sharpe_Ratio",
        "Information_Ratio",
    ]
    st.multiselect(
        "Select metrics",
        all_metrics,
        default=st.session_state.selected_metrics,
        key="metrics_select",
        on_change=update_selected_metrics,
    )

    for metric in st.session_state.selected_metrics:
        st.session_state.metric_weights[metric] = st.slider(
            f"Weight for {metric}",
            0.0,
            1.0,
            st.session_state.metric_weights.get(metric, 0.5),
            step=0.1,
            key=f"weight_{metric}",
        )

    # Risk-free rate
    st.session_state.risk_free_rate = st.number_input(
        "Risk-free rate",
        value=st.session_state.risk_free_rate,
        step=0.001,
        format="%.3f",
        key="risk_free_rate_input",
    )

    # Standardizer
    st.session_state.standardizer = st.selectbox(
        "Select standardizer",
        ["zscore", "minmax"],
        index=["zscore", "minmax"].index(st.session_state.standardizer),
        key="standardizer_select",
    )
