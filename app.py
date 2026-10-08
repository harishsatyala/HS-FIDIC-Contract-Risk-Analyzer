import io
import re

import pandas as pd
import streamlit as st
import plotly.express as px

from pdf_processing.extractor import extract_text_from_pdf
from services.risk_selector import select_top_risks
from services.knowledge_base import knowledge_base
from models.predictor import (
    predict_all_models,
    predict_hybrid
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="HS Contract Risk Analyzer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_OPTIONS = {
    "Logistic Regression": "Logistic Regression",
    "Support Vector Machine": "SVM",
    "Random Forest": "Random Forest",
    "Naive Bayes": "Naive Bayes",
    "Decision Tree": "Decision Tree",
    "XGBoost": "XGBoost",
    "LightGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "MLP / DNN": "MLP"
}

MODEL_LABELS = list(MODEL_OPTIONS.keys())

HYBRID_OPTIONS = [
    "Majority Vote",
    "Confidence Weighted Hybrid",
    "Rank Weighted Hybrid",
]

HYBRID_COMPONENTS = (
    "Support Vector Machine + "
    "Logistic Regression + "
    "Naive Bayes"
)

HYBRID_METHOD_DESCRIPTIONS = {
    "Majority Vote":
        "Equal-weight majority voting",

    "Confidence Weighted Hybrid":
        "Confidence-weighted voting",

    "Rank Weighted Hybrid":
        "Rank-weighted voting using 3-2-1 model weights",
}


# ============================================================
# PRIMARY HYBRID MODEL
# ============================================================

# The detailed risk-selection pipeline uses the
# confidence-weighted hybrid consisting of:
#
#   - Support Vector Machine
#   - Logistic Regression
#   - Naive Bayes
#
# The nine individual models are still executed automatically
# above and remain available for comparison.
#
# Because the real contracts do not have sentence-level
# ground-truth labels, the hybrid is selected as the
# research-prototype integration model based on model
# complementarity and unlabeled real-contract diagnostics.
# ============================================================

PRIMARY_MODEL_LABEL = "Confidence Weighted Hybrid"
PRIMARY_MODEL = PRIMARY_MODEL_LABEL


# ============================================================
# CACHED PDF EXTRACTION
# ============================================================

@st.cache_data(show_spinner=False)
def extract_contract_text(pdf_bytes):
    """
    Extract contract text once per uploaded PDF.
    """

    pdf_file = io.BytesIO(pdf_bytes)

    return extract_text_from_pdf(
        pdf_file
    )


# ============================================================
# CACHED SENTENCE SPLITTING
# ============================================================

@st.cache_data(show_spinner=False)
def split_contract_sentences(extracted_text):
    """
    Split extracted contract text into sentences.
    """

    sentences = re.split(
        r"(?<=[.!?])\s+",
        extracted_text
    )

    return tuple(
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    )


# ============================================================
# CACHED SINGLE-MODEL RISK ANALYSIS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=100
)
def analyze_contract(
    sentences,
    selected_model
):
    """
    Run the detailed risk-selection pipeline for one
    selected ML model.

    The selected model can be an individual ML model
    or the Confidence Weighted Hybrid.
    """

    return select_top_risks(
        list(sentences),
        top_k=20,
        selected_model=selected_model
    )


# ============================================================
# AUTOMATIC BATCH MODEL PREDICTIONS
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=20
)
def run_all_model_predictions(sentences):
    """
    Run all nine ML models using one shared preprocessing pass.

    The predictor preprocesses the contract sentences once,
    then reuses the cleaned text for all nine category models
    and the responsible-party model.

    The expensive knowledge-base risk retrieval is still
    performed only once later using the primary hybrid model.
    """

    return predict_all_models(
        list(sentences),
        MODEL_OPTIONS
    )


# ============================================================
# MODEL PREDICTION COMPARISON
# ============================================================

def build_model_comparison_summary(
    all_model_predictions
):
    """
    Build a summary showing how each ML model classified
    the uploaded contract.

    This is based on the actual predictions produced for
    the uploaded contract, not benchmark results.
    """

    summary = []

    for model_label in MODEL_LABELS:

        predictions = all_model_predictions.get(
            model_label,
            []
        )

        if not predictions:
            continue

        categories = [
            result.get(
                "category",
                "Not Available"
            )
            for result in predictions
        ]

        confidences = [
            float(
                result.get(
                    "category_confidence",
                    0.0
                )
            )
            for result in predictions
        ]

        category_series = (
            pd.Series(categories)
            .value_counts()
        )

        if len(category_series) > 0:

            dominant_category = (
                category_series.index[0]
            )

            dominant_count = (
                category_series.iloc[0]
            )

        else:

            dominant_category = "Not Available"
            dominant_count = 0

        summary.append(
            {
                "Model": model_label,

                "Sentences Classified": len(
                    predictions
                ),

                "Most Common Category":
                    dominant_category,

                "Most Common Category Count":
                    dominant_count,

                "Average Category Confidence (%)":
                    round(
                        sum(confidences)
                        / len(confidences),
                        2
                    )
            }
        )

    return pd.DataFrame(
        summary
    )


def build_model_category_distribution(
    predictions
):
    """
    Return category counts for one model.
    """

    categories = [
        result.get(
            "category",
            "Not Available"
        )
        for result in predictions
    ]

    return (
        pd.Series(categories)
        .value_counts()
    )


def calculate_model_agreement(
    all_model_predictions,
    sentences
):
    """
    Calculate the percentage of contract sentences for
    which all available individual models predict the
    same category.
    """

    if not sentences:
        return 0.0

    prediction_lists = []

    for model_label in MODEL_LABELS:

        predictions = all_model_predictions.get(
            model_label,
            []
        )

        if len(predictions) != len(sentences):
            continue

        prediction_lists.append(
            [
                result.get(
                    "category",
                    ""
                )
                for result in predictions
            ]
        )

    if not prediction_lists:
        return 0.0

    agreement_count = 0

    for index in range(
        len(sentences)
    ):

        sentence_predictions = [
            predictions[index]
            for predictions in prediction_lists
        ]

        if len(
            set(sentence_predictions)
        ) == 1:

            agreement_count += 1

    return round(
        (
            agreement_count
            / len(sentences)
        ) * 100,
        2
    )


# ============================================================
# CACHED COMPLETE RISK LOOKUP
# ============================================================

@st.cache_data(
    show_spinner=False,
    max_entries=500
)
def get_complete_risk_information(risk_id):
    """
    Retrieve complete qualitative and quantitative
    information for a Risk ID.
    """

    try:

        complete = knowledge_base.get_complete_risk(
            risk_id
        )

        if complete:

            return dict(
                complete
            )

    except Exception:

        pass

    try:

        quantitative = (
            knowledge_base.get_quantitative_risk(
                risk_id
            )
        )

        if quantitative:

            return dict(
                quantitative
            )

    except Exception:

        pass

    return {}


# ============================================================
# HEADER
# ============================================================

st.title(
    "📄 HS Contract Risk Analyzer"
)

st.write(
    "Upload one or more contract PDFs to identify and prioritize "
    "the most relevant contractual risks using machine "
    "learning and a structured risk knowledge base."
)

st.info(
    "All nine individual machine-learning models and all "
    "three hybrid models are executed automatically after "
    "the contract PDF is uploaded."
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_files = st.file_uploader(
    "Upload multiple contract PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF contracts. Each contract is analyzed separately."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_files:
    st.caption(f"{len(uploaded_files)} contract(s) selected. Results are reported separately for each file.")
    for contract_index, uploaded_file in enumerate(uploaded_files, start=1):
        st.divider()
        st.header(f"Contract {contract_index}: {uploaded_file.name}")

        # ========================================================
        # READ PDF
        # ========================================================

        pdf_bytes = uploaded_file.getvalue()

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )


        # ========================================================
        # PDF EXTRACTION
        # ========================================================

        with st.spinner(
            "Extracting contract text..."
        ):

            extracted_text = extract_contract_text(
                pdf_bytes
            )

        if not extracted_text.strip():

            st.error(
                "No readable text was found in the PDF."
            )

            continue


        # ========================================================
        # SENTENCE SPLITTING
        # ========================================================

        sentences = split_contract_sentences(
            extracted_text
        )

        st.subheader(
            f"📑 Contract Sentences: {len(sentences)}"
        )


        # ========================================================
        # AUTOMATIC BATCH ML ANALYSIS — ALL 9 MODELS + HYBRID
        # ========================================================

        with st.spinner(
            "Automatically running all 9 ML models and all 3 hybrids..."
        ):

            all_model_predictions = run_all_model_predictions(
                sentences
            )

            hybrid_predictions = {
        hybrid_name:
            predict_hybrid(
                list(sentences),
                hybrid_name
            )
        for hybrid_name in HYBRID_OPTIONS
    }


        # ========================================================
        # MODEL EXECUTION SUMMARY
        # ========================================================

        st.subheader(
            "🤖 Automatic Model Execution"
        )

        model_status = []

        for model_label in MODEL_LABELS:

            predictions = all_model_predictions.get(
                model_label,
                []
            )

            model_status.append(
                {
                    "Model": model_label,

                    "Sentences Classified": len(
                        predictions
                    ),

                    "Status": (
                        "Completed"
                        if len(predictions) == len(sentences)
                        else "Incomplete"
                    )
                }
            )

        model_status_df = pd.DataFrame(
            model_status
        )

        st.dataframe(
            model_status_df,
            use_container_width=True,
            hide_index=True
        )


        # ========================================================
        # HYBRID MODEL STATUS
        # ========================================================

        hybrid_status = []

        for hybrid_name in HYBRID_OPTIONS:

            predictions = hybrid_predictions.get(
                hybrid_name,
                []
                )


            hybrid_status.append(
                {
                    "Hybrid Model": hybrid_name,

                    "Base Models":
                    HYBRID_COMPONENTS,

                    "Method":
                    HYBRID_METHOD_DESCRIPTIONS[
                        hybrid_name
                        ],

                    "Sentences Classified":
                    len(predictions),

                    "Status": (
                        "Completed"
                        if len(predictions) == len(sentences)
                        else "Incomplete"
                        ),
                        }
                        )

            hybrid_status_df = pd.DataFrame(
                hybrid_status
                )

        st.dataframe(
            hybrid_status_df,
            use_container_width=True,
            hide_index=True
            )

        st.caption(
            "All nine individual ML models and all three hybrid "
            "approaches automatically classify the contract "
            "sentences in batch. The three hybrids combine "
            "Support Vector Machine, Logistic Regression, and "
            "Naive Bayes using different voting strategies. "
            "The Confidence Weighted Hybrid is currently used "
            "for detailed contractual risk prioritization."
            )


        # ========================================================
        # HYBRID MODEL DETAILS
        # ========================================================

        st.subheader(
            "🔀 Hybrid Model Details — 3 Hybrid Models"
        )

        st.write(
            "The system evaluates three hybrid approaches using "
            "the same three component models: "
            "**Support Vector Machine + Logistic Regression + "
            "Naive Bayes**. All three are executed automatically "
            "and are shown separately below."
        )

        hybrid_tabs = st.tabs(
            HYBRID_OPTIONS
        )

        for tab, hybrid_name in zip(
            hybrid_tabs,
            HYBRID_OPTIONS
        ):

            with tab:

                predictions = hybrid_predictions.get(
                    hybrid_name,
                    []
                )

                st.markdown(
                    f"### {hybrid_name}"
                )

                st.write(
                    f"**Base models:** {HYBRID_COMPONENTS}"
                )

                st.write(
                    f"**Combination method:** "
                    f"{HYBRID_METHOD_DESCRIPTIONS[hybrid_name]}"
                )

                st.caption(
                    "Decision strength represents the strength of "
                    "the hybrid voting decision. It is not calibrated "
                    "probability, accuracy, precision, recall, or F1."
                )

                if not predictions:

                    st.warning(
                        f"{hybrid_name} did not return predictions."
                    )

                    continue

                categories = [
                    result.get(
                        "category",
                        "Not Available"
                    )
                    for result in predictions
                ]

                distribution = (
                    pd.Series(categories)
                    .value_counts()
                )

                average_strength = round(
                    sum(
                        float(
                            result.get(
                                "category_confidence",
                                0.0
                            )
                        )
                        for result in predictions
                    )
                    / len(predictions),
                    2
                )

                dominant_category = (
                    distribution.index[0]
                    if not distribution.empty
                    else "Not Available"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Sentences Classified",
                        len(predictions)
                    )

                with col2:
                    st.metric(
                        "Most Common Category",
                        dominant_category
                    )

                with col3:
                    st.metric(
                        "Average Decision Strength",
                        f"{average_strength:.2f}%"
                    )

                st.write(
                    "Hybrid Predicted Category Distribution"
                )

                st.bar_chart(
                    distribution
                )

                hybrid_prediction_rows = []

                for result in predictions:

                    hybrid_prediction_rows.append(
                        {
                            "Contract Sentence":
                                result.get(
                                    "sentence",
                                    ""
                                ),

                            "Hybrid Predicted Category":
                                result.get(
                                    "category",
                                    ""
                                ),

                            "Decision Strength (%)":
                                result.get(
                                    "category_confidence",
                                    0.0
                                ),

                            "Responsible Party":
                                result.get(
                                    "responsible_party",
                                    ""
                                ),

                            "Party Confidence (%)":
                                result.get(
                                    "party_confidence",
                                    0.0
                                ),
                        }
                    )

                hybrid_prediction_df = pd.DataFrame(
                    hybrid_prediction_rows
                )

                st.write(
                    "Hybrid Sentence-Level Predictions"
                )

                st.dataframe(
                    hybrid_prediction_df,
                    use_container_width=True,
                    height=450,
                    hide_index=True
                )




        # ========================================================
        # MODEL PREDICTION COMPARISON
        # ========================================================

        st.subheader(
            "📊 Model Prediction Comparison"
        )

        st.write(
            "The following results show the actual category "
            "predictions produced by each model for the uploaded "
            "contract. These are contract-specific predictions, "
            "not cross-validation scores."
        )


        # --------------------------------------------------------
        # MODEL SUMMARY
        # --------------------------------------------------------

        model_comparison_df = (
            build_model_comparison_summary(
                all_model_predictions
            )
        )

        if not model_comparison_df.empty:

            st.dataframe(
                model_comparison_df,
                use_container_width=True,
                hide_index=True
            )

            # --------------------------------------------------------
        # HYBRID COMPARISON
        # --------------------------------------------------------

        st.write(
            "### Hybrid Model Comparison"
        )

        st.write(
            "Each hybrid combines the same three component models: "
            "**Support Vector Machine + Logistic Regression + "
            "Naive Bayes**. The hybrids differ only in their "
            "prediction-combination strategy."
        )

        hybrid_comparison_rows = []

        for hybrid_name in HYBRID_OPTIONS:

            predictions = hybrid_predictions.get(
                hybrid_name,
                []
            )

            if not predictions:
                continue

            categories = [
                result.get(
                    "category",
                    "Not Available"
                )
                for result in predictions
            ]

            strengths = [
                float(
                    result.get(
                        "category_confidence",
                        0.0
                    )
                )
                for result in predictions
            ]

            category_series = (
                pd.Series(categories)
                .value_counts()
            )

            dominant_category = (
                category_series.index[0]
                if not category_series.empty
                else "Not Available"
            )

            dominant_count = (
                int(category_series.iloc[0])
                if not category_series.empty
                else 0
            )

            hybrid_comparison_rows.append(
                {
                    "Hybrid Model":
                        hybrid_name,

                    "Type":
                        "Hybrid",

                    "Base Models":
                        HYBRID_COMPONENTS,

                    "Combination Method":
                        HYBRID_METHOD_DESCRIPTIONS[
                            hybrid_name
                        ],

                    "Sentences Classified":
                        len(predictions),

                    "Most Common Category":
                        dominant_category,

                    "Most Common Category Count":
                        dominant_count,

                    "Average Decision Strength (%)":
                        round(
                            sum(strengths)
                            / len(strengths),
                            2
                        ),
                }
            )

        hybrid_comparison_df = pd.DataFrame(
            hybrid_comparison_rows
        )

        st.dataframe(
            hybrid_comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------------
        # INDIVIDUAL VS HYBRID COMPARISON
        # --------------------------------------------------------

        st.write(
            "### Individual Models vs Hybrid Models"
        )

        st.write(
            "All nine individual models and all three hybrids are "
            "shown together below. Each hybrid explicitly lists "
            "its three component models and combination strategy."
        )

        individual_comparison_df = model_comparison_df.copy()

        individual_comparison_df["Type"] = "Individual Model"

        individual_comparison_df["Base Models"] = (
            individual_comparison_df["Model"]
        )

        individual_comparison_df["Combination Method"] = (
            "Single model"
        )

        individual_comparison_df = (
            individual_comparison_df.rename(
                columns={"Model": "Model / Hybrid"}
            )
        )

        individual_comparison_df = individual_comparison_df[
            [
                "Model / Hybrid",
                "Type",
                "Base Models",
                "Combination Method",
                "Sentences Classified",
                "Most Common Category",
                "Most Common Category Count",
                "Average Category Confidence (%)"
            ]
        ]

        hybrid_combined_df = hybrid_comparison_df.rename(
            columns={
                "Hybrid Model": "Model / Hybrid",
                "Average Decision Strength (%)":
                    "Average Category Confidence (%)"
            }
        )

        hybrid_combined_df = hybrid_combined_df[
            [
                "Model / Hybrid",
                "Type",
                "Base Models",
                "Combination Method",
                "Sentences Classified",
                "Most Common Category",
                "Most Common Category Count",
                "Average Category Confidence (%)"
            ]
        ]

        combined_comparison_df = pd.concat(
            [
                individual_comparison_df,
                hybrid_combined_df
            ],
            ignore_index=True
        )

        st.dataframe(
            combined_comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------------
        # MODEL AGREEMENT
        # --------------------------------------------------------

        model_agreement = (
            calculate_model_agreement(
                all_model_predictions,
                sentences
            )
        )

        st.metric(
            "All-Model Sentence Agreement",
            f"{model_agreement:.2f}%"
        )

        st.caption(
            "This percentage represents the share of contract "
            "sentences for which all available individual models "
            "predicted the same risk category. It is an agreement "
            "measure, not an accuracy measure."
        )


        # --------------------------------------------------------
        # INDIVIDUAL MODEL TABS
        # --------------------------------------------------------

        model_tabs = st.tabs(
            MODEL_LABELS
        )

        for tab, model_label in zip(
            model_tabs,
            MODEL_LABELS
        ):

            with tab:

                predictions = (
                    all_model_predictions.get(
                        model_label,
                        []
                    )
                )

                if not predictions:

                    st.warning(
                        f"No predictions were returned by "
                        f"{model_label}."
                    )

                    continue


                # ------------------------------------------------
                # MODEL SUMMARY
                # ------------------------------------------------

                category_distribution = (
                    build_model_category_distribution(
                        predictions
                    )
                )

                average_confidence = round(
                    sum(
                        float(
                            result.get(
                                "category_confidence",
                                0.0
                            )
                        )
                        for result in predictions
                    )
                    / len(predictions),
                    2
                )

                dominant_category = (
                    category_distribution.index[0]
                    if not category_distribution.empty
                    else "Not Available"
                )

                metric_col1, metric_col2, metric_col3 = (
                    st.columns(3)
                )

                with metric_col1:

                    st.metric(
                        "Sentences Classified",
                        len(predictions)
                    )

                with metric_col2:

                    st.metric(
                        "Most Common Category",
                        dominant_category
                    )

                with metric_col3:

                    st.metric(
                        "Average Category Confidence",
                        f"{average_confidence:.2f}%"
                    )


                # ------------------------------------------------
                # CATEGORY DISTRIBUTION
                # ------------------------------------------------

                st.write(
                    "Predicted Category Distribution"
                )

                st.bar_chart(
                    category_distribution
                )


                # ------------------------------------------------
                # SENTENCE-LEVEL PREDICTIONS
                # ------------------------------------------------

                model_prediction_rows = []

                for result in predictions:

                    model_prediction_rows.append(
                        {
                            "Contract Sentence":
                                result.get(
                                    "sentence",
                                    ""
                                ),

                            "Predicted Category":
                                result.get(
                                    "category",
                                    ""
                                ),

                            "Category Confidence (%)":
                                result.get(
                                    "category_confidence",
                                    0.0
                                ),

                            "Responsible Party":
                                result.get(
                                    "responsible_party",
                                    ""
                                ),

                            "Party Confidence (%)":
                                result.get(
                                    "party_confidence",
                                    0.0
                                )
                        }
                    )


                model_prediction_df = pd.DataFrame(
                    model_prediction_rows
                )


                st.write(
                    "Sentence-Level Predictions"
                )

                st.dataframe(
                    model_prediction_df,
                    use_container_width=True,
                    height=450,
                    hide_index=True
                )


            # ========================================================
        # QUANTITATIVE MODEL EVALUATION
        # ========================================================

        st.subheader("📊 Quantitative Model Evaluation")

        st.write(
            "Supervised evaluation metrics are shown separately by "
            "evaluation source. Internal benchmark results use the "
            "labelled training dataset, proxy external results use "
            "the research-defined reference labels, and real-contract "
            "predictions are kept separate because those contracts "
            "do not have sentence-level ground-truth labels."
        )

        metrics_path = "output/final_model_metrics.csv"

        try:
            metrics_df = pd.read_csv(metrics_path)

            # Keep the evaluation order explicit and consistent.
            evaluation_order = [
                "Internal benchmark",
                "Proxy external evaluation",
            ]

            metric_columns = [
                "Evaluation Type",
                "Model",
                "Accuracy",
                "Precision",
                "Recall",
                "F1",
                "Macro-F1",
                "Weighted-F1",
                "ROC-AUC",
                "Test Samples",
            ]

            available_metric_columns = [
                column
                for column in metric_columns
                if column in metrics_df.columns
            ]

            # --------------------------------------------------------
            # Internal benchmark
            # --------------------------------------------------------

            st.write("### Internal Benchmark")

            internal_metrics = metrics_df[
                metrics_df["Evaluation Type"] == "Internal benchmark"
            ].copy()

            if not internal_metrics.empty:

                st.caption(
                    "The internal benchmark contains 200 rows but only "
                    "5 unique risk statements (97.50% duplicate rate). "
                    "Its scores are therefore reported as benchmark "
                    "results and should not be interpreted as evidence "
                    "of generalization to unseen contracts."
                )

                internal_display = internal_metrics[
                    available_metric_columns
                ].copy()

                # Display metric values as percentages.
                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "Macro-F1",
                    "Weighted-F1",
                    "ROC-AUC",
                ]:
                    if column in internal_display.columns:
                        internal_display[column] = (
                            pd.to_numeric(
                                internal_display[column],
                                errors="coerce"
                            ) * 100
                        ).round(2)

                st.dataframe(
                    internal_display,
                    use_container_width=True,
                    hide_index=True,
                )

            # --------------------------------------------------------
            # Proxy external evaluation
            # --------------------------------------------------------

            st.write("### Proxy External Evaluation")

            proxy_metrics = metrics_df[
                metrics_df["Evaluation Type"]
                == "Proxy external evaluation"
            ].copy()

            if not proxy_metrics.empty:

                st.caption(
                    "These metrics are calculated against the defined "
                    "proxy/reference labels for the 3,000-row external "
                    "risk library. They are not expert-annotated "
                    "ground-truth measurements."
                )

                proxy_display = proxy_metrics[
                    available_metric_columns
                ].copy()

                for column in [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "Macro-F1",
                    "Weighted-F1",
                    "ROC-AUC",
                ]:
                    if column in proxy_display.columns:
                        proxy_display[column] = (
                            pd.to_numeric(
                                proxy_display[column],
                                errors="coerce"
                            ) * 100
                        ).round(2)

                st.dataframe(
                    proxy_display,
                    use_container_width=True,
                    hide_index=True,
                )

            # --------------------------------------------------------
            # Hybrid improvement analysis
            # --------------------------------------------------------

            st.write("### Hybrid Improvement Analysis")

            st.write(
                "This comparison tests whether each hybrid improves over "
                "its three component models. Macro-F1 is the primary "
                "comparison because it gives each risk class equal weight. "
                "The proxy external evaluation is used as the main comparison "
                "because the internal benchmark contains only five unique "
                "statements repeated across 200 rows."
            )

            hybrid_names = [
                "Majority Vote",
                "Confidence Weighted Hybrid",
                "Rank Weighted Hybrid",
            ]

            component_names = [
                "Support Vector Machine",
                "Logistic Regression",
                "Naive Bayes",
            ]

            proxy_all = metrics_df[
                metrics_df["Evaluation Type"] == "Proxy external evaluation"
            ].copy()

            if not proxy_all.empty:

                proxy_lookup = proxy_all.set_index("Model")
                hybrid_analysis_rows = []
                component_rows = proxy_all[
                    proxy_all["Model"].isin(component_names)
                ].copy()

                if not component_rows.empty:

                    component_macro = pd.to_numeric(
                        component_rows["Macro-F1"],
                        errors="coerce"
                    )

                    component_accuracy = pd.to_numeric(
                        component_rows["Accuracy"],
                        errors="coerce"
                    )

                    component_weighted = pd.to_numeric(
                        component_rows["Weighted-F1"],
                        errors="coerce"
                    )

                    best_macro = component_macro.max()
                    best_accuracy = component_accuracy.max()
                    best_weighted = component_weighted.max()

                    best_macro_models = component_rows.loc[
                        component_macro == best_macro,
                        "Model"
                    ].tolist()

                    component_macro_text = "; ".join(
                        f"{row['Model']}: {float(row['Macro-F1']) * 100:.2f}%"
                        for _, row in component_rows.iterrows()
                    )

                    for hybrid_name in hybrid_names:

                        if hybrid_name not in proxy_lookup.index:
                            continue

                        hybrid_macro = float(
                            pd.to_numeric(
                                proxy_lookup.loc[hybrid_name, "Macro-F1"],
                                errors="coerce"
                            )
                        )

                        hybrid_accuracy = float(
                            pd.to_numeric(
                                proxy_lookup.loc[hybrid_name, "Accuracy"],
                                errors="coerce"
                            )
                        )

                        hybrid_weighted = float(
                            pd.to_numeric(
                                proxy_lookup.loc[hybrid_name, "Weighted-F1"],
                                errors="coerce"
                            )
                        )

                        macro_delta = hybrid_macro - best_macro
                        accuracy_delta = hybrid_accuracy - best_accuracy
                        weighted_delta = hybrid_weighted - best_weighted

                        if macro_delta > 1e-9:
                            verdict = "Improves Macro-F1"
                        elif macro_delta < -1e-9:
                            verdict = "Does not improve Macro-F1"
                        else:
                            verdict = "Matches best component Macro-F1"

                        hybrid_analysis_rows.append(
                            {
                                "Hybrid Model": hybrid_name,
                                "Component Macro-F1": component_macro_text,
                                "Best Component": ", ".join(best_macro_models),
                                "Hybrid Macro-F1 (%)": round(hybrid_macro * 100, 2),
                                "Best Component Macro-F1 (%)": round(best_macro * 100, 2),
                                "Δ Macro-F1 (pp)": round(macro_delta * 100, 2),
                                "Hybrid Accuracy (%)": round(hybrid_accuracy * 100, 2),
                                "Best Component Accuracy (%)": round(best_accuracy * 100, 2),
                                "Δ Accuracy (pp)": round(accuracy_delta * 100, 2),
                                "Hybrid Weighted-F1 (%)": round(hybrid_weighted * 100, 2),
                                "Best Component Weighted-F1 (%)": round(best_weighted * 100, 2),
                                "Δ Weighted-F1 (pp)": round(weighted_delta * 100, 2),
                                "Verdict": verdict,
                            }
                        )

                    hybrid_analysis_df = pd.DataFrame(
                        hybrid_analysis_rows
                    )

                    if not hybrid_analysis_df.empty:

                        st.dataframe(
                            hybrid_analysis_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.caption(
                            "Δ values are hybrid minus the best-performing component "
                            "for that metric, expressed in percentage points. Positive "
                            "values indicate improvement, zero indicates a tie, and "
                            "negative values indicate that the hybrid did not improve "
                            "over the best component."
                        )

                        component_reference_df = component_rows[
                            ["Model", "Accuracy", "Macro-F1", "Weighted-F1"]
                        ].copy()

                        for column in [
                            "Accuracy",
                            "Macro-F1",
                            "Weighted-F1",
                        ]:
                            component_reference_df[column] = (
                                pd.to_numeric(
                                    component_reference_df[column],
                                    errors="coerce"
                                ) * 100
                            ).round(2)

                        st.write(
                            "**Component Model Reference — Proxy External Evaluation**"
                        )

                        st.dataframe(
                            component_reference_df,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.info(
                            "The hybrid methods are not treated as automatically "
                            "superior. The research conclusion is based on the measured "
                            "proxy metrics: a hybrid is considered an improvement only "
                            "when its metric exceeds the corresponding best component "
                            "metric. If it does not, the result is reported honestly."
                        )

                else:

                    st.warning(
                        "The SVM, Logistic Regression, and Naive Bayes component "
                        "metrics are not available, so the hybrid comparison cannot "
                        "be calculated."
                    )

            else:

                st.warning(
                    "Proxy external metrics are not available, so the hybrid "
                    "improvement comparison cannot be calculated."
                )

            # --------------------------------------------------------
            # Metric definitions
            # --------------------------------------------------------

            with st.expander("ℹ️ Metric definitions"):

                st.markdown(
                    """
                    **Accuracy** — proportion of predictions that are correct.

                    **Precision** — proportion of predicted classes that are correct.

                    **Recall** — proportion of actual reference classes that are detected.

                    **F1** — harmonic mean of precision and recall.

                    **Macro-F1** — mean F1 across classes, giving each class equal weight.

                    **Weighted-F1** — F1 averaged using each class's support.

                    **ROC-AUC** — area under the one-vs-rest ROC curve where the
                    required probability scores are available.

                    **Test Samples** — number of samples evaluated for that row.
                    """
                )

            st.caption(
                "For this application, metric tables are evaluation evidence; "
                "the confidence/decision-strength values shown for individual "
                "and hybrid predictions are separate quantities and must not "
                "be interpreted as Accuracy, Precision, Recall, F1, or ROC-AUC."
            )

        except FileNotFoundError:

            st.warning(
                "Quantitative metrics are not available yet. "
                "Run models\\final_metrics_evaluation.py first."
            )

        except Exception as exc:

            st.error(
                f"Could not load the quantitative metrics table: {exc}"
            )


        # ========================================================
        # PRIMARY RISK ANALYSIS
        # ========================================================

        with st.spinner(
            "Selecting and prioritizing contractual risks..."
        ):

            results = analyze_contract(
                sentences,
                PRIMARY_MODEL
            )


        if not results:

            st.warning(
                f"The primary {PRIMARY_MODEL_LABEL} model "
                "did not identify sufficiently relevant risks "
                "in this contract."
            )

            continue


        results_df = pd.DataFrame(
            results
        )


        # ========================================================
        # COMPLETE RISK INFORMATION
        # ========================================================

        for index, row in results_df.iterrows():

            risk_id = row.get(
                "Risk ID"
            )

            if not risk_id:
                continue

            complete_risk = (
                get_complete_risk_information(
                    risk_id
                )
            )

            if not complete_risk:
                continue

            for column, value in complete_risk.items():

                if (
                    column not in results_df.columns
                    or
                    pd.isna(
                        results_df.loc[
                            index,
                            column
                        ]
                    )
                ):

                    results_df.loc[
                        index,
                        column
                    ] = value


        # ========================================================
        # NORMALIZE COLUMN NAMES
        # ========================================================

        # --------------------------------------------------------
        # Probability
        # --------------------------------------------------------

        if "Probability" not in results_df.columns:

            if "Probability (L/M/H)" in results_df.columns:

                results_df["Probability"] = (
                    results_df[
                        "Probability (L/M/H)"
                    ]
                )

            else:

                results_df["Probability"] = None


        # --------------------------------------------------------
        # Impact
        # --------------------------------------------------------

        if "Impact" not in results_df.columns:

            if (
                "Impact (Cost / Time / Quality / HSE)"
                in results_df.columns
            ):

                results_df["Impact"] = (
                    results_df[
                        "Impact (Cost / Time / Quality / HSE)"
                    ]
                )

            else:

                results_df["Impact"] = None


        # ========================================================
        # NUMERIC CONVERSION
        # ========================================================

        numeric_columns = [

            "Gross Exposure (%)",

            "Gross Exposure (EUR)",

            "Probability Factor",

            "Expected Monetary Value (EUR)",

            "Mitigation Effectiveness",

            "Residual Exposure (EUR)",

            "Residual Exposure (% TCV)",

            "Risk Relevance Score"
        ]

        for column in numeric_columns:

            if column in results_df.columns:

                results_df[column] = pd.to_numeric(
                    results_df[column],
                    errors="coerce"
                )


        # ========================================================
        # CONFIDENCE VALUES
        # ========================================================

        for column in [

            "Category Confidence (%)",

            "Party Confidence (%)"

        ]:

            if column in results_df.columns:

                results_df[column] = (
                    pd.to_numeric(
                        results_df[column],
                        errors="coerce"
                    )
                    .clip(
                        lower=0,
                        upper=100
                    )
                    .round(2)
                )


        # ========================================================
        # MITIGATION EFFECTIVENESS → PERCENTAGE
        # ========================================================

        if "Mitigation Effectiveness" in results_df.columns:

            results_df[
                "Mitigation Effectiveness"
            ] = (
                pd.to_numeric(
                    results_df[
                        "Mitigation Effectiveness"
                    ],
                    errors="coerce"
                ) * 100
            ).round(1)


        # ========================================================
        # RANKING
        # ========================================================

        if "Risk Relevance Score" in results_df.columns:

            results_df = results_df.sort_values(
                by="Risk Relevance Score",
                ascending=False,
                na_position="last"
            ).reset_index(
                drop=True
            )


        # ========================================================
        # SUMMARY
        # ========================================================

        st.subheader(
            "📊 Risk Analysis Summary"
        )

        col1, col2 = st.columns(2)


        # --------------------------------------------------------
        # Risks
        # --------------------------------------------------------

        with col1:

            st.metric(
                "Risks Identified",
                len(results_df)
            )


        # --------------------------------------------------------
        # Categories
        # --------------------------------------------------------

        with col2:

            category_count = (
                results_df[
                    "Risk Category"
                ]
                .dropna()
                .nunique()
            )

            st.metric(
                "Risk Categories",
                category_count
            )


        st.caption(
            "All nine individual ML models and all three hybrid "
            "models are automatically executed for comparison. "
            "The Confidence Weighted Hybrid combines Support "
            "Vector Machine, Logistic Regression, and Naive Bayes "
            "predictions and is currently used for detailed "
            "contractual risk retrieval and prioritization. "
            "The hybrid integration choice is based on model "
            "complementarity and diagnostics from the available "
            "unlabeled real-contract evaluation."
        )


        # ========================================================
        # RISK DISTRIBUTION
        # ========================================================

        st.subheader(
            "📈 Risk Distribution"
        )

        chart_col1, chart_col2 = st.columns(2)


        # --------------------------------------------------------
        # CATEGORY
        # --------------------------------------------------------

        with chart_col1:

            st.write(
                "Risk Category"
            )

            category_chart_df = (
                results_df[
                    ["Risk ID", "Risk Category", "Risk Relevance Score", "Risk Statement"]
                ]
                .copy()
            )

            category_chart_df["Risk Category"] = (
                category_chart_df["Risk Category"].fillna("Not Available")
            )

            category_chart_df["Risk Relevance Score"] = pd.to_numeric(
                category_chart_df["Risk Relevance Score"],
                errors="coerce"
            )

            category_aggregated = (
                category_chart_df
                .groupby("Risk Category", sort=False)
                .agg(
                    Risk_Count=("Risk ID", "count"),
                    Risk_IDs=("Risk ID", lambda x: ", ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                    Relevance_Scores=("Risk Relevance Score", lambda x: ", ".join(
                        f"{v:.4f}" for v in x if pd.notna(v)
                    )),
                    Risk_Statements=("Risk Statement", lambda x: " | ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                )
                .reset_index()
            )

            category_plot = px.bar(
                category_aggregated,
                x="Risk Category",
                y="Risk_Count",
                custom_data=[
                    "Risk Category",
                    "Risk_Count",
                    "Risk_IDs",
                    "Relevance_Scores",
                    "Risk_Statements",
                ],
                labels={
                    "Risk Category": "Risk Category",
                    "Risk_Count": "Number of Risks",
                },
                title="Risk Category Distribution"
            )

            category_plot.update_traces(
                hovertemplate=(
                    "<b>Category:</b> %{customdata[0]}<br>"
                    "<b>Risk Count:</b> %{customdata[1]}<br>"
                    "<b>Risk IDs:</b> %{customdata[2]}<br>"
                    "<b>Relevance Scores:</b> %{customdata[3]}<br>"
                    "<b>Risk Statements:</b> %{customdata[4]}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                category_plot,
                use_container_width=True
            )


        # --------------------------------------------------------
        # PROBABILITY
        # --------------------------------------------------------

        with chart_col2:

            st.write(
                "Probability"
            )

            probability_chart_df = (
                results_df[
                    ["Risk ID", "Probability", "Risk Relevance Score", "Risk Statement"]
                ]
                .copy()
            )

            probability_chart_df["Probability"] = (
                probability_chart_df["Probability"].fillna("Not Available")
            )

            probability_chart_df["Risk Relevance Score"] = pd.to_numeric(
                probability_chart_df["Risk Relevance Score"],
                errors="coerce"
            )

            probability_aggregated = (
                probability_chart_df
                .groupby("Probability", sort=False)
                .agg(
                    Risk_Count=("Risk ID", "count"),
                    Risk_IDs=("Risk ID", lambda x: ", ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                    Relevance_Scores=("Risk Relevance Score", lambda x: ", ".join(
                        f"{v:.4f}" for v in x if pd.notna(v)
                    )),
                    Risk_Statements=("Risk Statement", lambda x: " | ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                )
                .reset_index()
            )

            probability_plot = px.bar(
                probability_aggregated,
                x="Probability",
                y="Risk_Count",
                custom_data=[
                    "Probability",
                    "Risk_Count",
                    "Risk_IDs",
                    "Relevance_Scores",
                    "Risk_Statements",
                ],
                labels={
                    "Probability": "Probability",
                    "Risk_Count": "Number of Risks",
                },
                title="Probability Distribution"
            )

            probability_plot.update_traces(
                hovertemplate=(
                    "<b>Probability:</b> %{customdata[0]}<br>"
                    "<b>Risk Count:</b> %{customdata[1]}<br>"
                    "<b>Risk IDs:</b> %{customdata[2]}<br>"
                    "<b>Relevance Scores:</b> %{customdata[3]}<br>"
                    "<b>Risk Statements:</b> %{customdata[4]}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                probability_plot,
                use_container_width=True
            )


        chart_col3, chart_col4 = st.columns(2)


        # --------------------------------------------------------
        # IMPACT
        # --------------------------------------------------------

        with chart_col3:

            st.write(
                "Impact"
            )

            impact_chart_df = (
                results_df[
                    ["Risk ID", "Impact", "Risk Relevance Score", "Risk Statement"]
                ]
                .copy()
            )

            impact_chart_df["Impact"] = (
                impact_chart_df["Impact"].fillna("Not Available")
            )

            impact_chart_df["Risk Relevance Score"] = pd.to_numeric(
                impact_chart_df["Risk Relevance Score"],
                errors="coerce"
            )

            impact_aggregated = (
                impact_chart_df
                .groupby("Impact", sort=False)
                .agg(
                    Risk_Count=("Risk ID", "count"),
                    Risk_IDs=("Risk ID", lambda x: ", ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                    Relevance_Scores=("Risk Relevance Score", lambda x: ", ".join(
                        f"{v:.4f}" for v in x if pd.notna(v)
                    )),
                    Risk_Statements=("Risk Statement", lambda x: " | ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                )
                .reset_index()
            )

            impact_plot = px.bar(
                impact_aggregated,
                x="Impact",
                y="Risk_Count",
                custom_data=[
                    "Impact",
                    "Risk_Count",
                    "Risk_IDs",
                    "Relevance_Scores",
                    "Risk_Statements",
                ],
                labels={
                    "Impact": "Impact",
                    "Risk_Count": "Number of Risks",
                },
                title="Impact Distribution"
            )

            impact_plot.update_traces(
                hovertemplate=(
                    "<b>Impact:</b> %{customdata[0]}<br>"
                    "<b>Risk Count:</b> %{customdata[1]}<br>"
                    "<b>Risk IDs:</b> %{customdata[2]}<br>"
                    "<b>Relevance Scores:</b> %{customdata[3]}<br>"
                    "<b>Risk Statements:</b> %{customdata[4]}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                impact_plot,
                use_container_width=True
            )


        # --------------------------------------------------------
        # RESPONSIBLE PARTY
        # --------------------------------------------------------

        with chart_col4:

            st.write(
                "Responsible Party"
            )

            party_chart_df = (
                results_df[
                    ["Risk ID", "Responsible Party", "Risk Relevance Score", "Risk Statement"]
                ]
                .copy()
            )

            party_chart_df["Responsible Party"] = (
                party_chart_df["Responsible Party"].fillna("Not Available")
            )

            party_chart_df["Risk Relevance Score"] = pd.to_numeric(
                party_chart_df["Risk Relevance Score"],
                errors="coerce"
            )

            party_aggregated = (
                party_chart_df
                .groupby("Responsible Party", sort=False)
                .agg(
                    Risk_Count=("Risk ID", "count"),
                    Risk_IDs=("Risk ID", lambda x: ", ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                    Relevance_Scores=("Risk Relevance Score", lambda x: ", ".join(
                        f"{v:.4f}" for v in x if pd.notna(v)
                    )),
                    Risk_Statements=("Risk Statement", lambda x: " | ".join(
                        str(v) for v in x if pd.notna(v)
                    )),
                )
                .reset_index()
            )

            party_plot = px.bar(
                party_aggregated,
                x="Responsible Party",
                y="Risk_Count",
                custom_data=[
                    "Responsible Party",
                    "Risk_Count",
                    "Risk_IDs",
                    "Relevance_Scores",
                    "Risk_Statements",
                ],
                labels={
                    "Responsible Party": "Responsible Party",
                    "Risk_Count": "Number of Risks",
                },
                title="Responsible Party Distribution"
            )

            party_plot.update_traces(
                hovertemplate=(
                    "<b>Responsible Party:</b> %{customdata[0]}<br>"
                    "<b>Risk Count:</b> %{customdata[1]}<br>"
                    "<b>Risk IDs:</b> %{customdata[2]}<br>"
                    "<b>Relevance Scores:</b> %{customdata[3]}<br>"
                    "<b>Risk Statements:</b> %{customdata[4]}"
                    "<extra></extra>"
                )
            )

            st.plotly_chart(
                party_plot,
                use_container_width=True
            )


        # --------------------------------------------------------
        # RISK-ID RELEVANCE RANKING
        # --------------------------------------------------------

        st.write("### Risk-ID Relevance Ranking")
        st.caption(
            "Hover over any bar to see the Risk ID, risk category, relevance score, and full risk statement."
        )

        risk_id_chart_df = (
            results_df[["Risk ID", "Risk Category", "Risk Relevance Score", "Risk Statement"]]
            .copy()
        )
        risk_id_chart_df["Risk ID"] = risk_id_chart_df["Risk ID"].fillna("Not Available")
        risk_id_chart_df["Risk Category"] = risk_id_chart_df["Risk Category"].fillna("Not Available")
        risk_id_chart_df["Risk Relevance Score"] = pd.to_numeric(
            risk_id_chart_df["Risk Relevance Score"], errors="coerce"
        )
        risk_id_chart_df = risk_id_chart_df.dropna(subset=["Risk Relevance Score"])
        risk_id_chart_df = risk_id_chart_df.sort_values(
            "Risk Relevance Score", ascending=True
        )

        risk_id_plot = px.bar(
            risk_id_chart_df,
            x="Risk Relevance Score",
            y="Risk ID",
            orientation="h",
            hover_data={
                "Risk ID": True,
                "Risk Category": True,
                "Risk Relevance Score": ":.4f",
                "Risk Statement": True,
            },
            labels={
                "Risk Relevance Score": "Relevance Score",
                "Risk ID": "Risk ID",
            },
            title="Risk-ID Relevance Ranking"
        )
        risk_id_plot.update_traces(
            hovertemplate=(
                "<b>Risk ID:</b> %{y}<br>"
                "<b>Category:</b> %{customdata[0]}<br>"
                "<b>Relevance Score:</b> %{x:.4f}<br>"
                "<b>Risk Statement:</b> %{customdata[2]}<extra></extra>"
            )
        )
        st.plotly_chart(risk_id_plot, use_container_width=True)


        # ========================================================
        # TOP RISKS
        # ========================================================

        st.subheader(
            f"🚨 Top {len(results_df)} Contract Risks"
        )


        # ========================================================
        # DISPLAY COLUMNS
        # ========================================================

        display_columns = [

            "Contract Sentence",

            "Sentence Type",

            "Risk ID",

            "Risk Statement",

            "Risk Category",

            "ML Predicted Category",

            "Category Confidence (%)",

            "Responsible Party",

            "ML Predicted Responsible Party",

            "Party Confidence (%)",

            "Probability",

            "Impact",

            "Project Type",

            "Probability Factor",

            "Mitigation Effectiveness",

            "Risk Owner",

            "Early Warning Indicators",

            "Mitigation Actions",

            "Risk Relevance Score"
        ]

        display_columns = [

            column
            for column in display_columns
            if column in results_df.columns

        ]

        # Use unambiguous terminology in the final risk-analysis display.
        display_df = results_df[display_columns].rename(
            columns={
                "Category Confidence (%)": "Decision Strength",
                "ML Predicted Responsible Party": "Predicted Responsible Party"
            }
        )

        # ========================================================
        # RISK-ANALYSIS TERMINOLOGY GUIDE
        # ========================================================

        with st.expander("ℹ️ What the risk-analysis numbers mean", expanded=True):
            st.markdown(
                """
                **ML Predicted Category** — the risk category assigned by the selected ML/hybrid classifier to the contract sentence.
                It is a classification output, **not a probability**.

                **Decision Strength** — the strength of the hybrid classification decision, shown as 0–100%.
                For the Confidence Weighted Hybrid, it is the winning category's share of the combined model confidence.
                It is **not model accuracy and is not a calibrated probability of correctness**.

                **Predicted Responsible Party** — the responsible party predicted by the independent party-classification model
                (Employer, Contractor, or Shared).

                **Party Confidence** — the party model's confidence/probability for its predicted responsible party, shown as 0–100%.
                It indicates model confidence in that class; it is **not the model's accuracy**.

                **Risk Relevance Score** — a heuristic prioritization score used to rank how strongly a risk-library item matches
                the contract sentence. It combines semantic similarity, concept overlap, category fit, party fit, and risk signals.
                Higher values mean higher retrieval relevance; it is **not a probability or percentage of risk occurrence**.

                **Probability Factor** — the numeric factor assigned to the risk library's qualitative Probability level for
                quantitative exposure screening: **Low = 0.15, Medium = 0.40, High = 0.70**.

                **Mitigation Effectiveness** — the assumed percentage reduction in expected monetary value after mitigation.
                The current risk library uses a **35% screening assumption**. This is an assumption from the risk library,
                **not an ML-estimated effectiveness percentage**.
                """
            )


        # ========================================================
        # TABLE
        # ========================================================

        st.dataframe(
            display_df,
            use_container_width=True,
            height=700
        )


        # ========================================================
        # CSV DOWNLOAD
        # ========================================================

        csv = (
            display_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            label="📥 Download Risk Analysis CSV",
            data=csv,
            file_name=f"{re.sub(r'[^A-Za-z0-9_-]+', '_', uploaded_file.name.rsplit('.', 1)[0])}_top_risks.csv",
            mime="text/csv"
        )


        # ========================================================
        # DISCLAIMER
        # ========================================================

        st.caption(
            "This system provides automated risk-identification "
            "and prioritization support based on contract text "
            "and a structured risk knowledge base. Results should "
            "be reviewed by qualified contract, legal, commercial, "
            "or project-risk professionals."
        )


else:

    st.info(
        "Please upload one or more contract PDFs to begin analysis."
    )