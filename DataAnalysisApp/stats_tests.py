"""
DataIQ Pro Analytics - Statistical Hypothesis Testing Suite

This module provides comprehensive hypothesis testing capabilities for data analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.weightstats import ztest
import warnings
warnings.filterwarnings('ignore')


def run_hypothesis_testing(df: pd.DataFrame) -> None:
    """
    Main function for running hypothesis testing suite.
    
    Args:
        df: The DataFrame to analyze.
    """
    st.markdown("### 🔬 Statistical Hypothesis Testing")
    
    # Alpha selector
    alpha = st.select_slider(
        "Significance Level (α)",
        options=[0.01, 0.05, 0.10],
        value=0.05,
        format_func=lambda x: f"{x:.0%}"
    )
    
    # Test category selector
    test_categories = {
        "Normality Tests": ["Shapiro-Wilk", "D'Agostino-Pearson"],
        "One-Sample t-test": ["One-Sample t-test"],
        "Independent Two-Sample t-test / Welch": ["Independent t-test", "Welch t-test"],
        "Paired t-test": ["Paired t-test"],
        "One-Way ANOVA": ["One-Way ANOVA"],
        "Chi-Square Test of Independence": ["Chi-Square Test"],
        "Mann-Whitney U": ["Mann-Whitney U"],
        "Wilcoxon Signed-Rank": ["Wilcoxon Signed-Rank"],
        "Kruskal-Wallis": ["Kruskal-Wallis"],
        "Levene / Bartlett Variance Test": ["Levene Test", "Bartlett Test"],
        "Pearson / Spearman / Kendall Correlation": ["Pearson", "Spearman", "Kendall"],
        "Z-Test for Proportions": ["Z-Test for Proportions"]
    }
    
    category = st.selectbox("Select Test Category", list(test_categories.keys()))
    
    if category in test_categories:
        test_type = st.selectbox("Select Test Type", test_categories[category])
        
        # Display test explanation
        _display_test_explanation(test_type, alpha)
        
        # Column selectors and test execution
        if test_type in ["Shapiro-Wilk", "D'Agostino-Pearson"]:
            _run_normality_test(df, test_type, alpha)
        elif test_type == "One-Sample t-test":
            _run_one_sample_ttest(df, alpha)
        elif test_type in ["Independent t-test", "Welch t-test"]:
            _run_independent_ttest(df, test_type, alpha)
        elif test_type == "Paired t-test":
            _run_paired_ttest(df, alpha)
        elif test_type == "One-Way ANOVA":
            _run_anova(df, alpha)
        elif test_type == "Chi-Square Test":
            _run_chi_square(df, alpha)
        elif test_type == "Mann-Whitney U":
            _run_mann_whitney(df, alpha)
        elif test_type == "Wilcoxon Signed-Rank":
            _run_wilcoxon(df, alpha)
        elif test_type == "Kruskal-Wallis":
            _run_kruskal_wallis(df, alpha)
        elif test_type in ["Levene Test", "Bartlett Test"]:
            _run_variance_test(df, test_type, alpha)
        elif test_type in ["Pearson", "Spearman", "Kendall"]:
            _run_correlation_test(df, test_type, alpha)
        elif test_type == "Z-Test for Proportions":
            _run_proportion_ztest(df, alpha)


def _display_test_explanation(test_type: str, alpha: float) -> None:
    """Display explanation for the selected test."""
    explanations = {
        "Shapiro-Wilk": {
            "purpose": "Tests whether a sample comes from a normally distributed population.",
            "when": "When you want to check if your data follows a normal distribution before applying parametric tests.",
            "h0": "H₀: The data is normally distributed",
            "h1": "H₁: The data is not normally distributed",
            "assumptions": "None (distribution-free test)"
        },
        "D'Agostino-Pearson": {
            "purpose": "Tests normality based on skewness and kurtosis.",
            "when": "Alternative to Shapiro-Wilk for larger samples (n > 5000).",
            "h0": "H₀: The data is normally distributed",
            "h1": "H₁: The data is not normally distributed",
            "assumptions": "None"
        },
        "One-Sample t-test": {
            "purpose": "Tests whether the mean of a single sample differs from a known population mean.",
            "when": "When comparing a sample mean to a known or hypothesized population mean.",
            "h0": "H₀: Sample mean = Population mean",
            "h1": "H₁: Sample mean ≠ Population mean",
            "assumptions": "Normal distribution, independent observations"
        },
        "Independent t-test": {
            "purpose": "Compares means of two independent groups.",
            "when": "When comparing means between two different groups (e.g., treatment vs control).",
            "h0": "H₀: μ₁ = μ₂ (means are equal)",
            "h1": "H₁: μ₁ ≠ μ₂ (means are different)",
            "assumptions": "Normal distribution, equal variances, independent samples"
        },
        "Welch t-test": {
            "purpose": "Compares means of two independent groups with unequal variances.",
            "when": "When comparing means between two groups with different variances.",
            "h0": "H₀: μ₁ = μ₂",
            "h1": "H₁: μ₁ ≠ μ₂",
            "assumptions": "Normal distribution, independent samples (unequal variances allowed)"
        },
        "Paired t-test": {
            "purpose": "Compares means of two related groups (before-after, matched pairs).",
            "when": "When measurements are taken on the same subjects before and after treatment.",
            "h0": "H₀: μ_before = μ_after",
            "h1": "H₁: μ_before ≠ μ_after",
            "assumptions": "Normal differences, paired observations"
        },
        "One-Way ANOVA": {
            "purpose": "Tests for differences in means across three or more groups.",
            "when": "When comparing means across multiple groups simultaneously.",
            "h0": "H₀: All group means are equal",
            "h1": "H₁: At least one group mean is different",
            "assumptions": "Normal distribution, equal variances, independent samples"
        },
        "Chi-Square Test": {
            "purpose": "Tests for association between two categorical variables.",
            "when": "When analyzing relationships between categorical variables in contingency tables.",
            "h0": "H₀: Variables are independent",
            "h1": "H₁: Variables are associated",
            "assumptions": "Expected frequencies ≥ 5 in 80% of cells"
        },
        "Mann-Whitney U": {
            "purpose": "Non-parametric test for differences between two independent groups.",
            "when": "When data doesn't meet normality assumptions for t-test.",
            "h0": "H₀: Distributions are identical",
            "h1": "H₁: One distribution is shifted relative to the other",
            "assumptions": "Ordinal or continuous data, independent samples"
        },
        "Wilcoxon Signed-Rank": {
            "purpose": "Non-parametric test for paired differences.",
            "when": "Non-parametric alternative to paired t-test.",
            "h0": "H₀: Median difference = 0",
            "h1": "H₁: Median difference ≠ 0",
            "assumptions": "Paired observations, ordinal or continuous data"
        },
        "Kruskal-Wallis": {
            "purpose": "Non-parametric alternative to one-way ANOVA.",
            "when": "When comparing three or more groups without normality assumption.",
            "h0": "H₀: All groups have identical distributions",
            "h1": "H₁: At least one group has different distribution",
            "assumptions": "Independent samples, ordinal or continuous data"
        },
        "Levene Test": {
            "purpose": "Tests for equality of variances across groups.",
            "when": "Before ANOVA to check homogeneity of variance assumption.",
            "h0": "H₀: Variances are equal across groups",
            "h1": "H₁: Variances are not equal across groups",
            "assumptions": "None (robust to non-normality)"
        },
        "Bartlett Test": {
            "purpose": "Tests for equality of variances (more sensitive to non-normality).",
            "when": "When data is normally distributed and you want to test variance equality.",
            "h0": "H₀: Variances are equal",
            "h1": "H₁: Variances are not equal",
            "assumptions": "Normal distribution"
        },
        "Pearson": {
            "purpose": "Tests linear relationship between two continuous variables.",
            "when": "When both variables are normally distributed and relationship is linear.",
            "h0": "H₀: ρ = 0 (no linear correlation)",
            "h1": "H₁: ρ ≠ 0 (linear correlation exists)",
            "assumptions": "Normal distribution, linear relationship, homoscedasticity"
        },
        "Spearman": {
            "purpose": "Tests monotonic relationship between variables.",
            "when": "When data is ordinal or doesn't meet Pearson assumptions.",
            "h0": "H₀: ρₛ = 0 (no monotonic correlation)",
            "h1": "H₁: ρₛ ≠ 0 (monotonic correlation exists)",
            "assumptions": "Monotonic relationship"
        },
        "Kendall": {
            "purpose": "Tests ordinal association between variables.",
            "when": "For small samples or when ties are present.",
            "h0": "H₀: τ = 0 (no association)",
            "h1": "H₁: τ ≠ 0 (association exists)",
            "assumptions": "Ordinal data"
        },
        "Z-Test for Proportions": {
            "purpose": "Tests differences between proportions in large samples.",
            "when": "When comparing proportions between two groups with large sample sizes.",
            "h0": "H₀: p₁ = p₂ (proportions are equal)",
            "h1": "H₁: p₁ ≠ p₂ (proportions are different)",
            "assumptions": "Large samples (np ≥ 5, n(1-p) ≥ 5)"
        }
    }
    
    if test_type in explanations:
        exp = explanations[test_type]
        st.info(f"""
        **{test_type} Test**
        
        **Purpose:** {exp['purpose']}
        
        **When to use:** {exp['when']}
        
        **Hypotheses:**
        - {exp['h0']}
        - {exp['h1']}
        
        **Assumptions:** {exp['assumptions']}
        
        **Significance Level:** α = {alpha}
        """)


def _run_normality_test(df: pd.DataFrame, test_type: str, alpha: float) -> None:
    """Run normality tests."""
    col = st.selectbox("Select numeric column", df.select_dtypes(include=[np.number]).columns)
    
    if st.button("Run Normality Test"):
        try:
            data = df[col].dropna()
            n = len(data)
            
            if n < 3:
                st.error("Need at least 3 valid observations")
                return
            
            # Descriptive stats
            desc_stats = {
                "N": n,
                "Mean": data.mean(),
                "Median": data.median(),
                "Std Dev": data.std(),
                "Skewness": data.skew(),
                "Kurtosis": data.kurtosis(),
                "Min": data.min(),
                "Max": data.max()
            }
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Descriptive Statistics")
                for key, value in desc_stats.items():
                    st.metric(key, f"{value:.3f}")
            
            with col2:
                st.subheader("Normality Test Results")
                
                if test_type == "Shapiro-Wilk":
                    if n > 5000:
                        st.warning("Shapiro-Wilk not recommended for n > 5000. Consider D'Agostino-Pearson.")
                    
                    stat, p_value = stats.shapiro(data)
                    test_name = "Shapiro-Wilk"
                    formula = "W = ∑(aᵢ(x₍ᵢ₎ - x̄))² / ∑(xᵢ - x̄)²"
                    
                else:  # D'Agostino-Pearson
                    stat, p_value = stats.normaltest(data)
                    test_name = "D'Agostino-Pearson"
                    formula = "K² = Zₛ² + Zₖ² (skewness + kurtosis test)"
                
                # Results
                st.metric("Test Statistic", f"{stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀ (not normal)" if significant else "Fail to reject H₀ (normal)"
                st.markdown(f"**Result:** {color} {result}")
                
                st.markdown(f"**Formula:** {formula}")
                
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_one_sample_ttest(df: pd.DataFrame, alpha: float) -> None:
    """Run one-sample t-test."""
    col = st.selectbox("Select numeric column", df.select_dtypes(include=[np.number]).columns)
    mu = st.number_input("Hypothesized population mean (μ₀)", value=0.0)
    
    if st.button("Run One-Sample t-test"):
        try:
            data = df[col].dropna()
            n = len(data)
            
            if n < 2:
                st.error("Need at least 2 observations")
                return
            
            t_stat, p_value = stats.ttest_1samp(data, mu)
            
            # Confidence interval
            ci_low, ci_high = stats.t.interval(1-alpha, n-1, loc=data.mean(), scale=stats.sem(data))
            
            # Cohen's d
            d = float((data.mean() - mu) / data.std())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                st.metric("Sample Mean", f"{data.mean():.4f}")
                st.metric("95% CI Lower", f"{ci_low:.4f}")
                st.metric("95% CI Upper", f"{ci_high:.4f}")
            
            with col3:
                d_label = "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small" if abs(d) >= 0.2 else "Negligible"
                st.metric("Cohen's d", f"{d:.4f}")
                st.metric("Effect Size", d_label)
            
            st.markdown("**Formula:** t = (x̄ - μ₀) / (s / √n)")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_independent_ttest(df: pd.DataFrame, test_type: str, alpha: float) -> None:
    """Run independent samples t-test."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    value_col = st.selectbox("Select value column", numeric_cols)
    
    if st.button("Run Independent t-test"):
        try:
            groups = df.groupby(group_col)[value_col].apply(list)
            
            if len(groups) != 2:
                st.error("Grouping column must have exactly 2 groups")
                return
            
            group1_data = groups.iloc[0]
            group2_data = groups.iloc[1]
            group1_name = groups.index[0]
            group2_name = groups.index[1]
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                st.error("Each group needs at least 2 observations")
                return
            
            # Test for equal variances
            levene_stat, levene_p = stats.levene(group1_data, group2_data)
            equal_var = levene_p > 0.05
            
            if test_type == "Independent t-test":
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=equal_var)
                test_name = "Student's t-test"
            else:  # Welch
                t_stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
                test_name = "Welch's t-test"
            
            # Means and difference
            mean1, mean2 = np.mean(group1_data), np.mean(group2_data)
            mean_diff = mean1 - mean2
            
            # Cohen's d (pooled)
            n1, n2 = len(group1_data), len(group2_data)
            var1, var2 = np.var(group1_data, ddof=1), np.var(group2_data, ddof=1)
            
            if equal_var:
                pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
            else:
                pooled_std = np.sqrt((var1/n1 + var2/n2))
            
            d = float(mean_diff / pooled_std)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{group1_name} Mean", f"{mean1:.4f}")
                st.metric(f"{group2_name} Mean", f"{mean2:.4f}")
                st.metric("Mean Difference", f"{mean_diff:.4f}")
            
            with col2:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col3:
                d_label = "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small" if abs(d) >= 0.2 else "Negligible"
                st.metric("Cohen's d", f"{d:.4f}")
                st.metric("Effect Size", d_label)
                st.checkbox("Equal variances assumed", value=equal_var)
            
            st.markdown("**Formula:** t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_paired_ttest(df: pd.DataFrame, alpha: float) -> None:
    """Run paired t-test."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col1 = st.selectbox("Select first measurement column", numeric_cols, key="paired1")
    col2 = st.selectbox("Select second measurement column", numeric_cols, key="paired2")
    
    if st.button("Run Paired t-test"):
        try:
            # Remove rows with missing values in either column
            paired_data = df[[col1, col2]].dropna()
            
            if len(paired_data) < 2:
                st.error("Need at least 2 paired observations")
                return
            
            data1 = paired_data[col1]
            data2 = paired_data[col2]
            
            t_stat, p_value = stats.ttest_rel(data1, data2)
            
            # Differences
            differences = data1 - data2
            mean_diff = differences.mean()
            
            # Cohen's d for paired data
            d = float(mean_diff / differences.std())
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mean Difference", f"{mean_diff:.4f}")
                st.metric("Paired Observations", len(paired_data))
            
            with col2:
                st.metric("t-statistic", f"{t_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col3:
                d_label = "Large" if abs(d) >= 0.8 else "Medium" if abs(d) >= 0.5 else "Small" if abs(d) >= 0.2 else "Negligible"
                st.metric("Cohen's d", f"{d:.4f}")
                st.metric("Effect Size", d_label)
            
            st.markdown("**Formula:** t = (x̄_d) / (s_d / √n) where d = x₁ - x₂")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_anova(df: pd.DataFrame, alpha: float) -> None:
    """Run one-way ANOVA."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    value_col = st.selectbox("Select value column", numeric_cols)
    
    if st.button("Run One-Way ANOVA"):
        try:
            groups = [group for name, group in df.groupby(group_col)[value_col] if len(group) > 0]
            
            if len(groups) < 2:
                st.error("Need at least 2 groups")
                return
            
            # ANOVA
            f_stat, p_value = stats.f_oneway(*groups)
            
            # Group summary
            group_names = [name for name, _ in df.groupby(group_col)[value_col]]
            group_stats = []
            
            for i, group_data in enumerate(groups):
                group_stats.append({
                    'Group': group_names[i],
                    'N': len(group_data),
                    'Mean': group_data.mean(),
                    'Std': group_data.std()
                })
            
            group_df = pd.DataFrame(group_stats)
            
            # Eta-squared effect size
            ss_between = sum(len(group) * (group.mean() - df[value_col].mean())**2 for group in groups)
            ss_total = sum((x - df[value_col].mean())**2 for x in df[value_col].dropna())
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("ANOVA Results")
                st.metric("F-statistic", f"{f_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
                st.metric("Eta²", f"{eta_squared:.4f}")
            
            with col2:
                st.subheader("Group Summary")
                st.dataframe(group_df.style.format({
                    'Mean': '{:.3f}',
                    'Std': '{:.3f}'
                }))
            
            # Post-hoc if significant
            if significant and len(groups) <= 5:  # Tukey HSD for small number of groups
                try:
                    from statsmodels.stats.multicomp import pairwise_tukeyhsd
                    tukey = pairwise_tukeyhsd(df[value_col], df[group_col])
                    st.subheader("Post-hoc Tukey HSD")
                    st.text(str(tukey))
                except:
                    st.info("Post-hoc analysis available for significant results with ≤5 groups")
            
            st.markdown("**Formula:** F = MS_between / MS_within")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_chi_square(df: pd.DataFrame, alpha: float) -> None:
    """Run Chi-Square test of independence."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    col1 = st.selectbox("Select first categorical column", cat_cols)
    col2 = st.selectbox("Select second categorical column", cat_cols)
    
    if st.button("Run Chi-Square Test"):
        try:
            # Create contingency table
            contingency = pd.crosstab(df[col1], df[col2])
            
            # Expected frequencies
            expected = stats.contingency.expected_freq(contingency)
            
            # Chi-square test
            chi2_stat, p_value, dof, expected_arr = stats.chi2_contingency(contingency)
            
            # Cramer's V
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2_stat / (n * min_dim)) if min_dim > 0 else 0
            
            # Check expected frequencies
            expected_low = (expected < 5).sum().sum()
            expected_pct = expected_low / expected.size * 100
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("Observed Frequencies")
                st.dataframe(contingency.style.background_gradient(cmap='Blues'))
            
            with col2:
                st.subheader("Expected Frequencies")
                st.dataframe(pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
                           .style.background_gradient(cmap='Greens'))
            
            with col3:
                st.metric("Chi² statistic", f"{chi2_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                st.metric("Degrees of Freedom", dof)
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
                st.metric("Cramer's V", f"{cramers_v:.4f}")
                
                if expected_pct > 20:
                    st.warning(f"⚠️ {expected_pct:.1f}% of expected frequencies < 5 (violates assumption)")
            
            st.markdown("**Formula:** χ² = Σ((Oᵢⱼ - Eᵢⱼ)² / Eᵢⱼ)")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_mann_whitney(df: pd.DataFrame, alpha: float) -> None:
    """Run Mann-Whitney U test."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    value_col = st.selectbox("Select value column", numeric_cols)
    
    if st.button("Run Mann-Whitney U Test"):
        try:
            groups = df.groupby(group_col)[value_col].apply(list)
            
            if len(groups) != 2:
                st.error("Grouping column must have exactly 2 groups")
                return
            
            group1_data = groups.iloc[0]
            group2_data = groups.iloc[1]
            group1_name = groups.index[0]
            group2_name = groups.index[1]
            
            if len(group1_data) < 2 or len(group2_data) < 2:
                st.error("Each group needs at least 2 observations")
                return
            
            u_stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            # Rank-biserial correlation as effect size
            n1, n2 = len(group1_data), len(group2_data)
            r = 1 - (2 * u_stat) / (n1 * n2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("U statistic", f"{u_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                r_label = "Large" if abs(r) >= 0.5 else "Medium" if abs(r) >= 0.3 else "Small" if abs(r) >= 0.1 else "Negligible"
                st.metric("Rank-biserial r", f"{r:.4f}")
                st.metric("Effect Size", r_label)
            
            st.markdown("**Formula:** U = min(U₁, U₂) where U₁ = R₁ - (n₁(n₁+1))/2")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_wilcoxon(df: pd.DataFrame, alpha: float) -> None:
    """Run Wilcoxon signed-rank test."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col1 = st.selectbox("Select first measurement column", numeric_cols, key="wilcox1")
    col2 = st.selectbox("Select second measurement column", numeric_cols, key="wilcox2")
    
    if st.button("Run Wilcoxon Signed-Rank Test"):
        try:
            paired_data = df[[col1, col2]].dropna()
            
            if len(paired_data) < 2:
                st.error("Need at least 2 paired observations")
                return
            
            data1 = paired_data[col1]
            data2 = paired_data[col2]
            
            # Check for ties
            differences = data1 - data2
            zero_diff = (differences == 0).sum()
            
            if zero_diff > 0:
                st.warning(f"⚠️ {zero_diff} zero differences found (ties)")
            
            stat, p_value = stats.wilcoxon(data1, data2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test statistic", f"{stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                st.metric("Paired Observations", len(paired_data))
                st.metric("Zero Differences", zero_diff)
            
            st.markdown("**Formula:** W = sum of ranks for positive differences")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_kruskal_wallis(df: pd.DataFrame, alpha: float) -> None:
    """Run Kruskal-Wallis test."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    value_col = st.selectbox("Select value column", numeric_cols)
    
    if st.button("Run Kruskal-Wallis Test"):
        try:
            groups = [group for name, group in df.groupby(group_col)[value_col] if len(group) > 0]
            
            if len(groups) < 2:
                st.error("Need at least 2 groups")
                return
            
            h_stat, p_value = stats.kruskal(*groups)
            
            # Group names
            group_names = [name for name, _ in df.groupby(group_col)[value_col]]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("H statistic", f"{h_stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                st.metric("Number of Groups", len(groups))
                st.metric("Total Observations", sum(len(g) for g in groups))
            
            st.markdown("**Formula:** H = (12/N(N+1)) * Σ[Rᵢ²/nᵢ] - 3(N+1)")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_variance_test(df: pd.DataFrame, test_type: str, alpha: float) -> None:
    """Run variance equality tests."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    value_col = st.selectbox("Select value column", numeric_cols)
    
    if st.button(f"Run {test_type}"):
        try:
            groups = [group for name, group in df.groupby(group_col)[value_col] if len(group) > 1]
            
            if len(groups) < 2:
                st.error("Need at least 2 groups with 2+ observations each")
                return
            
            if test_type == "Levene Test":
                stat, p_value = stats.levene(*groups)
                test_name = "Levene's Test"
            else:  # Bartlett
                stat, p_value = stats.bartlett(*groups)
                test_name = "Bartlett's Test"
            
            # Group variances
            group_names = [name for name, _ in df.groupby(group_col)[value_col]]
            variances = [np.var(group, ddof=1) for group in groups]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Test Statistic", f"{stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                st.subheader("Group Variances")
                for name, var in zip(group_names, variances):
                    st.metric(f"Var({name})", f"{var:.4f}")
            
            formula = "W = [N-k] * ln(s²_p) - Σ[(nᵢ-1) * ln(s²ᵢ)]" if test_type == "Bartlett" else "W = [Σ(nᵢ-1) * (Yᵢ-Y..)²] / [Σ(nᵢ-1) * (k-1)]"
            st.markdown(f"**Formula:** {formula}")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_correlation_test(df: pd.DataFrame, test_type: str, alpha: float) -> None:
    """Run correlation tests."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    col1 = st.selectbox("Select first variable", numeric_cols, key="corr1")
    col2 = st.selectbox("Select second variable", numeric_cols, key="corr2")
    
    if st.button(f"Run {test_type} Correlation"):
        try:
            data = df[[col1, col2]].dropna()
            
            if len(data) < 2:
                st.error("Need at least 2 paired observations")
                return
            
            x, y = data[col1], data[col2]
            
            if test_type == "Pearson":
                corr_coef, p_value = stats.pearsonr(x, y)
                r_squared = corr_coef ** 2
                test_name = "Pearson r"
                formula = "r = Σ((xᵢ-x̄)(yᵢ-ȳ)) / √[Σ(xᵢ-x̄)² * Σ(yᵢ-ȳ)²]"
            elif test_type == "Spearman":
                corr_coef, p_value = stats.spearmanr(x, y)
                r_squared = None
                test_name = "Spearman ρ"
                formula = "ρ = 1 - (6Σdᵢ²)/(n(n²-1))"
            else:  # Kendall
                corr_coef, p_value = stats.kendalltau(x, y)
                r_squared = None
                test_name = "Kendall τ"
                formula = "τ = (C - D) / (C + D)"
            
            # Strength interpretation
            abs_corr = abs(corr_coef)
            if abs_corr >= 0.9:
                strength = "Very Strong"
            elif abs_corr >= 0.7:
                strength = "Strong"
            elif abs_corr >= 0.5:
                strength = "Moderate"
            elif abs_corr >= 0.3:
                strength = "Weak"
            else:
                strength = "Negligible"
            
            direction = "positive" if corr_coef > 0 else "negative"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(f"{test_name}", f"{corr_coef:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            with col2:
                st.metric("Strength", strength)
                st.metric("Direction", direction)
                if r_squared is not None:
                    st.metric("R²", f"{r_squared:.4f}")
            
            with col3:
                st.metric("Sample Size", len(data))
                st.line_chart(data)
            
            st.markdown(f"**Formula:** {formula}")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")


def _run_proportion_ztest(df: pd.DataFrame, alpha: float) -> None:
    """Run Z-test for proportions."""
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    group_col = st.selectbox("Select grouping column", df.columns)
    binary_col = st.selectbox("Select binary outcome column", cat_cols)
    
    # Get unique values for binary column
    unique_vals = df[binary_col].dropna().unique()
    if len(unique_vals) != 2:
        st.error("Binary column must have exactly 2 unique values")
        return
    
    success_val = st.selectbox("Select 'success' value", unique_vals)
    
    if st.button("Run Z-Test for Proportions"):
        try:
            # Calculate proportions
            groups = df.groupby(group_col)[binary_col].apply(lambda x: (x == success_val).sum() / len(x))
            
            if len(groups) != 2:
                st.error("Grouping column must have exactly 2 groups")
                return
            
            p1, p2 = groups.iloc[0], groups.iloc[1]
            n1 = df.groupby(group_col)[binary_col].size().iloc[0]
            n2 = df.groupby(group_col)[binary_col].size().iloc[1]
            
            # Z-test
            stat, p_value = proportions_ztest([p1 * n1, p2 * n2], [n1, n2])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(f"Proportion {groups.index[0]}", f"{p1:.4f}")
                st.metric(f"Proportion {groups.index[1]}", f"{p2:.4f}")
                st.metric("Difference", f"{p1 - p2:.4f}")
            
            with col2:
                st.metric("Z statistic", f"{stat:.4f}")
                st.metric("p-value", f"{p_value:.4f}")
                significant = p_value < alpha
                color = "🟢" if significant else "🟠"
                result = "Reject H₀" if significant else "Fail to reject H₀"
                st.markdown(f"**Result:** {color} {result}")
            
            st.markdown("**Formula:** z = (p̂₁ - p̂₂) / √(p̂(1-p̂)(1/n₁ + 1/n₂))")
            
        except Exception as e:
            st.error(f"Error running test: {str(e)}")