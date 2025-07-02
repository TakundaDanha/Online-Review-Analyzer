import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
import os

def main():
    path = "data/processed/cleaned_reviews1.csv"
    df = pd.read_csv(path)

    # Groups
    verified = df[df['verified'] == 'Yes']['Score']
    unverified = df[df['verified'] == 'No']['Score']

    # Basic stats
    v_mean, v_std, v_n = verified.mean(), verified.std(), len(verified)
    u_mean, u_std, u_n = unverified.mean(), unverified.std(), len(unverified)

    # T-test (Welch)
    t_stat, t_p = ttest_ind(verified, unverified, equal_var=False)
    t_sig = "Yes" if t_p < 0.05 else "No"

    # Mann-Whitney U test
    u_stat, u_p = mannwhitneyu(verified, unverified, alternative='two-sided')
    u_sig = "Yes" if u_p < 0.05 else "No"

    print(f"\nT-test p-value: {t_p:.4e} => Significant? {t_sig}")
    print(f"Mann-Whitney p-value: {u_p:.4e} => Significant? {u_sig}")

    # Save to CSV
    result = {
        "Test": ["Welch t-test", "Mann-Whitney U"],
        "Verified_Mean": [v_mean, v_mean],
        "Unverified_Mean": [u_mean, u_mean],
        "Verified_Std": [v_std, v_std],
        "Unverified_Std": [u_std, u_std],
        "Verified_N": [v_n, v_n],
        "Unverified_N": [u_n, u_n],
        "Stat_Value": [t_stat, u_stat],
        "P_Value": [t_p, u_p],
        "Significant (p<0.05)": [t_sig, u_sig]
    }

    df_out = pd.DataFrame(result)
    out_path = "data/processed/ab_test_results.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
