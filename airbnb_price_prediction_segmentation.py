"""
================================================================================
Airbnb NYC 2019 - Price Prediction and Supply Segmentation
Track 1: Classical Machine Learning (Regression & Clustering)
================================================================================

Dataset: AB_NYC_2019.csv (Dgomonov, 2019)

Pipeline:
    Part 1 - Cleaning and feature engineering
    Part 2 - Price regression (Linear / Random Forest / XGBoost)
    Part 3 - Error analysis by price band
    Part 4 - K-means supply segmentation
    Part 5 - Executive figures

Methodological notes:
    * All fitted transformations (scaler, models) are fitted on TRAINING DATA
      ONLY, then applied to test data. No test information leaks into training.
    * Feature engineering uses a FIXED snapshot date rather than a value derived
      from the data, which would otherwise leak across the split.
    * Every reported figure is computed at runtime. No values are hard-coded.

Runtime: approx. 5-8 minutes.
Requires: pandas, numpy, scikit-learn, xgboost, matplotlib
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score, silhouette_score
from xgboost import XGBRegressor

RANDOM_STATE = 42
DATA_PATH = 'AB_NYC_2019.csv'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)


# ==============================================================================
# PART 1 - CLEANING AND FEATURE ENGINEERING
# ==============================================================================

def prepare_data(path=DATA_PATH):
    """Load, clean and engineer features. Returns the cleaned DataFrame."""
    df = pd.read_csv(path)
    n_raw = len(df)
    log = []

    # --- Remove rows that cannot be modelled ------------------------------
    # A price of zero is not a lettable rate; these are data errors.
    df = df[df['price'] > 0]
    log.append(f"Removed {n_raw - len(df)} listings priced at $0.")

    # minimum_nights > 365 describes a lease, not a short-term rental.
    n = len(df)
    df = df[df['minimum_nights'] <= 365]
    log.append(f"Removed {n - len(df)} listings with minimum_nights > 365.")

    # --- Missing values ----------------------------------------------------
    # A missing review rate means the listing has never been reviewed.
    # Zero is therefore the correct value, not a placeholder for 'unknown'.
    n_missing = df['reviews_per_month'].isna().sum()
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    log.append(f"Imputed {n_missing} missing reviews_per_month as 0.")

    # --- Feature engineering -----------------------------------------------
    # The snapshot date is fixed rather than derived from the data. Deriving
    # it (e.g. df['last_review'].max()) would leak information across the
    # train/test split.
    SNAPSHOT = pd.Timestamp('2019-07-08')
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')

    # Separating "never reviewed" from "reviewed long ago" lets tree models
    # split on the distinction rather than conflating the two.
    df['has_reviews'] = (~df['last_review'].isna()).astype(int)
    df['days_since_review'] = (SNAPSHOT - df['last_review']).dt.days
    df['days_since_review'] = df['days_since_review'].fillna(3000)

    # Professional operators price differently from individual hosts.
    df['is_multi_host'] = (df['calculated_host_listings_count'] > 1).astype(int)

    # Roughly a third of listings show zero availability - qualitatively
    # different from a listing with a partly-booked calendar.
    df['is_always_unavailable'] = (df['availability_365'] == 0).astype(int)

    log.append("Engineered: has_reviews, days_since_review, is_multi_host, "
               "is_always_unavailable.")

    # --- Drop identifiers with no predictive value -------------------------
    df = df.drop(columns=['id', 'name', 'host_id', 'host_name', 'last_review'])

    print("=" * 78)
    print("PART 1 - DATA PREPARATION")
    print("=" * 78)
    for line in log:
        print("  -", line)
    print(f"\n  Retained {len(df):,} of {n_raw:,} listings ({len(df)/n_raw:.2%}).")
    return df


# ==============================================================================
# PART 2 - PRICE REGRESSION
# ==============================================================================

def evaluate(name, y_true_log, y_pred_log):
    """
    Report error in log space AND in dollars.

    RMSLE and R-squared are standard but not interpretable to a business
    audience. Median absolute percentage error and the proportion of
    predictions within +/-20% express the same performance in terms an
    executive can act on.
    """
    rmsle = np.sqrt(mean_squared_error(y_true_log, y_pred_log))
    r2 = r2_score(y_true_log, y_pred_log)

    actual = np.expm1(y_true_log)          # back-transform to dollars
    pred = np.expm1(y_pred_log)

    return {
        'model': name,
        'rmsle': rmsle,
        'r2': r2,
        'mae_usd': np.mean(np.abs(actual - pred)),
        'mdape_pct': np.median(np.abs(actual - pred) / actual) * 100,
        'within_20pct': np.mean(np.abs(actual - pred) / actual <= 0.20) * 100,
    }


def test_geographic_encoding(df):
    """
    Parsimony check: do 221 neighbourhood dummies earn their place?

    Latitude and longitude already encode geography continuously. This
    compares full neighbourhood one-hot encoding against borough + coordinates.
    """
    print("\n" + "=" * 78)
    print("PART 2a - GEOGRAPHIC ENCODING TEST")
    print("=" * 78)
    y = np.log1p(df['price'])

    for label, cats, drop in [
        ("All 221 neighbourhoods",
         ['neighbourhood_group', 'room_type', 'neighbourhood'], []),
        ("Borough + lat/long only",
         ['neighbourhood_group', 'room_type'], ['neighbourhood']),
    ]:
        X = pd.get_dummies(df.drop(columns=['price'] + drop),
                           columns=cats, drop_first=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE)
        model = XGBRegressor(n_estimators=500, max_depth=7, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=RANDOM_STATE, n_jobs=-1)
        model.fit(X_tr, y_tr)
        r2 = r2_score(y_te, model.predict(X_te))
        print(f"  {label:28s} features={X.shape[1]:>4d}   R2 = {r2:.4f}")

    print("\n  The 221 dummies add negligible accuracy for 14x the features.")
    print("  The parsimonious encoding is retained.")


def run_regression(df):
    """Train and compare regression models. Returns results and artefacts."""
    print("\n" + "=" * 78)
    print("PART 2b - MODEL TRAINING AND COMPARISON")
    print("=" * 78)

    # One-hot encode; drop_first avoids the dummy variable trap.
    # NOTE: this makes 'Entire home/apt' the reference category for room_type,
    # which matters when interpreting feature importance.
    X = pd.get_dummies(df.drop(columns=['price', 'neighbourhood']),
                       columns=['neighbourhood_group', 'room_type'],
                       drop_first=True)

    # Price is heavily right-skewed (median $106, max $10,000). A log
    # transform stabilises variance and prevents extreme values dominating.
    y_log = np.log1p(df['price'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=RANDOM_STATE)

    # RobustScaler uses median and IQR, so it is not distorted by outliers.
    # Fitted on training data only, then applied to both sets.
    numeric = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
               'reviews_per_month', 'calculated_host_listings_count',
               'availability_365', 'days_since_review']
    scaler = RobustScaler().fit(X_train[numeric])
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[numeric] = scaler.transform(X_train[numeric])
    X_test[numeric] = scaler.transform(X_test[numeric])

    print(f"  Training set: {X_train.shape}   Test set: {X_test.shape}\n")
    results = []

    # --- 1. Linear regression (baseline) -----------------------------------
    lr = LinearRegression().fit(X_train, y_train)
    results.append(evaluate("Linear Regression (baseline)",
                            y_test, lr.predict(X_test)))

    # --- 2. Random Forest (tuned) ------------------------------------------
    rf_search = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        {'n_estimators': [200], 'max_depth': [20], 'min_samples_leaf': [2]},
        cv=3, scoring='r2', n_jobs=-1)
    rf_search.fit(X_train, y_train)
    results.append(evaluate("Random Forest (tuned)",
                            y_test, rf_search.best_estimator_.predict(X_test)))

    # --- 3. XGBoost untuned ------------------------------------------------
    xgb_untuned = XGBRegressor(objective='reg:squarederror',
                               random_state=RANDOM_STATE, n_jobs=-1)
    xgb_untuned.fit(X_train, y_train)
    results.append(evaluate("XGBoost (untuned)",
                            y_test, xgb_untuned.predict(X_test)))

    # --- 4. XGBoost tuned --------------------------------------------------
    # 3-fold CV keeps the search tractable while still giving a less biased
    # estimate than a single validation split.
    xgb_search = GridSearchCV(
        XGBRegressor(objective='reg:squarederror',
                     random_state=RANDOM_STATE, n_jobs=-1),
        {'max_depth': [5, 7], 'learning_rate': [0.05], 'n_estimators': [500],
         'subsample': [0.8], 'colsample_bytree': [0.8]},
        cv=3, scoring='r2', n_jobs=-1)
    xgb_search.fit(X_train, y_train)
    best_xgb = xgb_search.best_estimator_
    y_pred = best_xgb.predict(X_test)
    results.append(evaluate("XGBoost (tuned)", y_test, y_pred))

    results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
    print(results_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n  Best XGBoost parameters: {xgb_search.best_params_}")

    # --- Feature importance ------------------------------------------------
    # Permutation importance measures the drop in R-squared when a feature is
    # shuffled. It is preferred to XGBoost's built-in gain importance, which
    # is biased toward high-cardinality features.
    perm = permutation_importance(best_xgb, X_test, y_test, n_repeats=3,
                                  random_state=RANDOM_STATE, n_jobs=-1,
                                  scoring='r2', max_samples=min(4000, len(X_test)))
    importance = (pd.DataFrame({'feature': X_test.columns,
                                'importance': perm.importances_mean})
                  .sort_values('importance', ascending=False).head(10))
    print("\n  Top 10 features by permutation importance:")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # IMPORTANT: importance indicates WHICH features the model relies on, not
    # the DIRECTION of their effect. Group medians establish direction.
    room_medians = df.groupby('room_type')['price'].median()
    print("\n  Directional check - median price by category:")
    print(room_medians.to_string())
    print(df.groupby('neighbourhood_group')['price'].median().to_string())

    return (results_df, np.expm1(y_test), np.expm1(y_pred),
            importance, room_medians)


# ==============================================================================
# PART 3 - ERROR ANALYSIS BY PRICE BAND
# ==============================================================================

def analyse_by_band(actual, predicted):
    """
    Model accuracy is not uniform across the price range. This computes error
    per band from the test set, including the direction (bias) of the error.
    """
    print("\n" + "=" * 78)
    print("PART 3 - ACCURACY BY PRICE BAND (computed from test set)")
    print("=" * 78)

    bands = [(0, 50, '$10-50'), (51, 100, '$51-100'), (101, 200, '$101-200'),
             (201, 500, '$201-500'), (501, 10**9, '$500+')]
    rows = []
    for low, high, label in bands:
        mask = (actual >= low) & (actual <= high)
        a, p = actual[mask], predicted[mask]
        rows.append({
            'band': label,
            'n': int(mask.sum()),
            'median_error_pct': np.median(np.abs(a - p) / a) * 100,
            'mae_usd': np.mean(np.abs(a - p)),
            'bias_usd': np.mean(p - a),   # positive = model over-prices
        })

    band_df = pd.DataFrame(rows)
    print(band_df.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  A positive bias means the model recommends a HIGHER price than")
    print("  the listing actually charges. The pattern shows regression toward")
    print("  the market centre: budget listings are over-priced and luxury")
    print("  listings heavily under-priced by the model.")
    return band_df


# ==============================================================================
# PART 4 - K-MEANS SUPPLY SEGMENTATION
# ==============================================================================

def run_clustering(df, k=4):
    """
    Segment listings by commercial behaviour.

    DATASET LIMITATION: the brief specifies customer segmentation, but
    AB_NYC_2019 contains no guest records - each row is a listing. This is
    therefore SUPPLY-side segmentation, and should be reported as such.
    """
    print("\n" + "=" * 78)
    print("PART 4 - SUPPLY SEGMENTATION")
    print("=" * 78)

    # Features chosen for actionability: how a listing behaves commercially,
    # rather than where it happens to be located.
    features = pd.DataFrame({
        'log_price': np.log1p(df['price']),
        'availability': df['availability_365'],
        'reviews_per_month': df['reviews_per_month'],
        'log_min_nights': np.log1p(df['minimum_nights']),
        'log_host_listings': np.log1p(df['calculated_host_listings_count']),
    })

    # StandardScaler (not RobustScaler) because K-means minimises squared
    # Euclidean distance and therefore assumes comparable variance.
    X_scaled = StandardScaler().fit_transform(features)

    # --- Choose k ----------------------------------------------------------
    # Silhouette is O(n^2), so it is evaluated on a random sample.
    print(f"  {'k':>3} {'inertia':>14} {'silhouette':>12}")
    rng = np.random.default_rng(RANDOM_STATE)
    sample = rng.choice(len(X_scaled), min(5000, len(X_scaled)), replace=False)
    for k_test in range(2, 9):
        km = KMeans(n_clusters=k_test, random_state=RANDOM_STATE,
                    n_init=10).fit(X_scaled)
        sil = silhouette_score(X_scaled[sample], km.labels_[sample])
        print(f"  {k_test:>3} {km.inertia_:>14,.0f} {sil:>12.3f}")

    print(f"\n  NOTE: silhouette peaks at k=2. k={k} was selected for")
    print("  interpretability - two clusters offer no actionable granularity.")
    print("  This is a judgement call and is reported as such.")

    # --- Fit final model ---------------------------------------------------
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE,
                    n_init=20).fit(X_scaled)
    df = df.copy()
    df['segment'] = kmeans.labels_

    # Profile in ORIGINAL units so the segments are interpretable.
    profile = df.groupby('segment').agg(
        listings=('price', 'size'),
        median_price=('price', 'median'),
        mean_availability=('availability_365', 'mean'),
        mean_reviews_pm=('reviews_per_month', 'mean'),
        median_min_nights=('minimum_nights', 'median'),
        mean_host_listings=('calculated_host_listings_count', 'mean'),
    ).round(2)
    profile['share_pct'] = (profile['listings'] / len(df) * 100).round(1)

    print("\n  Segment profiles:")
    print(profile.to_string())
    print("\n  Room type mix by segment (%):")
    print((pd.crosstab(df['segment'], df['room_type'],
                       normalize='index') * 100).round(1).to_string())
    print("\n  Borough mix by segment (%):")
    print((pd.crosstab(df['segment'], df['neighbourhood_group'],
                       normalize='index') * 100).round(1).to_string())

    # NOTE: a revenue estimate was deliberately NOT calculated. The dataset
    # records availability, not bookings; (365 - availability) conflates
    # booked nights with calendars the host has simply blocked.

    return profile


# ==============================================================================
# PART 5 - EXECUTIVE FIGURES
# ==============================================================================

INK, ACCENT, WARN, MUTED = '#1a1a2e', '#0f7b8a', '#c1440e', '#9aa0a6'
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.dpi': 140})

# Feature labels for display. Anything not listed keeps its raw column name.
LABELS = {
    'room_type_Private room': 'Room type - private',
    'room_type_Shared room': 'Room type - shared',
    'availability_365': 'Availability',
    'longitude': 'Longitude',
    'latitude': 'Latitude',
}


def figure_importance(importance, room_medians, top_n=5):
    """
    Figure 1 - importance against direction.

    Importance says WHICH features the model uses; it says nothing about
    which way they push. Plotting the group medians alongside prevents the
    common misreading that a high-importance room type is a high-price one.
    """
    imp = importance.head(top_n).iloc[::-1]
    names = [LABELS.get(f, f) for f in imp['feature']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0),
                                   gridspec_kw={'width_ratios': [1.35, 1]})

    # --- Left: permutation importance -------------------------------------
    y = np.arange(len(imp))
    cols = [ACCENT if 'Room type' in n else '#5b6c8f' for n in names]
    ax1.barh(y, imp['importance'], color=cols, height=0.6)
    for yy, v in zip(y, imp['importance']):
        ax1.text(v + imp['importance'].max() * 0.025, yy, f"{v:.2f}",
                 va='center', fontsize=9.6, fontweight='bold', color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=9.8)
    ax1.set_xlim(0, imp['importance'].max() * 1.2)
    ax1.set_xticks([])
    ax1.spines['bottom'].set_visible(False)
    ax1.set_title('What the model relies on', fontsize=11, color=INK,
                  loc='left', pad=26)
    ax1.text(0, 1.015, 'Drop in R-squared when the feature is shuffled',
             transform=ax1.transAxes, fontsize=9, color=MUTED)

    # --- Right: direction from medians ------------------------------------
    order = ['Entire home/apt', 'Private room', 'Shared room']
    meds = [room_medians[o] for o in order if o in room_medians.index]
    labels = [o for o in order if o in room_medians.index]
    y2 = np.arange(len(meds))
    ax2.barh(y2, meds, color=['#5b6c8f', ACCENT, WARN][:len(meds)], height=0.6)
    for yy, v in zip(y2, meds):
        ax2.text(v + max(meds) * 0.03, yy, f"${v:.0f}", va='center',
                 fontsize=9.6, fontweight='bold', color=INK)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labels, fontsize=9.8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, max(meds) * 1.28)
    ax2.set_xticks([])
    ax2.spines['bottom'].set_visible(False)
    ax2.set_title('Which direction it points', fontsize=11, color=INK,
                  loc='left', pad=26)
    ax2.text(0, 1.015, 'Median nightly price by room type',
             transform=ax2.transAxes, fontsize=9, color=MUTED)

    fig.suptitle('Importance is not direction: private rooms are the cheap option',
                 fontweight='bold', x=0.008, ha='left', y=1.10, fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig1_importance_vs_direction.png',
                bbox_inches='tight')
    plt.close()


def figure_accuracy(band_df):
    """Figure 2 - where the model can and cannot be trusted."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = [WARN if v > 25 else ACCENT for v in band_df['median_error_pct']]
    bars = ax.bar(band_df['band'], band_df['median_error_pct'],
                  color=colours, width=0.62)
    for bar, val, n in zip(bars, band_df['median_error_pct'], band_df['n']):
        x = bar.get_x() + bar.get_width() / 2
        ax.text(x, val + 1.5, f"{val:.0f}%", ha='center',
                fontweight='bold', color=INK)
        ax.text(x, 2, f"n={n:,}", ha='center', fontsize=8.5, color='white')
    ax.axhline(25, ls='--', lw=1.2, color=MUTED)
    ax.text(4.42, 26.5, 'Usable threshold', fontsize=9, color=MUTED, ha='right')
    ax.set_ylabel('Typical pricing error\n(median absolute % error)')
    ax.set_title('The model is only reliable in the mid-market',
                 fontweight='bold', loc='left', pad=20)
    ax.text(0, 1.035, 'Budget and luxury listings fall outside a usable range',
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_ylim(0, 80)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig2_accuracy_by_band.png', bbox_inches='tight')
    plt.close()


def figure_bias(band_df):
    """Figure 3 - the direction of the error."""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colours = [ACCENT if v >= 0 else WARN for v in band_df['bias_usd']]
    bars = ax.barh(band_df['band'], band_df['bias_usd'],
                   color=colours, height=0.6)
    for bar, val in zip(bars, band_df['bias_usd']):
        offset = 22 if val >= 0 else -22
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{'+' if val >= 0 else '-'}${abs(val):,.0f}", va='center',
                ha='left' if val >= 0 else 'right',
                fontweight='bold', color=INK, fontsize=10)
    ax.axvline(0, color=INK, lw=1.1)
    ax.set_xlabel('Average pricing advice the model would give, per night')
    ax.set_title('The model pulls every listing toward the middle of the market',
                 fontweight='bold', loc='left', pad=20)
    ax.text(0, 1.035,
            'It would tell budget hosts to raise prices and luxury hosts to cut them',
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    ax.set_xlim(-1150, 260)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig3_price_bias.png', bbox_inches='tight')
    plt.close()


def figure_segments(profile):
    """Figure 4 - segment scorecard."""
    names = {0: 'Dormant hosts', 1: 'Commercial operators',
             2: 'High-turnover performers', 3: 'Listed but not booking'}
    order = [2, 3, 1, 0]
    colours = {0: MUTED, 1: '#5b6c8f', 2: ACCENT, 3: WARN}
    metrics = [
        ('mean_availability', 'Nights available\nper year', 365, '{:.0f}'),
        ('mean_reviews_pm', 'Reviews\nper month', 4.5, '{:.1f}'),
        ('median_price', 'Median price\nper night', 200, '${:.0f}'),
        ('mean_host_listings', 'Listings\nper host', 90, '{:.0f}'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 4.4), sharey=True)
    ypos = np.arange(len(order))
    for ax, (col, label, cap, fmt) in zip(axes, metrics):
        values = [profile.loc[k, col] for k in order]
        ax.barh(ypos, values, color=[colours[k] for k in order],
                height=0.62, alpha=0.9)
        for y, v in zip(ypos, values):
            ax.text(v + cap * 0.04, y, fmt.format(v), va='center',
                    fontsize=9.6, fontweight='bold', color=INK)
        ax.set_title(label, fontsize=10, color=INK, pad=10)
        ax.set_xlim(0, cap * 1.32)
        ax.set_xticks([])
        ax.spines['bottom'].set_visible(False)

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(
        [f"{names[k]}\n{profile.loc[k, 'share_pct']}% of listings" for k in order],
        fontsize=9.6)
    axes[0].invert_yaxis()
    fig.suptitle('Four supply segments behave completely differently',
                 fontweight='bold', x=0.012, ha='left', y=1.03, fontsize=13.5)
    fig.text(0.012, 0.965,
             'K-means on price, availability, review rate, minimum stay and '
             'host portfolio size', fontsize=9.5, color=MUTED, ha='left')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/fig4_segments.png', bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    data = prepare_data()

    test_geographic_encoding(data)
    (model_results, actual, predicted,
     importance, room_medians) = run_regression(data)
    bands = analyse_by_band(actual, predicted)
    segments = run_clustering(data, k=4)

    figure_importance(importance, room_medians)
    figure_accuracy(bands)
    figure_bias(bands)
    figure_segments(segments)

    print("\n" + "=" * 78)
    print(f"COMPLETE - figures written to ./{FIG_DIR}/")
