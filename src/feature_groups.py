"""
Feature Groups Module

This module defines explicit feature subsets for phased modeling to prevent temporal leakage.
The dataset contains features from different time points:
  - Enrollment: Available at admission
  - Semester 1: Available after first semester
  - Semester 2: Available after second semester

CRITICAL: These feature phases are the foundation of methodologically sound retention forecasting.
"""

# Target variable
TARGET = "Target"

# ============================================================================
# PHASE A: ENROLLMENT FEATURES (Time 0 - At Admission)
# ============================================================================
# These features are available at the time of enrollment and can be used
# for early intervention planning before any academic performance data exists.

ENROLLMENT_FEATURES = [
    # Demographics
    'Marital Status',
    'Nacionality',
    'Age at enrollment',
    'Gender',
    'International',

    # Application Information
    'Application mode',
    'Application order',
    'Course',
    'Daytime/evening attendance',

    # Prior Education
    'Previous qualification',
    'Previous qualification (grade)',
    'Admission grade',

    # Parental Education
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",

    # Financial/Support
    'Displaced',
    'Educational special needs',
    'Debtor',
    'Tuition fees up to date',
    'Scholarship holder',

    # Economic Context (at enrollment)
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]

# ============================================================================
# PHASE B: SEMESTER 1 FEATURES (Time 1 - After First Semester)
# ============================================================================
# These features become available only after the first semester is complete.
# They provide academic performance signals but cannot be used for enrollment decisions.

SEMESTER_1_FEATURES = [
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)'
]

# ============================================================================
# PHASE C: SEMESTER 2 FEATURES (Time 2 - After Second Semester)
# ============================================================================
# These features become available only after the second semester is complete.
# By this point, dropout prediction may be too late for effective intervention.

SEMESTER_2_FEATURES = [
    'Curricular units 2nd sem (credited)',
    'Curricular units 2nd sem (enrolled)',
    'Curricular units 2nd sem (evaluations)',
    'Curricular units 2nd sem (approved)',
    'Curricular units 2nd sem (grade)',
    'Curricular units 2nd sem (without evaluations)'
]

# ============================================================================
# FEATURE PHASE COMBINATIONS
# ============================================================================

def get_features_for_phase(phase: str) -> list:
    """
    Get the appropriate feature set for a given modeling phase.

    Args:
        phase: One of 'enrollment', 'semester_1', 'semester_2'

    Returns:
        list: Feature names available at that phase

    Raises:
        ValueError: If phase is not recognized
    """
    phase = phase.lower()

    if phase == 'enrollment':
        return ENROLLMENT_FEATURES.copy()

    elif phase == 'semester_1':
        # Cumulative: enrollment + semester 1
        return ENROLLMENT_FEATURES + SEMESTER_1_FEATURES

    elif phase == 'semester_2':
        # Cumulative: enrollment + semester 1 + semester 2
        return ENROLLMENT_FEATURES + SEMESTER_1_FEATURES + SEMESTER_2_FEATURES

    else:
        raise ValueError(
            f"Invalid phase: {phase}. Must be one of: 'enrollment', 'semester_1', 'semester_2'"
        )


def validate_features(df, phase: str) -> bool:
    """
    Validate that all required features for a phase exist in the dataframe.

    Args:
        df: DataFrame to validate
        phase: Phase name ('enrollment', 'semester_1', 'semester_2')

    Returns:
        bool: True if all features exist, raises error otherwise

    Raises:
        ValueError: If required features are missing
    """
    required_features = get_features_for_phase(phase)
    missing_features = [f for f in required_features if f not in df.columns]

    if missing_features:
        raise ValueError(
            f"Missing features for phase '{phase}': {missing_features}"
        )

    return True


def get_phase_description(phase: str) -> dict:
    """
    Get metadata about a modeling phase.

    Args:
        phase: Phase name

    Returns:
        dict: Phase metadata including name, timing, feature count, and use case
    """
    descriptions = {
        'enrollment': {
            'name': 'Enrollment Phase',
            'timing': 'At admission (Time 0)',
            'feature_count': len(ENROLLMENT_FEATURES),
            'use_case': 'Identify at-risk students before classes begin for proactive support',
            'intervention_window': 'Maximum - Full academic year available'
        },
        'semester_1': {
            'name': 'Post-Semester 1 Phase',
            'timing': 'After first semester (Time 1)',
            'feature_count': len(ENROLLMENT_FEATURES + SEMESTER_1_FEATURES),
            'use_case': 'Refine risk assessment with early academic performance',
            'intervention_window': 'Medium - Second semester available'
        },
        'semester_2': {
            'name': 'Post-Semester 2 Phase',
            'timing': 'After second semester (Time 2)',
            'feature_count': len(ENROLLMENT_FEATURES + SEMESTER_1_FEATURES + SEMESTER_2_FEATURES),
            'use_case': 'Final risk assessment with full year of data',
            'intervention_window': 'Limited - May be too late for some students'
        }
    }

    return descriptions.get(phase.lower(), {})


# ============================================================================
# FEATURE CATEGORIES (For Analysis)
# ============================================================================

DEMOGRAPHIC_FEATURES = [
    'Marital Status', 'Nacionality', 'Age at enrollment', 'Gender', 'International'
]

ACADEMIC_BACKGROUND_FEATURES = [
    'Previous qualification', 'Previous qualification (grade)', 'Admission grade'
]

SOCIOECONOMIC_FEATURES = [
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Scholarship holder'
]

FINANCIAL_FEATURES = [
    'Debtor', 'Tuition fees up to date', 'Scholarship holder'
]

ECONOMIC_CONTEXT_FEATURES = [
    'Unemployment rate', 'Inflation rate', 'GDP'
]

PERFORMANCE_FEATURES_SEM1 = SEMESTER_1_FEATURES
PERFORMANCE_FEATURES_SEM2 = SEMESTER_2_FEATURES


if __name__ == "__main__":
    # Display feature phase information
    print("=" * 70)
    print("STUDENT RETENTION FORECASTING - FEATURE PHASES")
    print("=" * 70)

    for phase in ['enrollment', 'semester_1', 'semester_2']:
        desc = get_phase_description(phase)
        features = get_features_for_phase(phase)

        print(f"\n{desc['name'].upper()}")
        print(f"  Timing: {desc['timing']}")
        print(f"  Features: {desc['feature_count']}")
        print(f"  Use Case: {desc['use_case']}")
        print(f"  Intervention Window: {desc['intervention_window']}")

    print("\n" + "=" * 70)
    print("LEAKAGE PREVENTION")
    print("=" * 70)
    print("""
    ⚠️  CRITICAL: Do NOT use semester 1 or 2 features in enrollment models!

    WHY THIS MATTERS:
    - Enrollment models must predict dropout risk at admission
    - Using future performance data creates temporal leakage
    - This makes the model unusable in production (data won't exist yet)

    CORRECT APPROACH:
    - Phase A (Enrollment): Predict using only enrollment features
    - Phase B (Semester 1): Predict using enrollment + semester 1 features
    - Phase C (Semester 2): Predict using all available features
    """)
