"""
Production Data Preprocessing Module for Diabetes Risk Prediction

This module contains the validated preprocessing pipeline extracted from the
interactive notebook development. It includes all the methods that were tested
and validated in the 02_data_processing.ipynb notebook.

Key Features:
- Target variable cleanup (drops additional targets)
- Domain-specific outlier treatment
- Categorical encoding (ordinal + one-hot)
- Feature scaling and transformation
- Mutual information feature selection
- Stratified train/validation/test splits
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, Tuple, List, Optional, Any, Union
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiabetesDataPreprocessor:
    """
    Complete preprocessing pipeline for diabetes risk prediction dataset.
    
    This class implements the validated preprocessing steps from the notebook:
    1. Data loading and validation
    2. Target variable cleanup
    3. Outlier treatment
    4. Categorical encoding
    5. Feature scaling and transformation
    6. Feature selection
    7. Train/validation/test splitting
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Configuration from validated notebook
        self.target_column = 'diagnosed_diabetes'
        self.additional_targets = ['diabetes_risk_score', 'diabetes_stage']
        
        # Categorical feature configuration
        self.categorical_features = [
            'gender', 'ethnicity', 'education_level', 
            'income_level', 'employment_status', 'smoking_status'
        ]
        
        self.ordinal_features = {
            'education_level': ['No formal', 'Highschool', 'Graduate', 'Postgraduate'],
            'income_level': ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
        }
        
        self.nominal_features = ['gender', 'ethnicity', 'employment_status', 'smoking_status']
        
        # Outlier treatment configuration
        self.outlier_caps = {
            'alcohol_consumption_per_week': (0, 14),
            'physical_activity_minutes_per_week': (0, 2000),
            'sleep_hours_per_day': (4, 12),
            'screen_time_hours_per_day': (0, 16),
            'bmi': (15, 50)
        }
        
        self.outlier_preserve = ['cardiovascular_history']
        
        # High correlation pairs to remove
        self.corr_removals = {
            'glucose_postprandial': 'hba1c',
            'ldl_cholesterol': 'cholesterol_total',
            'waist_to_hip_ratio': 'bmi'
        }
        
        # Processing artifacts
        self.scaler = StandardScaler()
        self.categorical_mappings = {}
        self.transformation_log = {}
        self.feature_selection_info = {}
        self.preprocessing_metadata = {}
        
    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """
        Load raw data and perform initial validation.
        
        Args:
            filepath: Path to the raw CSV file
            
        Returns:
            Validated DataFrame
        """
        logger.info(f"📂 Loading raw diabetes dataset from: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        
        # Validation checks
        expected_shape = (100000, 31)
        logger.info(f"✅ Dataset loaded: {df.shape}")
        logger.info(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Validate structure
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
            
        # Check data quality
        missing_count = df.isnull().sum().sum()
        duplicate_count = df.duplicated().sum()
        
        logger.info(f"🔍 Data quality: {missing_count} missing values, {duplicate_count} duplicates")
        
        if missing_count > 0:
            logger.warning(f"⚠️ {missing_count} missing values detected")
        if duplicate_count > 0:
            logger.warning(f"⚠️ {duplicate_count} duplicate rows detected")
            
        return df
    
    def cleanup_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove additional target variables to focus on single target.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional targets removed
        """
        logger.info("🗑️ Dropping additional target variables")
        
        # Drop additional target columns
        df_clean = df.drop(columns=self.additional_targets, errors='ignore')
        
        dropped_count = len(self.additional_targets)
        logger.info(f"✅ Dropped {dropped_count} additional target columns")
        logger.info(f"📏 New shape: {df_clean.shape}")
        
        return df_clean
    
    def treat_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply domain-specific outlier treatment.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (treated DataFrame, treatment log)
        """
        logger.info("🚨 Applying outlier treatment")
        
        df_treated = df.copy()
        treatment_log = {}
        
        # Apply winsorization to specified features
        for feature, (min_cap, max_cap) in self.outlier_caps.items():
            if feature in df_treated.columns:
                outliers_before = len(df_treated[(df_treated[feature] < min_cap) | 
                                                (df_treated[feature] > max_cap)])
                
                original_min, original_max = df_treated[feature].min(), df_treated[feature].max()
                df_treated[feature] = df_treated[feature].clip(lower=min_cap, upper=max_cap)
                
                outliers_after = len(df_treated[(df_treated[feature] < min_cap) | 
                                               (df_treated[feature] > max_cap)])
                
                treatment_log[feature] = {
                    'method': 'winsorization',
                    'min_cap': min_cap,
                    'max_cap': max_cap,
                    'original_range': (original_min, original_max),
                    'outliers_before': outliers_before,
                    'outliers_after': outliers_after
                }
                
                logger.info(f"   • {feature}: {outliers_before} → {outliers_after} outliers")
        
        # Special handling for diet_score
        if 'diet_score' in df_treated.columns:
            Q1 = df_treated['diet_score'].quantile(0.05)
            Q3 = df_treated['diet_score'].quantile(0.95)
            
            outliers_before = len(df_treated[(df_treated['diet_score'] < Q1) | 
                                           (df_treated['diet_score'] > Q3)])
            df_treated['diet_score'] = df_treated['diet_score'].clip(lower=Q1, upper=Q3)
            
            treatment_log['diet_score'] = {
                'method': 'percentile_clipping',
                'lower_percentile': Q1,
                'upper_percentile': Q3,
                'outliers_before': outliers_before,
                'outliers_after': 0
            }
            
            logger.info(f"   • diet_score: {outliers_before} → 0 outliers (5-95% clipping)")
        
        logger.info(f"✅ Outlier treatment completed for {len(treatment_log)} features")
        return df_treated, treatment_log
    
    def encode_categorical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical features using ordinal and one-hot encoding.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (encoded DataFrame, encoding mappings)
        """
        logger.info("🔢 Encoding categorical features")
        
        df_encoded = df.copy()
        categorical_mappings = {}
        one_hot_columns = []
        
        # Label encode ordinal features
        logger.info("🔢 Applying label encoding to ordinal features")
        for feature, order in self.ordinal_features.items():
            if feature in df_encoded.columns:
                unique_vals = df_encoded[feature].unique()
                
                # Create mapping based on logical order
                if set(unique_vals).issubset(set(order)):
                    mapping = {val: idx for idx, val in enumerate(order) if val in unique_vals}
                else:
                    mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}
                
                df_encoded[feature] = df_encoded[feature].map(mapping)
                categorical_mappings[feature] = mapping
                
                logger.info(f"   • {feature}: {len(mapping)} classes")
        
        # One-hot encode nominal features
        logger.info("🏷️ Applying one-hot encoding to nominal features")
        for feature in self.nominal_features:
            if feature in df_encoded.columns:
                unique_vals = df_encoded[feature].unique()
                
                # Create one-hot encoded columns
                one_hot = pd.get_dummies(df_encoded[feature], prefix=feature, dtype=int)
                
                # Add to main dataframe
                df_encoded = pd.concat([df_encoded, one_hot], axis=1)
                one_hot_columns.extend(one_hot.columns.tolist())
                
                # Drop original column
                df_encoded = df_encoded.drop(columns=[feature])
                
                # Store mapping
                categorical_mappings[feature] = {
                    'type': 'one_hot',
                    'columns': one_hot.columns.tolist(),
                    'original_values': unique_vals.tolist()
                }
                
                logger.info(f"   • {feature}: {len(unique_vals)} → {len(one_hot.columns)} columns")
        
        logger.info(f"📊 Encoding complete: {df_encoded.shape[1]} total columns")
        logger.info(f"🏷️ One-hot columns: {len(one_hot_columns)}")
        
        return df_encoded, categorical_mappings
    
    def scale_and_transform_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply scaling and transformations to numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (scaled DataFrame, transformation log)
        """
        logger.info("📊 Scaling and transforming features")
        
        df_transformed = df.copy()
        transformation_log = {}
        
        # Identify numerical features
        one_hot_cols = [col for col in df.columns 
                       if any(col.startswith(f'{cat}_') for cat in self.nominal_features)]
        
        numerical_features = [col for col in df.columns 
                            if col not in one_hot_cols + [self.target_column]
                            and df[col].dtype in ['int64', 'float64']]
        
        logger.info(f"🔢 Identified {len(numerical_features)} numerical features")
        
        # Apply log transformation to skewed features
        skewed_features = []
        for col in numerical_features:
            skew_val = df_transformed[col].skew()
            if abs(skew_val) > 1.0:
                skewed_features.append(col)
        
        logger.info(f"📈 Applying log transformation to {len(skewed_features)} skewed features")
        
        for feature in skewed_features:
            original_skew = df_transformed[feature].skew()
            
            # Ensure positive values
            min_val = df_transformed[feature].min()
            if min_val <= 0:
                shift_val = abs(min_val) + 1
                df_transformed[feature] = df_transformed[feature] + shift_val
                transformation_log[feature] = {'type': 'log_shifted', 'shift': shift_val}
            else:
                transformation_log[feature] = {'type': 'log', 'shift': 0}
            
            # Apply log transformation
            df_transformed[feature] = np.log1p(df_transformed[feature])
            new_skew = df_transformed[feature].skew()
            
            transformation_log[feature].update({
                'original_skew': original_skew,
                'new_skew': new_skew
            })
            
            logger.info(f"   • {feature}: {original_skew:.2f} → {new_skew:.2f}")
        
        # Apply standard scaling
        logger.info("⚖️ Applying StandardScaler")
        
        df_scaled = df_transformed.copy()
        numerical_data = df_scaled[numerical_features]
        
        scaled_data = self.scaler.fit_transform(numerical_data)
        df_scaled[numerical_features] = scaled_data
        
        logger.info("✅ Scaling completed (mean ≈ 0, std ≈ 1)")
        
        return df_scaled, transformation_log
    
    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Perform feature selection using correlation analysis and mutual information.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (selected features DataFrame, selection info)
        """
        logger.info("🎯 Performing feature selection")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        logger.info(f"📊 Starting with {X.shape[1]} features")
        
        # Remove highly correlated features
        features_to_remove = []
        for remove_feat, keep_feat in self.corr_removals.items():
            if remove_feat in X.columns:
                features_to_remove.append(remove_feat)
                logger.info(f"   ❌ Removing {remove_feat} (keeping {keep_feat})")
        
        X_filtered = X.drop(columns=features_to_remove)
        logger.info(f"📊 After correlation removal: {X_filtered.shape[1]} features")
        
        # Mutual information feature selection
        logger.info("📈 Calculating mutual information scores")
        mi_scores = mutual_info_classif(X_filtered, y, random_state=self.random_state)
        
        mi_df = pd.DataFrame({
            'Feature': X_filtered.columns,
            'MI_Score': mi_scores
        }).sort_values('MI_Score', ascending=False)
        
        # Select top features (80% or at least 20)
        n_features_to_keep = max(20, int(0.8 * len(X_filtered.columns)))
        top_features = mi_df.head(n_features_to_keep)['Feature'].tolist()
        
        X_selected = X_filtered[top_features]
        
        logger.info(f"🎯 Selected top {len(top_features)} features")
        logger.info(f"📉 Total reduction: {((X.shape[1] - X_selected.shape[1]) / X.shape[1] * 100):.1f}%")
        
        # Create feature selection info
        selection_info = {
            'selected_features': top_features,
            'removed_correlated': features_to_remove,
            'mi_scores': mi_df.to_dict('records'),
            'selection_method': 'mutual_information',
            'n_features_selected': len(top_features)
        }
        
        # Recombine with target
        df_selected = pd.concat([X_selected, y], axis=1)
        
        return df_selected, selection_info
    
    def create_train_val_test_splits(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create stratified train/validation/test splits.
        
        Args:
            df: Processed DataFrame
            
        Returns:
            Dictionary containing all splits
        """
        logger.info("✂️ Creating train/validation/test splits")
        
        # Separate features and target
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        
        logger.info(f"📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"🎯 Target distribution: {y.value_counts(normalize=True).round(3).to_dict()}")
        
        # First split: 70% train, 30% temp
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=self.random_state, stratify=y
        )
        
        # Second split: 15% validation, 15% test
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=self.random_state, stratify=y_temp
        )
        
        logger.info(f"🚂 Training: {len(X_train):,} samples (70.0%)")
        logger.info(f"✅ Validation: {len(X_val):,} samples (15.0%)")
        logger.info(f"🧪 Test: {len(X_test):,} samples (15.0%)")
        
        # Verify stratification
        train_dist = y_train.value_counts(normalize=True).sort_index()
        val_dist = y_val.value_counts(normalize=True).sort_index()
        test_dist = y_test.value_counts(normalize=True).sort_index()
        
        max_diff = max((train_dist - val_dist).abs().max(), (train_dist - test_dist).abs().max())
        if max_diff < 0.01:
            logger.info("✅ Stratification successful")
        else:
            logger.warning(f"⚠️ Stratification difference: {max_diff:.4f}")
        
        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test
        }
    
    def export_processed_data(self, datasets: Dict, output_dir: str = '../data/processed') -> None:
        """
        Export all processed datasets and artifacts.
        
        Args:
            datasets: Dictionary containing all dataset splits
            output_dir: Output directory path
        """
        logger.info("💾 Exporting processed data")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Export datasets
        for name, data in datasets.items():
            filepath = f'{output_dir}/{name}.csv'
            data.to_csv(filepath, index=False)
            logger.info(f"   ✅ {name}: {filepath} ({data.shape})")
        
        # Export complete processed dataset
        X_complete = pd.concat([datasets['X_train'], datasets['X_val'], datasets['X_test']])
        y_complete = pd.concat([datasets['y_train'], datasets['y_val'], datasets['y_test']])
        complete_processed = pd.concat([X_complete, y_complete], axis=1)
        complete_processed.to_csv(f'{output_dir}/diabetes_processed_complete.csv', index=False)
        
        # Export preprocessing artifacts
        os.makedirs('../models', exist_ok=True)
        joblib.dump(self.scaler, '../models/feature_scaler.pkl')
        joblib.dump(self.categorical_mappings, '../models/categorical_mappings.pkl')
        joblib.dump(self.transformation_log, '../models/transformation_log.pkl')
        joblib.dump(self.feature_selection_info, '../models/feature_selection_info.pkl')
        
        # Create and export metadata
        self.preprocessing_metadata = {
            'processing_date': '2025-12-15',
            'target_column': self.target_column,
            'categorical_mappings': self.categorical_mappings,
            'transformation_log': self.transformation_log,
            'scaler_type': 'StandardScaler',
            'split_sizes': {
                'train': len(datasets['X_train']),
                'validation': len(datasets['X_val']),
                'test': len(datasets['X_test'])
            },
            'feature_selection_info': self.feature_selection_info,
            'random_state': self.random_state
        }
        
        with open(f'{output_dir}/preprocessing_metadata.json', 'w') as f:
            json.dump(self.preprocessing_metadata, f, indent=2, default=str)
        
        logger.info("✅ All artifacts exported successfully")
    
    def fit_transform(self, filepath: str, output_dir: str = '../data/processed') -> Dict[str, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to raw data file
            output_dir: Output directory for processed data
            
        Returns:
            Dictionary containing all processed datasets
        """
        logger.info("🚀 Starting complete preprocessing pipeline")
        
        # 1. Load and validate data
        df = self.load_and_validate_data(filepath)
        
        # 2. Cleanup target variables
        df = self.cleanup_target_variables(df)
        
        # 3. Treat outliers
        df, self.transformation_log = self.treat_outliers(df)
        
        # 4. Encode categorical features
        df, self.categorical_mappings = self.encode_categorical_features(df)
        
        # 5. Scale and transform features
        df, transform_log = self.scale_and_transform_features(df)
        self.transformation_log.update(transform_log)
        
        # 6. Select features
        df, self.feature_selection_info = self.select_features(df)
        
        # 7. Create splits
        datasets = self.create_train_val_test_splits(df)
        
        # 8. Export everything
        self.export_processed_data(datasets, output_dir)
        
        logger.info("🎉 Preprocessing pipeline completed successfully!")
        
        return datasets


def main():
    """Example usage of the preprocessing pipeline."""
    
    # Initialize preprocessor
    preprocessor = DiabetesDataPreprocessor(random_state=42)
    
    # Run complete pipeline
    datasets = preprocessor.fit_transform(
        filepath='../data/raw/diabetes_dataset.csv',
        output_dir='../data/processed'
    )
    
    print("\n📊 PREPROCESSING SUMMARY:")
    print(f"   • Training samples: {len(datasets['X_train']):,}")
    print(f"   • Validation samples: {len(datasets['X_val']):,}")
    print(f"   • Test samples: {len(datasets['X_test']):,}")
    print(f"   • Features: {datasets['X_train'].shape[1]}")
    print("   • Ready for model training! 🚀")


if __name__ == "__main__":
    main()