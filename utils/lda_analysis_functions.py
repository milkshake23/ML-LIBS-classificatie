"""
LDA Analysis Functions for Machine Learning Models
Provides comprehensive Linear Discriminant Analysis (LDA) capabilities for 
dimensionality reduction with support for multiple classification algorithms.
"""

import numpy as np # type: ignore[import]
import pandas as pd # type: ignore[import]
import matplotlib.pyplot as plt # type: ignore[import]
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore[import]
from sklearn.model_selection import cross_val_score, StratifiedKFold # type: ignore[import]
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier # type: ignore[import]
from sklearn.tree import DecisionTreeClassifier # type: ignore[import]
from sklearn.ensemble import RandomForestClassifier # type: ignore[import]
from sklearn.linear_model import LogisticRegression # type: ignore[import]
from sklearn.naive_bayes import GaussianNB # type: ignore[import]
from sklearn.svm import SVC # type: ignore[import]
from sklearn.preprocessing import LabelEncoder # type: ignore[import]
import seaborn as sns # type: ignore[import]
import warnings # type: ignore[import]
warnings.filterwarnings('ignore')


class LDAAnalyzer:
    """
    Comprehensive LDA analysis for machine learning classification tasks.
    
    Supports multiple classifiers and provides detailed analysis of Linear 
    Discriminant Analysis effects on classification performance.
    """
    
    def __init__(self):
        """Initialize LDA analyzer with default classifier configurations."""
        self.supported_classifiers = {
            'KNN': {
                'class': KNeighborsClassifier,
                'param_grid': {'n_neighbors': range(1, 21)},
                'param_name': 'n_neighbors',
                'default_params': {'n_neighbors': 5}
            },
            'Decision Tree': {
                'class': DecisionTreeClassifier,
                'param_grid': {'max_depth': [3, 5, 7, 10, 15, None]},
                'param_name': 'max_depth',
                'default_params': {'max_depth': 10, 'random_state': 42}
            },
            'Random Forest': {
                'class': RandomForestClassifier,
                'param_grid': {'n_estimators': [50, 100, 200]},
                'param_name': 'n_estimators',
                'default_params': {'n_estimators': 100, 'random_state': 42}
            },
            'Logistic Regression': {
                'class': LogisticRegression,
                'param_grid': {'C': [0.1, 1.0, 10.0]},
                'param_name': 'C',
                'default_params': {'C': 1.0, 'random_state': 42, 'max_iter': 1000}
            },
            'Naive Bayes': {
                'class': GaussianNB,
                'param_grid': {'var_smoothing': [1e-9, 1e-8, 1e-7]},
                'param_name': 'var_smoothing',
                'default_params': {}
            },
            'SVM': {
                'class': SVC,
                'param_grid': {'C': [0.1, 1.0, 10.0]},
                'param_name': 'C',
                'default_params': {'C': 1.0, 'random_state': 42}
            }
        }
        
    def analyze_lda_components(self, X_train_scaled, y_train, plot_analysis=True):
        """
        Analyze LDA components and class separability.
        
        Parameters:
        -----------
        X_train_scaled : array-like
            Scaled training data
        y_train : array-like
            Training labels
        plot_analysis : bool, default=True
            Whether to create LDA visualization plots
            
        Returns:
        --------
        dict : Analysis results including LDA models and component information
        """
        print("Performing LDA analysis...")
        
        # Determine number of classes and maximum LDA components
        unique_classes = np.unique(y_train)
        n_classes = len(unique_classes)
        max_lda_components = n_classes - 1
        
        print(f"Number of classes: {n_classes}")
        print(f"Maximum LDA components: {max_lda_components}")
        print(f"Class labels: {unique_classes}")
        
        # Test different numbers of LDA components (1 to max)
        component_options = list(range(1, max_lda_components + 1))
        lda_models = {}
        
        for n_comp in component_options:
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            X_lda = lda.fit_transform(X_train_scaled, y_train)
            
            lda_models[n_comp] = {
                'model': lda,
                'transformed_data': X_lda,
                'explained_variance_ratio': lda.explained_variance_ratio_,
                'total_variance_explained': np.sum(lda.explained_variance_ratio_),
                'n_components': n_comp
            }
            
            print(f"LDA with {n_comp} component(s):")
            print(f"  - Explained variance ratio: {lda.explained_variance_ratio_}")
            print(f"  - Total variance explained: {np.sum(lda.explained_variance_ratio_):.4f}")
        
        if plot_analysis:
            self._plot_lda_analysis(lda_models, X_train_scaled, y_train, unique_classes)
        
        return {
            'lda_models': lda_models,
            'n_classes': n_classes,
            'max_components': max_lda_components,
            'unique_classes': unique_classes,
            'component_options': component_options
        }
    
    def _plot_lda_analysis(self, lda_models, X_train_scaled, y_train, unique_classes):
        """Create comprehensive LDA visualization plots."""
        max_components = max(lda_models.keys())
        
        if max_components == 1:
            # Single component analysis
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        else:
            # Multi-component analysis
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            ax1, ax2, ax3, ax4 = axes.flatten()
        
        # Plot 1: Explained variance by component
        n_components = list(lda_models.keys())
        total_variances = [lda_models[n]['total_variance_explained'] for n in n_components]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(n_components)))
        bars = ax1.bar(n_components, total_variances, color=colors, alpha=0.7)
        ax1.set_title('LDA Total Explained Variance by Components', fontweight='bold')
        ax1.set_xlabel('Number of LDA Components')
        ax1.set_ylabel('Total Explained Variance')
        ax1.grid(True, alpha=0.3)
        
        # Add variance values on bars
        for bar, variance in zip(bars, total_variances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{variance:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Class distribution in LDA space
        if max_components >= 2:
            # 2D scatter plot for 2+ components
            lda_2d = lda_models[2] if 2 in lda_models else lda_models[max_components]
            X_lda_2d = lda_2d['transformed_data']
            
            # Ensure we have proper label encoding
            if isinstance(y_train[0], str):
                label_encoder = LabelEncoder()
                y_numeric = label_encoder.fit_transform(y_train)
                class_names = label_encoder.classes_
            else:
                y_numeric = y_train
                class_names = unique_classes
            
            # Create scatter plot
            colors_scatter = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
            
            for i, class_name in enumerate(class_names):
                mask = y_numeric == i
                n_samples = np.sum(mask)
                
                if n_samples > 0:
                    x_coords = X_lda_2d[mask, 0]
                    if X_lda_2d.shape[1] > 1:
                        y_coords = X_lda_2d[mask, 1]
                        ax2.scatter(x_coords, y_coords,
                                   label=f'{class_name} (n={n_samples})',
                                   alpha=0.7, s=50, c=[colors_scatter[i]], 
                                   edgecolors='black', linewidth=0.5)
                        
                        # Add class centroid
                        centroid_x = np.mean(x_coords)
                        centroid_y = np.mean(y_coords)
                        ax2.scatter(centroid_x, centroid_y, c='black', s=200, 
                                   marker='x', linewidth=3, alpha=0.8)
                        ax2.annotate(class_name, (centroid_x, centroid_y), 
                                    xytext=(5, 5), textcoords='offset points',
                                    fontweight='bold', fontsize=10)
            
            variance_1 = lda_2d['explained_variance_ratio'][0]
            variance_2 = lda_2d['explained_variance_ratio'][1] if len(lda_2d['explained_variance_ratio']) > 1 else 0
            
            ax2.set_xlabel(f'LD1 ({variance_1:.3f} var. explained)')
            ax2.set_ylabel(f'LD2 ({variance_2:.3f} var. explained)')
            ax2.set_title('LDA - Class Separation in 2D Discriminant Space')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
        
        else:
            # 1D histogram for single component
            lda_1d = lda_models[1]
            X_lda_1d = lda_1d['transformed_data']
            
            if isinstance(y_train[0], str):
                label_encoder = LabelEncoder()
                y_numeric = label_encoder.fit_transform(y_train)
                class_names = label_encoder.classes_
            else:
                y_numeric = y_train
                class_names = unique_classes
            
            colors_hist = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
            
            for i, class_name in enumerate(class_names):
                mask = y_numeric == i
                if np.sum(mask) > 0:
                    ax2.hist(X_lda_1d[mask, 0], alpha=0.7, label=class_name, 
                            bins=20, color=colors_hist[i])
            
            variance_1 = lda_1d['explained_variance_ratio'][0]
            ax2.set_xlabel(f'LD1 ({variance_1:.3f} var. explained)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('LDA - Class Distribution on Single Discriminant')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Additional plots for multi-component case
        if max_components >= 2 and max_components > 1:
            # Plot 3: Individual component explained variance
            if max_components >= 2:
                max_lda = lda_models[max_components]
                component_names = [f'LD{i+1}' for i in range(max_components)]
                individual_variances = max_lda['explained_variance_ratio']
                
                ax3.bar(component_names, individual_variances, 
                       color=plt.cm.plasma(np.linspace(0, 1, max_components)), alpha=0.7)
                ax3.set_title('Individual LDA Component Explained Variance')
                ax3.set_xlabel('Linear Discriminant')
                ax3.set_ylabel('Explained Variance Ratio')
                ax3.grid(True, alpha=0.3)
                
                # Add values on bars
                for i, (comp, var) in enumerate(zip(component_names, individual_variances)):
                    ax3.text(i, var + 0.01, f'{var:.3f}', 
                            ha='center', va='bottom', fontweight='bold')
            
            # Plot 4: Cumulative explained variance
            if max_components >= 2:
                cumulative_variances = []
                for n_comp in range(1, max_components + 1):
                    cumulative_variances.append(lda_models[n_comp]['total_variance_explained'])
                
                ax4.plot(range(1, max_components + 1), cumulative_variances, 
                        marker='o', linewidth=2, markersize=8, color='red')
                ax4.fill_between(range(1, max_components + 1), cumulative_variances, 
                                alpha=0.3, color='red')
                ax4.set_title('Cumulative Explained Variance')
                ax4.set_xlabel('Number of LDA Components')
                ax4.set_ylabel('Cumulative Explained Variance')
                ax4.grid(True, alpha=0.3)
                ax4.set_ylim(0, 1.1)
                
                # Add percentage labels
                for i, var in enumerate(cumulative_variances):
                    ax4.text(i + 1, var + 0.02, f'{var:.2%}', 
                            ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def optimize_classifier_hyperparameters(self, classifier_name, X_train, y_train, cv_folds=5):
        """
        Optimize hyperparameters for a specific classifier using cross-validation.
        
        Parameters:
        -----------
        classifier_name : str
            Name of the classifier to optimize
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        cv_folds : int, default=5
            Number of cross-validation folds
            
        Returns:
        --------
        dict : Optimization results including best parameters and scores
        """
        if classifier_name not in self.supported_classifiers:
            raise ValueError(f"Classifier '{classifier_name}' not supported. "
                           f"Available: {list(self.supported_classifiers.keys())}")
        
        classifier_config = self.supported_classifiers[classifier_name]
        ClassifierClass = classifier_config['class']
        param_grid = classifier_config['param_grid']
        param_name = classifier_config['param_name']
        
        print(f"Optimizing {classifier_name} hyperparameters...")
        
        cv_scores = []
        param_values = list(param_grid.values())[0]
        
        for param_value in param_values:
            classifier_params = classifier_config['default_params'].copy()
            classifier_params[param_name] = param_value
            
            classifier = ClassifierClass(**classifier_params)
            scores = cross_val_score(classifier, X_train, y_train, 
                                   cv=cv_folds, scoring='accuracy')
            cv_scores.append(scores.mean())
        
        optimal_idx = np.argmax(cv_scores)
        optimal_param_value = param_values[optimal_idx]
        optimal_score = cv_scores[optimal_idx]
        
        print(f"Optimal {param_name}: {optimal_param_value} (CV Score: {optimal_score:.4f})")
        
        return {
            'optimal_param_value': optimal_param_value,
            'optimal_score': optimal_score,
            'param_name': param_name,
            'cv_scores': cv_scores,
            'param_values': param_values
        }
    
    def evaluate_lda_performance(self, X_train_scaled, X_test_scaled, y_train, y_test, 
                                classifier_name, component_options=2, 
                                label_encoder=None, cv_folds=5):
        """
        Evaluate classifier performance with different numbers of LDA components.
        
        Parameters:
        -----------
        X_train_scaled : array-like
            Scaled training features
        X_test_scaled : array-like
            Scaled test features
        y_train : array-like
            Training labels
        y_test : array-like
            Test labels
        classifier_name : str
            Name of classifier to use
        component_options : list, optional
            List of component counts to test
        label_encoder : LabelEncoder, optional
            Label encoder for classification reports
        cv_folds : int, default=5
            Cross-validation folds
            
        Returns:
        --------
        dict : Comprehensive evaluation results
        """
        # Perform LDA analysis first if component_options not provided
        if component_options is None:
            lda_analysis = self.analyze_lda_components(X_train_scaled, y_train, plot_analysis=False)
            component_options = lda_analysis['component_options']
            max_components = lda_analysis['max_components']
        else:
            max_components = max(component_options)
        
        print(f"Testing {classifier_name} with different LDA component numbers...")
        print(f"Component options: {component_options}")
        
        lda_results = []
        
        for n_comp in component_options:
            print(f"\nEvaluating {n_comp} components...")
            
            # Apply LDA
            lda = LinearDiscriminantAnalysis(n_components=n_comp)
            X_train_lda = lda.fit_transform(X_train_scaled, y_train)
            X_test_lda = lda.transform(X_test_scaled)
            
            print(f"LDA transformed shape: {X_train_lda.shape}")
            print(f"Explained variance ratio: {lda.explained_variance_ratio_}")
            print(f"Total variance explained: {np.sum(lda.explained_variance_ratio_):.4f}")
            
            # Optimize hyperparameters for LDA features
            optimization_results = self.optimize_classifier_hyperparameters(
                classifier_name, X_train_lda, y_train, cv_folds
            )
            
            # Train final model with optimal parameters
            classifier_config = self.supported_classifiers[classifier_name]
            optimal_params = classifier_config['default_params'].copy()
            optimal_params[optimization_results['param_name']] = optimization_results['optimal_param_value']
            
            classifier = classifier_config['class'](**optimal_params)
            classifier.fit(X_train_lda, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test_lda)
            accuracy = accuracy_score(y_test, y_pred)
            
            result = {
                'n_components': n_comp,
                'lda_model': lda,
                'classifier': classifier,
                'optimal_params': optimal_params,
                'cv_score': optimization_results['optimal_score'],
                'test_accuracy': accuracy,
                'y_pred': y_pred,
                'explained_variance_ratio': lda.explained_variance_ratio_,
                'total_variance_explained': np.sum(lda.explained_variance_ratio_),
                'X_train_lda': X_train_lda,
                'X_test_lda': X_test_lda
            }
            
            lda_results.append(result)
            print(f"{classifier_name} with {n_comp} LDA components: {accuracy:.4f} accuracy")
        
        # Find best configuration
        best_idx = np.argmax([r['test_accuracy'] for r in lda_results])
        best_result = lda_results[best_idx]
        
        print(f"\nBest LDA configuration: {best_result['n_components']} components")
        print(f"Best accuracy: {best_result['test_accuracy']:.4f}")
        print(f"Best variance explained: {best_result['total_variance_explained']:.4f}")
        
        # Generate detailed report for best configuration
        if label_encoder is not None:
            print(f"\nClassification Report ({classifier_name} - {best_result['n_components']} components):")
            print(classification_report(y_test, best_result['y_pred'], 
                                      target_names=label_encoder.classes_))
        
        return {
            'classifier_name': classifier_name,
            'lda_results': lda_results,
            'best_result': best_result,
            'best_n_components': best_result['n_components'],
            'best_accuracy': best_result['test_accuracy'],
            'max_possible_components': max_components
        }
    
    def plot_confusion_matrix(self, y_test, y_pred, classifier_name, n_components, 
                            label_encoder=None, figsize=(8, 6)):
        """Plot confusion matrix for LDA classification results."""
        plt.figure(figsize=figsize)
        cm = confusion_matrix(y_test, y_pred)
        
        labels = label_encoder.classes_ if label_encoder else None
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=labels, yticklabels=labels)
        
        plt.title(f'Confusion Matrix - {classifier_name} on LDA Features ({n_components} components)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def plot_lda_discriminant_space(self, X_lda, y, class_names=None, n_components=2, 
                                   explained_variance_ratio=None, figsize=(10, 8)):
        """
        Plot data distribution in LDA discriminant space.
        
        Parameters:
        -----------
        X_lda : array-like
            LDA transformed data
        y : array-like
            Class labels
        class_names : list, optional
            Names of classes
        n_components : int, default=2
            Number of components to plot (1 or 2)
        explained_variance_ratio : array-like, optional
            Explained variance ratio for each component
        figsize : tuple, default=(10, 8)
            Figure size
        """
        plt.figure(figsize=figsize)
        
        # Handle string labels
        if isinstance(y[0], str):
            label_encoder = LabelEncoder()
            y_numeric = label_encoder.fit_transform(y)
            if class_names is None:
                class_names = label_encoder.classes_
        else:
            y_numeric = y
            if class_names is None:
                class_names = [f'Class {i}' for i in np.unique(y_numeric)]
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(class_names)))
        
        if n_components == 1 or X_lda.shape[1] == 1:
            # 1D histogram
            for i, class_name in enumerate(class_names):
                mask = y_numeric == i
                if np.sum(mask) > 0:
                    plt.hist(X_lda[mask, 0], alpha=0.7, label=class_name, 
                            bins=30, color=colors[i])
            
            var_label = f" ({explained_variance_ratio[0]:.3f})" if explained_variance_ratio is not None else ""
            plt.xlabel(f'LD1{var_label}')
            plt.ylabel('Frequency')
            plt.title('Class Distribution in LDA Space')
            
        else:
            # 2D scatter plot
            for i, class_name in enumerate(class_names):
                mask = y_numeric == i
                n_samples = np.sum(mask)
                
                if n_samples > 0:
                    plt.scatter(X_lda[mask, 0], X_lda[mask, 1],
                               label=f'{class_name} (n={n_samples})',
                               alpha=0.7, s=50, c=[colors[i]], 
                               edgecolors='black', linewidth=0.5)
                    
                    # Add class centroid
                    centroid_x = np.mean(X_lda[mask, 0])
                    centroid_y = np.mean(X_lda[mask, 1])
                    plt.scatter(centroid_x, centroid_y, c='black', s=200, 
                               marker='x', linewidth=3, alpha=0.8)
                    plt.annotate(class_name, (centroid_x, centroid_y), 
                                xytext=(5, 5), textcoords='offset points',
                                fontweight='bold', fontsize=10)
            
            var1_label = f" ({explained_variance_ratio[0]:.3f})" if explained_variance_ratio is not None else ""
            var2_label = f" ({explained_variance_ratio[1]:.3f})" if explained_variance_ratio is not None and len(explained_variance_ratio) > 1 else ""
            
            plt.xlabel(f'LD1{var1_label}')
            plt.ylabel(f'LD2{var2_label}')
            plt.title('Class Separation in LDA Discriminant Space')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def compare_multiple_classifiers(self, X_train_scaled, X_test_scaled, y_train, y_test,
                                   classifiers=None, label_encoder=None, cv_folds=5):
        """
        Compare multiple classifiers with LDA dimensionality reduction.
        
        Parameters:
        -----------
        X_train_scaled : array-like
            Scaled training features
        X_test_scaled : array-like
            Scaled test features
        y_train : array-like
            Training labels
        y_test : array-like
            Test labels
        classifiers : list, optional
            List of classifier names to compare
        label_encoder : LabelEncoder, optional
            Label encoder for reports
        cv_folds : int, default=5
            Cross-validation folds
            
        Returns:
        --------
        dict : Comparison results for all classifiers
        """
        if classifiers is None:
            classifiers = list(self.supported_classifiers.keys())
        
        # Perform LDA analysis once
        print("Performing initial LDA analysis...")
        lda_analysis = self.analyze_lda_components(X_train_scaled, y_train)
        component_options = lda_analysis['component_options']
        
        # Evaluate each classifier
        comparison_results = {}
        for classifier_name in classifiers:
            print(f"\n{'='*60}")
            print(f"Evaluating {classifier_name} with LDA")
            print(f"{'='*60}")
            
            results = self.evaluate_lda_performance(
                X_train_scaled, X_test_scaled, y_train, y_test,
                classifier_name, component_options, label_encoder, cv_folds
            )
            
            comparison_results[classifier_name] = results
        
        # Create comparison summary
        summary_data = []
        for classifier_name, results in comparison_results.items():
            best_result = results['best_result']
            summary_data.append({
                'Classifier': classifier_name,
                'Best_Components': best_result['n_components'],
                'Test_Accuracy': best_result['test_accuracy'],
                'CV_Score': best_result['cv_score'],
                'Variance_Explained': best_result['total_variance_explained'],
                'Max_Possible_Components': results['max_possible_components'],
                'Optimal_Params': str(best_result['optimal_params'])
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        
        print(f"\n{'='*80}")
        print("LDA CLASSIFIER COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(summary_df.to_string(index=False))
        
        return {
            'individual_results': comparison_results,
            'summary': summary_df,
            'lda_analysis': lda_analysis
        }
    
    def plot_comparison_results(self, comparison_results, figsize=(15, 10)):
        """Create comprehensive visualization of classifier comparison results."""
        summary_df = comparison_results['summary']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        classifiers = summary_df['Classifier']
        colors = plt.cm.Set3(np.linspace(0, 1, len(classifiers)))
        
        # Test accuracy comparison
        accuracies = summary_df['Test_Accuracy']
        bars1 = ax1.bar(classifiers, accuracies, color=colors, alpha=0.7)
        ax1.set_title('LDA + Classifier Test Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis='y', alpha=0.3)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Add accuracy values on bars
        for bar, accuracy in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Component count comparison
        components = summary_df['Best_Components']
        bars2 = ax2.bar(classifiers, components, color=colors, alpha=0.7)
        ax2.set_title('Optimal LDA Components Used', fontweight='bold')
        ax2.set_ylabel('Number of Components')
        ax2.grid(axis='y', alpha=0.3)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add component counts on bars
        for bar, comp_count in zip(bars2, components):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                     f'{comp_count}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation score comparison
        cv_scores = summary_df['CV_Score']
        bars3 = ax3.bar(classifiers, cv_scores, color=colors, alpha=0.7)
        ax3.set_title('Cross-Validation Scores', fontweight='bold')
        ax3.set_ylabel('CV Score')
        ax3.grid(axis='y', alpha=0.3)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Variance explained comparison
        variance_explained = summary_df['Variance_Explained']
        bars4 = ax4.bar(classifiers, variance_explained, color=colors, alpha=0.7)
        ax4.set_title('Variance Explained by LDA', fontweight='bold')
        ax4.set_ylabel('Variance Explained')
        ax4.grid(axis='y', alpha=0.3)
        plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()


# Convenience function for easy usage
def quick_lda_analysis(X_train_scaled, X_test_scaled, y_train, y_test, 
                      classifier_name='KNN', label_encoder=None):
    """
    Quick LDA analysis for a single classifier.
    
    Parameters:
    -----------
    X_train_scaled : array-like
        Scaled training features
    X_test_scaled : array-like  
        Scaled test features
    y_train : array-like
        Training labels
    y_test : array-like
        Test labels
    classifier_name : str, default='KNN'
        Classifier to analyze
    label_encoder : LabelEncoder, optional
        Label encoder for reports
        
    Returns:
    --------
    dict : Analysis results
    """
    analyzer = LDAAnalyzer()
    
    results = analyzer.evaluate_lda_performance(
        X_train_scaled, X_test_scaled, y_train, y_test,
        classifier_name, label_encoder=label_encoder
    )
    
    # Plot confusion matrix for best result
    best_result = results['best_result']
    analyzer.plot_confusion_matrix(
        y_test, best_result['y_pred'], classifier_name,
        best_result['n_components'], label_encoder
    )
    
    # Plot LDA discriminant space
    analyzer.plot_lda_discriminant_space(
        best_result['X_test_lda'], y_test,
        class_names=label_encoder.classes_ if label_encoder else None,
        n_components=best_result['n_components'],
        explained_variance_ratio=best_result['explained_variance_ratio']
    )
    
    return results