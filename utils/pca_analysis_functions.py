import numpy as np #type: ignore[import]
import pandas as pd #type: ignore[import]
import matplotlib.pyplot as plt #type: ignore[import]
import seaborn as sns #type: ignore[import]
from sklearn.decomposition import PCA #type: ignore[import]  
from sklearn.model_selection import cross_val_score, GridSearchCV #type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier #type: ignore[import]
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score #type: ignore[import]
from sklearn.svm import SVC #type: ignore[import]
from sklearn.ensemble import RandomForestClassifier #type: ignore[import]
from sklearn.tree import DecisionTreeClassifier #type: ignore[import]
from sklearn.naive_bayes import GaussianNB #type: ignore[import]
from sklearn.linear_model import LogisticRegression #type: ignore[import]
import warnings #type: ignore[import]
warnings.filterwarnings('ignore')

class PCAAnalyzer:
    """
    Comprehensive PCA analysis tool for dimensionality reduction and classification
    """
    
    def __init__(self):
        self.pca_model = None
        self.explained_variance_ratio_ = None
        self.components_ = None
        self.n_components_95 = None
        # self.n_components_99 = None
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
            },
            'LR': {
                'class': LogisticRegression,
                'param_grid': {'C': [0.1, 1.0, 10.0]},
                'param_name': 'C',
                'default_params': {'C': 1.0, 'random_state': 42, 'max_iter': 1000}
            }
        }

    def fit_pca_analysis(self, X_train_scaled, variance_thresholds=[0.95]):
        """
        Perform comprehensive PCA analysis to determine optimal components
        """
        print("Performing comprehensive PCA analysis...")
        
        # Input validation
        if X_train_scaled is None or len(X_train_scaled) == 0:
            raise ValueError("X_train_scaled cannot be None or empty")
        
        try:
            # Fit PCA with all components
            pca_full = PCA()
            pca_full.fit(X_train_scaled)
            
            # Store results
            self.explained_variance_ratio_ = pca_full.explained_variance_ratio_
            self.components_ = pca_full.components_
            
            # Calculate cumulative explained variance
            cumsum_variance = np.cumsum(self.explained_variance_ratio_)
            
            # Find components for different thresholds
            results = {}
            for threshold in variance_thresholds:
                n_components = np.argmax(cumsum_variance >= threshold) + 1
                results[f'n_components_{int(threshold*100)}'] = n_components
                
                if threshold == 0.95:
                    self.n_components_95 = n_components
                # elif threshold == 0.99:
                #     self.n_components_99 = n_components
            
            results.update({
                'explained_variance_ratio': self.explained_variance_ratio_,
                'cumulative_variance': cumsum_variance,
                'total_components': len(self.explained_variance_ratio_),
                'n_components_95': self.n_components_95,
                # 'n_components_99': self.n_components_99
            })
            
            print(f"✅ PCA analysis completed successfully")
            print(f"   Components for 95% variance: {self.n_components_95}")
            # print(f"   Components for 99% variance: {self.n_components_99}")
            
            return results
            
        except Exception as e:
            print(f"❌ Error in PCA analysis: {e}")
            raise e
    
    def evaluate_pca_performance(self, X_train_scaled, X_test_scaled, y_train, y_test, 
                               classifier_name=None, label_encoder=None, 
                               test_components=None):
        """
        Evaluate classification performance with different PCA components
        """
        print(f"\n🔍 Starting PCA performance evaluation...")
        
        # Validation checks
        if self.n_components_95 is None:
            raise ValueError("Please run fit_pca_analysis first!")
        
        if X_train_scaled is None or X_test_scaled is None:
            raise ValueError("Training and test data cannot be None")
        
        if len(X_train_scaled) == 0 or len(X_test_scaled) == 0:
            raise ValueError("Training and test data cannot be empty")
        
        print(f"Evaluating {classifier_name} performance with PCA...")
        
        # Default component numbers to test
        if test_components is None:
            test_components = [
                self.n_components_95,
                # self.n_components_99,
            ]
            # Remove duplicates and sort
            test_components = sorted(list(set(test_components)))
            test_components = [n for n in test_components if n <= X_train_scaled.shape[1]]
        
        print(f"Testing components: {test_components}")
        
        results = {
            'component_numbers': test_components,
            'accuracies': [],
            'detailed_results': [],
            'best_result': None
        }
        
        best_accuracy = 0
        
        try:
            for n_comp in test_components:
                if n_comp > X_train_scaled.shape[1]:
                    print(f"Skipping {n_comp} components (exceeds feature count)")
                    continue
                    
                print(f"  Testing with {n_comp} components...")
                
                # Apply PCA
                pca = PCA(n_components=n_comp)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                
                # Get and optimize classifier
                best_classifier = self.optimize_classifier_hyperparameters(classifier_name, X_train_pca, y_train, cv_folds=5)
                
                # Train and evaluate
                best_classifier.fit(X_train_pca, y_train)
                y_pred = best_classifier.predict(X_test_pca)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Store results
                result_detail = {
                    'n_components': n_comp,
                    'accuracy': accuracy,
                    'classifier': best_classifier,
                    'y_pred': y_pred,
                    'variance_explained': np.sum(pca.explained_variance_ratio_),
                    'pca_model': pca
                }
                
                results['accuracies'].append(accuracy)
                results['detailed_results'].append(result_detail)
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    results['best_result'] = result_detail
                
                print(f"    ✅ {classifier_name} with {n_comp} components: {accuracy:.4f} "
                      f"({result_detail['variance_explained']:.3f} variance explained)")
            
            # Ensure we have a best result
            if results['best_result'] is None and len(results['detailed_results']) > 0:
                results['best_result'] = results['detailed_results'][0]
            
            # Summary
            if results['best_result'] is not None:
                best_result = results['best_result']
                print(f"\n🏆 Best {classifier_name} result:")
                print(f"   Components: {best_result['n_components']}")
                print(f"   Accuracy: {best_result['accuracy']:.4f}")
                print(f"   Variance explained: {best_result['variance_explained']:.3f}")
            
            return results, X_train_pca, X_test_pca
            
        except Exception as e:
            print(f"❌ Error in performance evaluation: {e}")
            raise e
    
    def _get_classifier(self, classifier_name):
        """Get classifier instance based on name"""
        classifier_name = classifier_name.upper()
        
        if classifier_name == 'KNN':
            return KNeighborsClassifier()
        elif classifier_name == 'SVM':
            return SVC(random_state=42)
        elif classifier_name == 'RF':
            return RandomForestClassifier(random_state=42)
        elif classifier_name == 'DT':
            return DecisionTreeClassifier(random_state=42)
        elif classifier_name == 'NB':
            return GaussianNB()
        elif classifier_name == 'LR':
            return LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unsupported classifier: {classifier_name}")
    
    def optimize_classifier_hyperparameters(self, classifier_name, X_train, y_train, cv_folds=5):
        """
        Optimize hyperparameters for a specific classifier using cross-validation.
        
        Returns:
        --------
        sklearn classifier : Trained classifier with optimal parameters
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
        param_values = list(param_grid.values())[0]  # Get first parameter's values
        
        for param_value in param_values:
            # Create classifier with current parameter
            classifier_params = classifier_config['default_params'].copy()
            classifier_params[param_name] = param_value
            
            classifier = ClassifierClass(**classifier_params)
            
            # Perform cross-validation
            scores = cross_val_score(classifier, X_train, y_train, 
                                cv=cv_folds, scoring='accuracy')
            cv_scores.append(scores.mean())
        
        # Find optimal parameter and create best classifier
        optimal_idx = np.argmax(cv_scores)
        optimal_param_value = param_values[optimal_idx]
        optimal_score = cv_scores[optimal_idx]
        
        # Create classifier with optimal parameters
        optimal_params = classifier_config['default_params'].copy()
        optimal_params[param_name] = optimal_param_value
        best_classifier = ClassifierClass(**optimal_params)
        
        print(f"Optimal {param_name}: {optimal_param_value} (CV Score: {optimal_score:.4f})")
        
        return best_classifier

    def get_optimization_details(self, classifier_name, X_train, y_train, cv_folds=5):
        """
        Get detailed optimization results (separate method if you need the dict)
        
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
        
        return {
            'optimal_param_value': optimal_param_value,
            'optimal_score': optimal_score,
            'param_name': param_name,
            'cv_scores': cv_scores,
            'param_values': param_values
        }
    
    def plot_confusion_matrix(self, y_true, y_pred, classifier_name, n_components, label_encoder):
        """Plot confusion matrix for specific result"""
        try:
            plt.figure(figsize=(8, 6))
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Use label encoder classes if available
            if label_encoder is not None and hasattr(label_encoder, 'classes_'):
                labels = label_encoder.classes_
            else:
                labels = [f'Class {i}' for i in range(len(np.unique(y_true)))]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=labels, yticklabels=labels)
            
            plt.title(f'Confusion Matrix - {classifier_name} with {n_components} PCA Components', 
                     fontsize=14, fontweight='bold')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error plotting confusion matrix: {e}")
    
    def get_component_recommendations(self, results):
        """Get recommendations for optimal number of components"""
        try:
            if results is None:
                print("No results provided for recommendations")
                return None
            
            recommendations = {
                'speed_optimized': self.n_components_95 if self.n_components_95 else 10,
                'accuracy_optimized': self.n_components_99 if self.n_components_99 else 20,
                'balanced': None
            }
            
            if 'component_numbers' in results and 'accuracies' in results:
                component_numbers = results['component_numbers']
                accuracies = results['accuracies']
                
                if len(component_numbers) > 0 and len(accuracies) > 0:
                    # Calculate efficiency scores
                    efficiency_scores = []
                    for i, (n_comp, acc) in enumerate(zip(component_numbers, accuracies)):
                        # Normalize components (lower is better for speed)
                        norm_comp = 1 - (n_comp / max(component_numbers))
                        # Accuracy (higher is better)
                        norm_acc = acc / max(accuracies) if max(accuracies) > 0 else 0
                        # Combined score (70% accuracy, 30% speed)
                        efficiency = 0.7 * norm_acc + 0.3 * norm_comp
                        efficiency_scores.append(efficiency)
                    
                    if len(efficiency_scores) > 0:
                        best_balanced_idx = np.argmax(efficiency_scores)
                        recommendations['balanced'] = component_numbers[best_balanced_idx]
            
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            return {
                'speed_optimized': 10,
                'accuracy_optimized': 50,
                'balanced': 25
            }


def quick_pca_analysis(X_train_scaled, X_test_scaled, y_train, y_test, 
                      classifier_name='KNN', label_encoder=None, plot=False):
    """
    Quick PCA analysis with classification evaluation
    
    Returns:
    --------
    tuple : (pca_analyzer, analysis_results, performance_results)
    """
    try:
        print("🚀 Starting quick PCA analysis...")
        
        # Input validation
        if X_train_scaled is None or X_test_scaled is None:
            raise ValueError("Training and test data cannot be None")
        
        analyzer = PCAAnalyzer()
        
        # Perform PCA analysis
        print("Step 1: Fitting PCA analysis...")
        analysis_results = analyzer.fit_pca_analysis(X_train_scaled)
        
        # Evaluate performance
        print("Step 2: Evaluating performance...")
        performance_results = analyzer.evaluate_pca_performance(
            X_train_scaled, X_test_scaled, y_train, y_test, 
            classifier_name, label_encoder
        )
        
        print("✅ Quick PCA analysis completed successfully!")
        
        return analyzer, analysis_results, performance_results
        
    except Exception as e:
        print(f"❌ Error in quick_pca_analysis: {e}")
        print(f"Error type: {type(e)}")
        raise e