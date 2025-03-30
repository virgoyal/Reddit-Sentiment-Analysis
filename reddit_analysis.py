import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

def predict_community_sentiment_balanced(file_path, sample_size=5000, random_state=42, balance_strategy='combined'):
    """
    Improved Reddit community sentiment prediction with balanced classes and proper validation.
    
    Parameters:
    - file_path: Path to the Reddit hyperlinks data
    - sample_size: Number of samples to use (for computational efficiency)
    - random_state: Random seed for reproducibility
    - balance_strategy: How to handle class imbalance ('oversample', 'undersample', 'combined')
    """
    print(f"Loading Reddit hyperlink data from {file_path}...")
    
    # Read the data
    df = pd.read_csv(file_path, sep='\t', low_memory=False)
    print(f"Loaded {len(df):,} hyperlinks")
    
    # Sample to make computation manageable
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=random_state)
        print(f"Sampled down to {len(df):,} hyperlinks for faster computation")
    
    # Get sentiment column name (handle different datasets)
    sentiment_col = 'LINK_SENTIMENT' if 'LINK_SENTIMENT' in df.columns else 'POST_LABEL'
    
    # Convert sentiment to numeric and then to binary (-1/1)
    if sentiment_col in df.columns:
        df[sentiment_col] = pd.to_numeric(df[sentiment_col], errors='coerce')
        # Convert to binary sentiment: -1 or 1
        df['binary_sentiment'] = df[sentiment_col].apply(lambda x: 1 if x > 0 else -1)
    else:
        print(f"Sentiment column '{sentiment_col}' not found. Cannot proceed.")
        return
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add edges with sentiment as attribute
    for _, row in df.iterrows():
        source = row['SOURCE_SUBREDDIT']
        target = row['TARGET_SUBREDDIT']
        sentiment = row['binary_sentiment']
        
        # Add nodes if they don't exist
        if source not in G:
            G.add_node(source)
        if target not in G:
            G.add_node(target)
        
        # Add edge with sentiment attribute
        G.add_edge(source, target, sentiment=sentiment)
    
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Calculate graph features
    print("Calculating graph features...")
    
    # In-degree and out-degree centrality
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # PageRank with fewer iterations
    pagerank = nx.pagerank(G, max_iter=50, tol=1e-3)
    
    # Create edge dataset for sentiment prediction
    edge_data = []
    
    # Go through each edge in the graph
    for source, target, data in G.edges(data=True):
        # Other graph-based features
        source_pagerank = pagerank.get(source, 0)
        target_pagerank = pagerank.get(target, 0)
        source_in_degree = in_degree.get(source, 0)
        source_out_degree = out_degree.get(source, 0)
        target_in_degree = in_degree.get(target, 0)
        target_out_degree = out_degree.get(target, 0)
        
        # Get sentiment label
        sentiment = data['sentiment']
        
        # Common neighbors (measure of structural similarity)
        common_neighbors = len(list(nx.common_neighbors(G.to_undirected(), source, target)))
        
        # Jaccard coefficient
        try:
            jaccard = len(list(nx.common_neighbors(G.to_undirected(), source, target))) / len(
                set(G.to_undirected()[source]) | set(G.to_undirected()[target]))
        except ZeroDivisionError:
            jaccard = 0
        
        # Preferential attachment
        preferential_attachment = source_out_degree * target_in_degree
        
        # Store features and label
        features = {
            'source': source,
            'target': target,
            'source_pagerank': source_pagerank,
            'target_pagerank': target_pagerank,
            'source_in_degree': source_in_degree,
            'source_out_degree': source_out_degree,
            'target_in_degree': target_in_degree,
            'target_out_degree': target_out_degree,
            'common_neighbors': common_neighbors,
            'jaccard': jaccard,
            'preferential_attachment': preferential_attachment,
            'sentiment': sentiment
        }
        
        edge_data.append(features)
    
    # Convert to DataFrame
    edge_df = pd.DataFrame(edge_data)
    print(f"Created edge dataset with {len(edge_df)} rows")
    
    # Check class distribution
    class_distribution = edge_df['sentiment'].value_counts()
    print(f"Original class distribution:\n{class_distribution}")
    
    # Define feature list explicitly
    feature_list = [
        'source_pagerank', 'target_pagerank',
        'source_in_degree', 'source_out_degree', 
        'target_in_degree', 'target_out_degree',
        'common_neighbors', 'jaccard', 'preferential_attachment'
    ]
    
    # Prepare data for modeling - splitting into training and holdout test sets
    X = edge_df[feature_list]
    y = edge_df['sentiment']
    
    # Create a stratified holdout test set (20% of data)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y)
    
    print(f"Training+Validation set size: {len(X_train_val)}, Holdout Test set size: {len(X_test)}")
    print(f"Training+Validation set class distribution: {Counter(y_train_val)}")
    print(f"Holdout Test set class distribution: {Counter(y_test)}")
    
    # Find optimal hyperparameters using cross-validation on the training+validation set
    best_model = find_optimal_model(X_train_val, y_train_val, balance_strategy, random_state)
    
    # Final evaluation on the holdout test set
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics on the holdout test set
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Model Performance on Holdout Test Set:")
    print(f"Accuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Visualize model performance on the test set
    visualize_model_performance(X_test, y_test, y_pred, best_model, feature_list)
    
    # Generate new community predictions
    predict_new_relationships(G, edge_df, best_model, in_degree, out_degree, pagerank, feature_list)
    
    return best_model, edge_df

def find_optimal_model(X, y, balance_strategy, random_state=42):
    """
    Find the optimal model using cross-validation with balanced classes.
    
    Parameters:
    - X: Features DataFrame
    - y: Target variable
    - balance_strategy: Strategy for handling class imbalance
    - random_state: Random seed
    
    Returns:
    - Best trained model
    """
    print("\nPerforming cross-validation with balanced classes...")
    
    # Define cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    
    # Define model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state, class_weight='balanced')
    
    # Define resampling strategies for class imbalance
    if balance_strategy == 'oversample':
        resampler = SMOTE(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        print(f"After SMOTE oversampling: {Counter(y_resampled)}")
    elif balance_strategy == 'undersample':
        resampler = RandomUnderSampler(random_state=random_state)
        X_resampled, y_resampled = resampler.fit_resample(X, y)
        print(f"After undersampling: {Counter(y_resampled)}")
    elif balance_strategy == 'combined':
        # Combined approach: undersample majority class and oversample minority class
        oversampler = SMOTE(random_state=random_state)
        undersampler = RandomUnderSampler(random_state=random_state)
        
        # First oversample the minority class
        X_over, y_over = oversampler.fit_resample(X, y)
        print(f"After oversampling: {Counter(y_over)}")
        
        # Then undersample the majority class
        X_resampled, y_resampled = undersampler.fit_resample(X_over, y_over)
        print(f"After combined resampling: {Counter(y_resampled)}")
    else:
        # No resampling, just use class weights in the model
        X_resampled, y_resampled = X, y
        print("Using model class weights only (no resampling)")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=cv, scoring='f1_macro')
    print(f"Cross-validation F1-macro scores: {cv_scores}")
    print(f"Mean CV F1-macro score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Train final model on all resampled data
    model.fit(X_resampled, y_resampled)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature importance:")
    for _, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return model

def visualize_model_performance(X_test, y_test, y_pred, model, feature_list):
    """
    Create comprehensive visualizations of model performance on the test set.
    """
    plt.figure(figsize=(18, 12))
    
    # Confusion matrix - subplot 1
    plt.subplot(2, 3, 1)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Feature importance - subplot 2
    plt.subplot(2, 3, 2)
    feature_importance = pd.DataFrame({
        'feature': feature_list,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    
    # ROC curve - subplot 3
    plt.subplot(2, 3, 3)
    try:
        y_score = model.predict_proba(X_test)[:, 1]  # Positive class probability
        fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    except:
        plt.text(0.5, 0.5, "ROC curve calculation failed", ha='center', va='center')
    
    # Precision-Recall curve - subplot 4
    plt.subplot(2, 3, 4)
    try:
        precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=1)
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
    except:
        plt.text(0.5, 0.5, "PR curve calculation failed", ha='center', va='center')
    
    # Class distribution - subplot 5
    plt.subplot(2, 3, 5)
    class_counts = pd.Series(y_test).value_counts()
    plt.pie(class_counts, 
            labels=['Positive' if i == 1 else 'Negative' for i in class_counts.index],
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90)
    plt.axis('equal')
    plt.title('Test Set Class Distribution')
    
    # Prediction distribution - subplot 6
    plt.subplot(2, 3, 6)
    pred_dist = pd.Series(y_pred).value_counts()
    plt.pie(pred_dist,
            labels=['Positive' if i == 1 else 'Negative' for i in pred_dist.index],
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90)
    plt.axis('equal')
    plt.title('Prediction Distribution')
    
    plt.tight_layout()
    plt.savefig('model_performance_test_set.png', dpi=300, bbox_inches='tight')
    plt.close()

def predict_new_relationships(G, edge_df, model, in_degree, out_degree, pagerank, feature_list, max_communities=50):
    """
    Predict sentiment for communities that haven't directly interacted.
    """
    print("\nPredicting sentiment for communities that haven't interacted yet...")
    
    # Get most active communities for prediction
    active_sources = sorted([(node, out_degree.get(node, 0)) for node in G.nodes()], 
                          key=lambda x: x[1], reverse=True)[:max_communities]
    active_targets = sorted([(node, in_degree.get(node, 0)) for node in G.nodes()], 
                          key=lambda x: x[1], reverse=True)[:max_communities]
    
    active_sources = [node for node, _ in active_sources]
    active_targets = [node for node, _ in active_targets]
    
    # Find potential pairs that haven't interacted
    non_interacting_pairs = []
    for source in active_sources:
        for target in active_targets:
            if source != target and not G.has_edge(source, target):
                non_interacting_pairs.append((source, target))
    
    # Sample a subset for prediction if there are too many
    import random
    max_pairs = min(500, len(non_interacting_pairs))
    if max_pairs > 0:
        sampled_pairs = random.sample(non_interacting_pairs, max_pairs)
    else:
        sampled_pairs = []
    
    # Prepare features for prediction
    prediction_data = []
    
    for source, target in sampled_pairs:
        # Extract graph-based features
        source_pagerank_val = pagerank.get(source, 0)
        target_pagerank_val = pagerank.get(target, 0)
        source_in_degree_val = in_degree.get(source, 0)
        source_out_degree_val = out_degree.get(source, 0)
        target_in_degree_val = in_degree.get(target, 0)
        target_out_degree_val = out_degree.get(target, 0)
        
        # Common neighbors
        common_neighbors_val = len(list(nx.common_neighbors(G.to_undirected(), source, target)))
        
        # Jaccard coefficient
        try:
            jaccard_val = len(list(nx.common_neighbors(G.to_undirected(), source, target))) / len(
                set(G.to_undirected()[source]) | set(G.to_undirected()[target]))
        except (ZeroDivisionError, KeyError):
            jaccard_val = 0
        
        # Preferential attachment
        preferential_attachment_val = source_out_degree_val * target_in_degree_val
        
        # Store features
        features = {
            'source': source,
            'target': target,
            'source_pagerank': source_pagerank_val,
            'target_pagerank': target_pagerank_val,
            'source_in_degree': source_in_degree_val,
            'source_out_degree': source_out_degree_val,
            'target_in_degree': target_in_degree_val,
            'target_out_degree': target_out_degree_val,
            'common_neighbors': common_neighbors_val,
            'jaccard': jaccard_val,
            'preferential_attachment': preferential_attachment_val
        }
        
        prediction_data.append(features)
    
    # Convert to DataFrame
    prediction_df = pd.DataFrame(prediction_data)
    
    if len(prediction_df) > 0:
        # Predict sentiment
        X_pred = prediction_df[feature_list]
        pred_sentiment = model.predict(X_pred)
        pred_proba = model.predict_proba(X_pred)
        
        # Add predictions to DataFrame
        prediction_df['predicted_sentiment'] = pred_sentiment
        prediction_df['positive_probability'] = [p[1] if len(p) > 1 else 0.5 for p in pred_proba]
        
        # Sort by sentiment prediction confidence
        prediction_df['confidence'] = np.abs(prediction_df['positive_probability'] - 0.5) * 2
        prediction_df = prediction_df.sort_values('confidence', ascending=False)
        
        # Display top predictions with highest confidence
        top_positive = prediction_df[prediction_df['predicted_sentiment'] == 1].head(10)
        top_negative = prediction_df[prediction_df['predicted_sentiment'] == -1].head(10)
        
        print("\nTop 10 community pairs predicted to have positive sentiment:")
        for _, row in top_positive.iterrows():
            print(f"r/{row['source']} → r/{row['target']}: {row['positive_probability']:.2f} confidence")
        
        print("\nTop 10 community pairs predicted to have negative sentiment:")
        for _, row in top_negative.iterrows():
            print(f"r/{row['source']} → r/{row['target']}: {1-row['positive_probability']:.2f} confidence")
        
        # Visualize top predictions
        visualize_top_predictions(top_positive, top_negative)
        visualize_prediction_distributions(prediction_df)
        visualize_network_with_predictions(G, top_positive, top_negative)
        
        return prediction_df
    else:
        print("No valid prediction pairs found.")
        return None

def visualize_top_predictions(top_positive, top_negative):
    """Create a visual comparison of top predicted community pairs"""
    plt.figure(figsize=(15, 6))
    
    # Function to create labels
    def create_label(row):
        return f"r/{row['source']} → r/{row['target']}"
    
    # Positive predictions
    if len(top_positive) > 0:
        ax1 = plt.subplot(1, 2, 1)
        pos_labels = [create_label(row) for _, row in top_positive.iterrows()]
        pos_probs = top_positive['positive_probability'].values
        
        # Create horizontal bar chart for positive predictions
        ax1.barh(pos_labels, pos_probs, color='green', alpha=0.7)
        for i, v in enumerate(pos_probs):
            ax1.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        ax1.set_xlim(0, 1.1)
        ax1.set_title('Top Positive Sentiment Predictions', fontsize=14)
        ax1.set_xlabel('Probability of Positive Sentiment')
    
    # Negative predictions
    if len(top_negative) > 0:
        ax2 = plt.subplot(1, 2, 2)
        neg_labels = [create_label(row) for _, row in top_negative.iterrows()]
        neg_probs = 1 - top_negative['positive_probability'].values
        
        # Create horizontal bar chart for negative predictions
        ax2.barh(neg_labels, neg_probs, color='red', alpha=0.7)
        for i, v in enumerate(neg_probs):
            ax2.text(v + 0.01, i, f"{v:.2f}", va='center')
        
        ax2.set_xlim(0, 1.1)
        ax2.set_title('Top Negative Sentiment Predictions', fontsize=14)
        ax2.set_xlabel('Probability of Negative Sentiment')
    
    plt.tight_layout()
    plt.savefig('top_community_pairs.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_prediction_distributions(prediction_df):
    """Visualize the distribution of predictions"""
    plt.figure(figsize=(15, 5))
    
    # Distribution of sentiment predictions
    plt.subplot(131)
    sentiment_counts = prediction_df['predicted_sentiment'].value_counts()
    labels = ['Positive' if i == 1 else 'Negative' for i in sentiment_counts.index]
    plt.pie(sentiment_counts, 
            labels=labels, 
            autopct='%1.1f%%',
            colors=['green', 'red'],
            startangle=90)
    plt.axis('equal')
    plt.title('Predicted Sentiment Distribution')
    
    # Distribution of confidence scores
    plt.subplot(132)
    sns.histplot(prediction_df['confidence'], bins=20, kde=True)
    plt.title('Distribution of Prediction Confidence')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    
    # Distribution of positive probability
    plt.subplot(133)
    sns.histplot(prediction_df['positive_probability'], bins=20, kde=True)
    plt.axvline(0.5, color='red', linestyle='--')
    plt.title('Distribution of Positive Probability')
    plt.xlabel('Probability of Positive Sentiment')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig('prediction_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def visualize_network_with_predictions(G, top_positive, top_negative, max_communities=20):
    """Create a network visualization with predicted links"""
    # Get communities involved in top predictions
    pred_communities = set()
    
    # Add top prediction communities
    for df in [top_positive, top_negative]:
        if len(df) > 0:
            for _, row in df.iterrows():
                pred_communities.add(row['source'])
                pred_communities.add(row['target'])
    
    # Add some high-degree nodes to make the network more connected
    high_degree_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:10]
    pred_communities.update(high_degree_nodes)
    
    # Limit to manageable number
    pred_communities = list(pred_communities)[:max_communities]
    
    # Create subgraph for visualization
    viz_graph = nx.DiGraph()
    
    # Add nodes
    for node in pred_communities:
        viz_graph.add_node(node)
    
    # Add existing edges between these communities
    for source in pred_communities:
        for target in pred_communities:
            if G.has_edge(source, target):
                sentiment = G[source][target].get('sentiment', 0)
                if sentiment > 0:
                    viz_graph.add_edge(source, target, color='green', style='solid', width=1.5)
                else:
                    viz_graph.add_edge(source, target, color='red', style='solid', width=1.5)
    
    # Add predicted edges
    for _, row in top_positive.iterrows():
        source, target = row['source'], row['target']
        if source in pred_communities and target in pred_communities:
            if not viz_graph.has_edge(source, target):
                confidence = row['confidence']
                viz_graph.add_edge(source, target, color='lightgreen', style='dashed', 
                                  width=0.5 + confidence)
    
    for _, row in top_negative.iterrows():
        source, target = row['source'], row['target']
        if source in pred_communities and target in pred_communities:
            if not viz_graph.has_edge(source, target):
                confidence = row['confidence']
                viz_graph.add_edge(source, target, color='pink', style='dashed', 
                                  width=0.5 + confidence)
    
    # Draw the network
    plt.figure(figsize=(14, 14))
    
    # Try different layout algorithms
    try:
        pos = nx.spring_layout(viz_graph, k=0.3, iterations=50, seed=42)
    except:
        try:
            pos = nx.kamada_kawai_layout(viz_graph)
        except:
            pos = nx.circular_layout(viz_graph)
    
    # Draw nodes
    nx.draw_networkx_nodes(viz_graph, pos, node_size=800, node_color='lightblue', alpha=0.8)
    
    # Draw edges by type
    edge_solid = [(u, v) for u, v in viz_graph.edges() if viz_graph[u][v]['style'] == 'solid']
    edge_dashed = [(u, v) for u, v in viz_graph.edges() if viz_graph[u][v]['style'] == 'dashed']
    
    # Colors and widths
    colors_solid = [viz_graph[u][v]['color'] for u, v in edge_solid]
    colors_dashed = [viz_graph[u][v]['color'] for u, v in edge_dashed]
    widths_solid = [viz_graph[u][v]['width'] for u, v in edge_solid]
    widths_dashed = [viz_graph[u][v]['width'] for u, v in edge_dashed]
    
    # Draw existing edges
    if edge_solid:
        nx.draw_networkx_edges(viz_graph, pos, edgelist=edge_solid, 
                              edge_color=colors_solid, width=widths_solid, 
                              arrowsize=15)
    
    # Draw predicted edges
    if edge_dashed:
        nx.draw_networkx_edges(viz_graph, pos, edgelist=edge_dashed, 
                              edge_color=colors_dashed, width=widths_dashed,
                              style='dashed', arrowsize=10, alpha=0.7)
    
    # Draw labels
    nx.draw_networkx_labels(viz_graph, pos, font_size=10, font_weight='bold')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', edgecolor='black', label='Existing Positive'),
        Patch(facecolor='red', edgecolor='black', label='Existing Negative'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Predicted Positive'),
        Patch(facecolor='pink', edgecolor='black', label='Predicted Negative')
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    plt.title('Reddit Communities Network with Predicted Sentiment Links', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('network_with_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        balance_strategy = sys.argv[2] if len(sys.argv) > 2 else 'combined'
    else:
        # Use a default path
        file_path = "soc-redditHyperlinks-body.tsv"
        balance_strategy = 'combined'
    
    predict_community_sentiment_balanced(file_path, balance_strategy=balance_strategy)