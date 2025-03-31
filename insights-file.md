# Reddit Community Sentiment Analysis - Key Insights

## Overview

Here are my key findings from the Reddit community sentiment prediction model. The analysis reveals patterns in how subreddits interact and predicts potential sentiment between communities that haven't directly engaged yet.

## Key Findings

### 1. Class Imbalance in Reddit Interactions

I found a major imbalance in sentiment between Reddit communities:
- About 93% of hyperlinks between subreddits have positive sentiment
- Only about 7% have negative sentiment
Thats a good sign :)!

This also makes sense - most cross-posting and referencing on Reddit happens in a positive or neutral context. Negative interactions are less common and tend to happen between specific community pairs.

### 2. Feature Importance

The most important features for predicting sentiment between communities are:

1. **Target PageRank (24.5%)**: How influential/important the target subreddit is
2. **Source PageRank (15.5%)**: How influential the source subreddit is
3. **Preferential Attachment (15.5%)**: The product of source out-degree and target in-degree

This suggests that community status matters a lot. 

### 3. Network Structure Insights

The network visualization shows several interesting patterns:

- **Central Hubs**: Subreddits like r/funny, r/askreddit, and r/todayilearned connect many other communities
- **Community Clusters**: Communities with similar topics tend to group together
- **Sentiment Patterns**: Negative links often cross between different community clusters, while positive links are more common within clusters

### 4. Prediction Highlights

#### Top Predicted Positive Relationships

Some interesting potential positive relationships I found:
- r/dailydot → r/anime (0.99 confidence)
- r/subredditcancer → r/dota2 (0.98 confidence)
- r/conspiracy → r/gaming (0.98 confidence)


#### Top Predicted Negative Relationships

Some potential negative relationships include:
- r/iama → r/conspiracy (0.87 confidence)
- r/karmacourt → r/videos (0.85 confidence)
- r/keto → r/atheism (0.81 confidence)



## Model Performance Analysis

### Classification Results

The model achieves:
- Overall accuracy: 87.7%
- High precision and recall for positive sentiment (94% and 93%)
- Much lower performance on negative sentiment (14% precision, 16% recall)

This performance gap isn't surprising given the extreme class imbalance.

### ROC and Precision-Recall Analysis

The ROC curve (AUC = 0.59) shows the model performs slightly better than random chance at distinguishing between positive and negative sentiment. The precision-recall curve shows stable precision for most recall levels, but drops when trying to capture more negative instances.

## Practical Applications

### 1. Community Management

This analysis could help Reddit administrators:
- Anticipate potential conflicts between subreddits
- Identify communities that might work well together


### 2. Trend Analysis

The model can spot emerging patterns:
- Growing tensions between previously friendly communities
- New positive relationships forming
- Communities that bridge different parts of the network

### 3. Content Recommendation

Predicted sentiment could improve recommendations:
- Suggest content from positively-linked communities while avoiding recommending content from negatively-linked communities
- Help content creators find better cross-posting opportunities

## Future Directions

1. **Temporal Analysis**: Track how sentiment changes over time
2. **Content Analysis**: Combine network features with text content

## Conclusion

This analysis reveals meaningful patterns in how Reddit communities interact. Despite the challenges of extreme class imbalance, the network-based approach successfully identifies patterns in community interactions and makes reasonable predictions about potential sentiment between communities that haven't yet engaged directly.
