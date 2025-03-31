# Dataset Information

## Overview

This project uses the Reddit Hyperlink Network dataset from the Stanford Network Analysis Project (SNAP). The dataset contains timestamped hyperlinks between subreddits from Reddit with sentiment annotations.

## Dataset Details

### Source

- **Name**: Reddit Hyperlink Network
- **Provider**: Stanford Network Analysis Project (SNAP)
- **URL**: [https://snap.stanford.edu/data/soc-RedditHyperlinks.html](https://snap.stanford.edu/data/soc-RedditHyperlinks.html)

### Files

1. **soc-redditHyperlinks-body.tsv** - Hyperlinks created within the body of posts
2. **soc-redditHyperlinks-title.tsv** - Hyperlinks created in post titles

This analysis primarily uses the body hyperlinks dataset.

### Format

The data is in tab-separated value (TSV) format with the following columns:

- **SOURCE_SUBREDDIT**: The subreddit where the hyperlink was posted
- **TARGET_SUBREDDIT**: The subreddit that the hyperlink points to
- **POST_ID**: The ID of the post containing the hyperlink
- **TIMESTAMP**: When the hyperlink was posted
- **LINK_SENTIMENT**: The sentiment of the hyperlink (-1 for negative, 0 for neutral, 1 for positive)
- Additional metadata columns

### Statistics

- **Time span**: 2014-2017
- **Number of subreddits**: ~36,000
- **Number of hyperlinks**: ~860,000

## Data Preparation

For this analysis, the data is:

1. Loaded from the TSV file
2. Sampled to make computation manageable
3. Converted to a directed graph with subreddits as nodes and hyperlinks as edges
4. Sentiment is binarized to positive (1) or negative (-1)

## Data Access

Due to the size of the dataset files, they are not included in this repository. To run the analysis, download the data from the [SNAP website](https://snap.stanford.edu/data/soc-RedditHyperlinks.html) 
