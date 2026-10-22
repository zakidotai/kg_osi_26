Decompose from dataset extracted numeric values - each a feature.

Indexes lined up to specific questions, that we get an LLM to answer them.

Compute error. Turn questions on and off / extractors. If that improves error, we have our partial derivative/gradient for that question, and remove that.

Ask agent to generate NEW questions that hopefully reduce error.

====================

# Basically:



One reliable extractor, PL. Can we use this to improve the rest of our extraction?

* Two parts of computing derivative:
===

# 1\. Query to DB. What are solutes, etc.? Turn Qs on and off based on error. Turn off questions if that improves

# 2\. Extraction from the subset.

# 

Given partial derivatives for each questions, can the agent compute new candidate set, the compute similarity to Qs that reduce error the most?

# Feature extraction, get smaller dataset from query to the DB. From features, condition and fit GP. 5x cross-validation. For each of those calculate diff in peak PL, and average that --> 1 scalar at the end.

# 

Extractor would get T or t from synthesis text or embedding answer to question over synthesis text.

Increasing extraction reliability + speed. --> More scalable.

# Create extractors that can be turned off turn on to get a gradient. Single function call? Maybe that extractor has instructions? Temperature or time passed in?

# 

Synthesis TEXT --> ASK Q questions over this text --> We get Q answers --> Each answer gets embedded --> Matrix of Q x 248.

Normalize units. Mat-Sci-BERT extract everything but restrict to these named entities.

# How to optimize questions for given dataset?

# 

Add additional search engines that pull out dataset. Take dataset + natural language query --> subset of dataset.



===

