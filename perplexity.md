The way of collecting perplexity:

Suppose we have a string of tokens and we want to calculate their perplexity:

"The answer to 1 + 1 is 2".

This appears in the context of the following two lines:

<system>You are a helpful agent.</system><user>What is 1 + 1?</user><assistant>

<system>You are a helpful agent.</system><user>What is 1 + 1?</user><assistant>The answer to 1 + 1 is 2

We want the log probability of the LLM to output each token in the answer given the previous string.
For example, we get the log probability of the first letter "T" by passing in:

<system>You are a helpful agent.</system><user>What is 1 + 1?</user><assistant>

Then, we get the log probability of the second letter "h" by passing in:

<system>You are a helpful agent.</system><user>What is 1 + 1?</user><assistant>T

And the log probability of the last character "2" by passing in:

<system>You are a helpful agent.</system><user>What is 1 + 1?</user><assistant>The answer to 1 + 1 is 

This assumes that one token corresponds to one character. This should be generalized by doing the same operation in the token level.

In this way, we collected all the log probabilities of the tokens. Finally, we calculate the log perplexity to be the average log probabilities of the tokens.

