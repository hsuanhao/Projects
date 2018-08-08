# Spelling Recommender

In this project, I created three different spelling recommenders by implementing **nltk library in Python for natural language processing** and three distance metrics: **Jaccard distance on the trigrams of the two words**; **Jaccard distance on the 4-grams of the two words**; **Edit distance on the two words with transpositions**.


## How to run
- Save the text you want to check in test.txt
- Run spelling_recommender.ipynb

## Files
- test.txt : The text to check the spelling
  (The sample text is from the first paragraph on this [website](https://en.wikipedia.org/wiki/News), and I put some typos on purpose.)
- spelling_recommender.ipynb : The main program
- metric_distance.py : Include functions to calculate distance between two words

## References
- Jaccard distance on [Wiki](https://en.wikipedia.org/wiki/Jaccard_index).
- Edit distance on the two words with transpositions on [Wiki](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance).


## Requirements

Python packages:

- nltk = 3.3
- jupyter notebook
