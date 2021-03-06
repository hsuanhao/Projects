import nltk
import string
import re

def spelling_checker(words, correct_spellings):
    """
    spelling_checker: the function to check the spelling of words in text
    
    input: 
         words: list of words to examine
         corect_spellings : list of correct words
    output: 
         correction : corrected text
    """
    correction = list()
    porter = nltk.PorterStemmer()
    
    for word in words:
        w = word.lower()
        
        # If word is punctuation, there's no need to check spelling
        if w in string.punctuation: 
            correction.append(w)
            continue
            
        # Check spelling by using Edit distance on the two words with transpositions
        correct_word1, dist1 = edit_distance_trans(w, correct_spellings)
        if dist1 < 0.1:
            correction.append(word)
            continue
        
        # Check spelling by using Edit distance on the two words with transpositions
        # after taking off ending letters, ing, ed, s 
        regex = r'(\w+)(ing|ed|s)$'
        if re.search(regex,word):
            w, ending = re.findall(regex,word)[0]
            correct_word2, dist2 = edit_distance_trans(w.lower(), correct_spellings)
            correction.append(correct_word2 + ending)
            continue
        
        correction.append(correct_word1)
        
    return correction


def edit_distance_trans(word, correct_spellings):
    """
    edit_distance_trans: the function to calculate Edit distance on the two words with transpositions
    
    input: 
         word: word to examine
         corect_spellings : list of correct words
    output: 
         edit : recommend correctly spelled words
         dist : distance between word and edit
    """
    edit = list()
    temp = list()
    for recommend in correct_spellings:
        recommend = recommend.lower()
        # Consider the word starting with the same letter as the examined word
        if recommend[0] != word[0]: continue
        distance = nltk.edit_distance(word,recommend)
        temp.append( (recommend, distance) )  
    edit, dist = sorted(temp, key= lambda x: x[1])[0]  
    return edit, dist
