#!/usr/bin/python
"""

dnautil module contains useful functions to process DNA sequence.

"""
__author__ = "Hsuan-Hao Fan"


def readGenome(filename):
    """
    ===========================================================
    
    Parses a DNA reference genome from a file in the FASTA format
    
    ===========================================================
    
    Parameters
    ----------
    filename: str file name in FASTA format
    
    Returns
    -------
    genome: DNA sequence
    
    """
    
    genome = ''
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                # Ignore header line with genome information
                if not line[0] == '>':
                    genome += line.rstrip()
    except IOError:
        print("%s does not exist!!" % filename)
    return genome


def readFastq(filename):
    """
    ===================================================================
    
    Parses the read and quality strings from a FASTQ file containing 
    sequencing reads FASTQ format
    
    ===================================================================
    
    Parameters
    ----------
    filename: str file name in FASTQ format
    
    Returns
    -------
    (sequences, qualities): list of strings   
    
    sequences: DNA sequences
    qualities: base qualities
    
    References
    ----------
    
    Example: 
    
    @SEQ_ID
    GATTTGGGGTTCAAAGCAGTATCGATCAAATAGTAAATCCATTTGTTCAACTCACAGTTT
    +
    !''*((((***+))%%%++)(%%%%).1***-+*''))**55CCF>>>>>>CCCCCCC65
    
    see https://en.wikipedia.org/wiki/FASTQ_format
    
    """
    sequences = []
    qualities = []
    try:
        with open(filename) as fh:
            while True:
                fh.readline()                   # Skip name line
                seq = fh.readline().rstrip()    # read base sequence
                fh.readline()                   # skip placeholder line
                qual = fh.readline().rstrip()   # base quality line
            
                if len(seq) == 0:
                    break
            
                sequences.append(seq)
                qualities.append(qual)
            
    except IOError:
        print("The file, %s, does not exist!!" % filename)
              
    return sequences, qualities


def naive(p, t):
    """
    ========================================================
    
    A function to carry out exact matching by implementing
    naive algorithm
    
    ========================================================
    
    Parameters
    ----------
    p: str pattern
    
    t: str text
    
    Returns
    -------
    occurrences: list of integers 
                 All the indices where p matches agains t.
    
    """
    
    occurrences = [] 
    
    for i in range(len(t) - len(p) + 1):   # loop over alignments
        match = True
        for j in range(len(p)):            # loop over characters
            if t[i+j] != p[j]:
                match = False
                break
                
        if match: occurrences.append(i)
    
    return occurrences


def naive_with_rc(p, t):
    """
    ========================================================
    
    A function to carry out strand-aware exact matching 
    by implementing naive algorithm
    
    ========================================================
    
    Parameters
    ----------
    p: str pattern
    
    t: str text
    
    Returns
    -------
    occurrences: list of integers 
    
                 All the indices where p or its reverse complement 
                 matches agains t.
    
    """
    
    record = [] 
    
    record = naive(p,t)
    
    rc = reverse_complement(p)
    
    record += naive(rc,t)
    
    # Remove repeated counts
    occurrences = sorted(list(set(record)))
    
    return occurrences
    


def gc(dna):
    """
    ==========================================================
    
    This function compute the GC percentage of a DNA sequence
    
    ==========================================================
    
    Parameters
    ---------
    dna: str A DNA sequence
    
    Returns
    -------
    gcpercent: float GC percentage of a DNA sequence
    
    
    """
    
    # nbases is number of non-defined bases n or N
    nbases = dna.count('n') + dna.count('N')
    
    # gcpercent is GC percentage of a DNA sequence
    gcpercent = (dna.count('c') + dna.count('C') + dna.count('g') + dna.count('G'))*1.0/(len(dna)-nbases)
    
    return gcpercent

def dna_has_stop(dna, frame=0):
    """
    ===================================================
    
    This function check if a given DNA sequence
    contains an in-frame stop codon.
    
    ===================================================
    
    Parameters
    ---------
    dna: str A DNA sequence
    
    frame: int frame argument equal to 0, 1, or 2.
               Default is 0.
    
    Returns
    -------
    stop_codon_found: boolean Whether the DNA sequence contains stop codon
    
    """
    stop_codon_found = False
    stop_codons = ['tga', 'tag', 'taa']
    
  
    for i in range(frame, len(dna), 3):
        codon = dna[i:i+3].lower()
        
        # Check whether codon is a stop codon
        if codon in stop_codons:
            stop_codon_found = True
            break
    
    if stop_codon_found:
        print("Input sequence has an in-frame stop codon.")
    else:
        print("Input sequence has no in-frame stop codon.")
        
    return stop_codon_found

def reverse_complement(dna):
    """
    ======================================================
    
    A function to reverse complement of the dna sequence.
    
    ======================================================
    
    Parameters
    ----------
    dna: str A DNA sequence
    
    
    Returns
    -------
    seq: str A reverse complement of the input DNA sequence
    
    """
    base_complement = {'A':'T', 'C':'G', 'G':'C', 'T':'A', \
                      'N':'N', 'a':'t', 'c':'g', 'g':'c', \
                      't':'a', 'n':'n'}
    seq = ""
    for base in dna:
        seq = base_complement[base] + seq

    return seq
    
    
    
