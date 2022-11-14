import string


DNA_ALPHABET = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

PROTEIN_ALPHABET = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10,
    'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

ENGLISH_ALPHABET = {char: ord(char) - ord('A') for char in string.ascii_uppercase}

# More info about IUPAC Alphabet here: https://www.cottongen.org/help/nomenclature/IUPAC_nt
IUPAC_ALPHABET = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'G': 4, 'H': 5, 'K': 6, 'M': 7, 'N': 8, 'R': 9, 'S': 10, 
    'T': 11, 'V': 12, 'W': 13, 'Y': 14
}

alphabets = {'DNA': DNA_ALPHABET, 'IUPAC': IUPAC_ALPHABET, 'PROTEIN': PROTEIN_ALPHABET, 'ENGLISH': ENGLISH_ALPHABET}