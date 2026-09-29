"""Coarse Spanish grapheme -> viseme mapping.
Each vowel and bilabial-closure consonant produces one (openness, roundedness) event.
openness: 0=closed, 0.5=mid, 1=open.  roundedness: 0=spread, 1=rounded.
Silent h, and consonants with no strong independent visible shape, are skipped.
"""
VOWELS = {
    'a': (1.0, 0.0),
    'e': (0.6, 0.0),
    'i': (0.2, 0.0),
    'o': (0.6, 1.0),
    'u': (0.2, 1.0),
}
BILABIAL = set('pbvm')  # momentary closure

def word_to_viseme_sequence(word):
    word = word.lower()
    seq = []
    for ch in word:
        if ch in VOWELS:
            seq.append(VOWELS[ch])
        elif ch in BILABIAL:
            seq.append((0.05, 0.3))  # closed lips, roundedness ~ neutral/slightly rounded
        # other consonants: skip (no strong independent visible shape assumed)
    return seq

if __name__ == "__main__":
    for w in ["hola", "perro", "gracias", "familia", "decidir", "actividad"]:
        print(w, "->", word_to_viseme_sequence(w))
