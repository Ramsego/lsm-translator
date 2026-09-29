"""Real G2P (epitran, Spanish IPA) -> viseme mapping, replacing the hand-built letter table.
Each phoneme produces one (openness, roundedness) event.
openness: 0=closed, 0.5=mid, 1=open.  roundedness: 0=spread, 1=rounded.
Consonants with no strong independent visible lip shape are skipped (tongue/palate articulations);
the surrounding vowels dominate the visible signal, matching real mouthing behavior.
"""
import epitran
_epi = epitran.Epitran('spa-Latn')

VISEME = {
    # vowels
    'a': (1.0, 0.0), 'e': (0.6, 0.0), 'i': (0.2, 0.0), 'o': (0.6, 1.0), 'u': (0.2, 1.0),
    # glides
    'j': (0.15, 0.0),   # palatal glide (as in familia -> familja) -- closed, spread
    'w': (0.15, 1.0),   # labio-velar glide -- closed, rounded
    # bilabial closure
    'p': (0.05, 0.3), 'b': (0.05, 0.3), 'm': (0.05, 0.3),
    # labiodental (partial closure, not rounded)
    'f': (0.15, 0.0),
}
# everything else (t,d,k,g,s,n,l,r,ɾ,x,tʃ,ɲ,θ, etc.) -- no strong independent visible shape, skip

def word_to_viseme_sequence(word):
    ipa = _epi.transliterate(word.lower())
    seq = []
    for ch in ipa:
        if ch in VISEME:
            seq.append(VISEME[ch])
    return seq

if __name__ == "__main__":
    for w in ["hola", "perro", "gracias", "familia", "decidir", "actividad"]:
        print(w, "->", _epi.transliterate(w), "->", word_to_viseme_sequence(w))
