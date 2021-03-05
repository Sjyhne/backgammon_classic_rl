import numpy as np

"""
    Hvis vi gjør det samme som over, men på en litt annen måte får vi redusert problemet veldig, men det er fortsatt altfor stort
    1: Player turn
    2: First location hvor 0 -> 0 brikker der, 1 - 6 -> 1 - 6 hvite brikker der og 7 - 12 -> 1 - 6 svarte brikker
    3: ..
    10: Off location for hvit
    11: Off location for svart
    12: From location in action
    13: To location in action
"""

"""
    Meaning of array positions(?) in the array
    1: Which player turn it is: 0 -> White, 1 -> Black (min: 0, max: 1)

    2: First location on map for white (min: 0, max: 6)
    3: Second location on map for white (min: 0, max: 6)
    4: Third location on map for white (min: 0, max: 6)
    5: Fourth location on map for white (min: 0, max: 6)
    6: Fifth location on map for white (min: 0, max: 6)
    7: Sixth location on map for white (min: 0, max: 6)
    8: Seventh location on map for white (min: 0, max: 6)
    9: Eigth location on map for white (min: 0, max: 6)
    10: Ninth location on map for white (min: 0, max: 6)
    11: Off location on map for white (min: 0, max: 6)

    12: First location on map for black (min: 0, max: 6)
    13: ...
    21: Off location on map for black (min: 0, max: 6)

    22: Action from location
    23: Action to location
"""


"""
    (-1 - 0), (0 -> 1), (0 -> 2), (0 -> 3), (0 - 4), (0 - 5), (0 - 6), 7, 8, 9, 10
"""

# Da kan vi initialisere et array med 0 på følgende måte:
#                          1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 22, 23
# representation = np.zeros((2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 11, 11), np.float16)

#representation = np.zeros((2, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 9, 9), np.float16)

#0 = 1, 1 = 2, 2 = 3

# representation = np.zeros((54 * 10 ** 6))


#print((representation.size * representation.itemsize) / (1024 ** 2))

"""
    Tallene som går nedover er dimensjonene

    1: Player turn (min: 0, max: 1)
    2: First location hvor index 0 -> 0 brikker, index 1 - 2 -> 1 - 2 hvite brikker, index 3 - 4 -> 1 - 2 svarte brikker (min: 0, max: 4)
    3: ...
    8: Bar location hvit (min: 0, max: 4)
    9: Bar location svart (min: 0, max: 4)
    10: Off location hvit (min: 0, max: 4)
    11: Off location svart (min: 0, max: 4)
    12: First dice throw (min: 0, max: 1)
    13: Second dice throw (min: 0, max: 1)
    14: From Location Action (min: 0, max: 9
    15: To Location Action
"""

dice_one = [1, 1, 1, 1, 1, 2, 2, 2]
dice_two = [1, 1, 1, 2, 2, 2, 2, 2]
# (BAR) - 0, BAR - 1, BAR - 2, BAR - 3, BAR - 4, BAR - 5, BAR - 6, BAR - 7, BAR - OUT
#                   1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 14, 15
q_table = np.zeros((2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 9, 9), np.float16)

"""
    La oss da si at jeg har lyst på q-verdien til action fra 1 til 2
    Det finnes følgende brikker på brettet:

    Det er svart sin tur

    Terningkastet er 1 og 2

    1 svart på 1
    0 brikker 2
    2 hvite på 3
    0 brikker på 4
    2 svarte på 5
    0 brikker på 6
    1 hvit på bar
    1 svart på bar
    1 hvit er off
    0 svarte er off

    Så kan vi printe q_verdien til action slik:
"""
#              1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5
#print(q_table[1, 3, 0, 2, 0, 4, 0, 1, 1, 1, 0, 0, 1, 1, 2])

# Man kan da også sjekke hvor mange gig ram det tar ved å kjøre linja under her:
print((q_table.size * q_table.itemsize) / (1024**2))
