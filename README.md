# Backgammon Classic Reinforcement Learning
Solving backgammon using classical reinforcement learning techniques

# PLAN

* Sander
    * :-)
* Sigurd
    * :-)
* Jørgen
    * :-)


Sander mot Random:
Win: 34
Lose: 11
Current winration: 34/45 = 0.75555

Selfnotes:
Graph som plotter absolutt sum av q table over tid

Prøve ut self play hvor det er to forskjellige agenter, og statene ikke blir "swappet". Som er motsatt av hvordan jeg har gjort det nå. Hvor det er én og samme agent som spiller mot seg selv, mens staten og observation flippes slik at den tror den alltid spiller for svart.

Kult å sammenligne disse mot hverandre!



Must explore and understand temporal difference learning and policy gradient learning. Find examples of implementations/possible algorithms to use.

What is Monte Carlo and what is Dynamic Programming and what is Markov Decision Policies


## Gym

We are using the https://github.com/dellalibera/gym-backgammon gym for training the reinforcement learning model

### Installation

If there are no pip environments - create one by issuing the following command
> virtualenv env

Then activate the virtual environment
> source env/bin/activate


Clone the following github repository, containing the gym
> git clone https://github.com/dellalibera/gym-backgammon.git

Change directory into the gym, and pip install the gym by issuing the following command
> cd gym-backgammon/ && pip install -e .

Or just run the install script by doing the following steps

1. First make the file executable by issuing the following command
    - >chmod +x install_environment.sh
2. If you do not have virtualenv installed, then install it for your distro. It can be installed in Ubuntu by issuing the following command
    - >sudo apt install python-virtualenv
3. Then run the script
    - >./install_environment.sh

## Interesting Links

[Temporal Difference Learning and TD-Gammon](https://cling.csd.uwo.ca/cs346a/extra/tdgammon.pdf)

[Deep reinforcement learning compared with Q-table learning applied to backgammon](https://www.kth.se/social/files/58865ec8f27654607fb6e9a4/PFinnman_MWinberg_dkand16.pdf)

[Gym Environment](https://github.com/dellalibera/gym-backgammon.git)



### Must remember to note things we've tried to do in order to solve the problem

Et eks er jo feks at vi laget et gym, men det er altfor stor action/state mapping

Kan vise det med følgende kode:
```
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

# Da kan vi initialisere et array med 0 på følgende måte:
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 22, 23
representation = np.zeros((2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 11, 11), np.float16)
```

Da får man følgende feilmelding:
*"Unable to allocate 1.53 EiB for an array with shape (2, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 11, 11) and data type float16"*

Og da mangler vi fortsatt å ta med terningene som en del av informasjonen til observation/representation

EiB: an exbibyte is one of the larger measurements on the binary scale

```
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

representation = np.zeros((2, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 11, 11), np.float16)
```

Da får man følgende feilmelding:                       0, 1,  2,  3,  4,  5,  6,  7,  8,  9,  0, 1, 12, 13
*"Unable to allocate 81.8 TiB for an array with shape (2, 12, 12, 12, 12, 12, 12, 12, 12, 12, 6, 6, 11, 11) and data type float16"*

Og TiB er jo da terabyte...

Og her har jeg fortsatt ikke tatt med terningkastet......


Derfor foreslår jeg at vi reduserer spillet enda mer...

Når jeg regner på det, er jo usikker på om det jeg gjør er korrekt, men så får jeg dette ihvertfall:

Da tenker jeg også at vi kan redusere antall lovlige brikker per plass til 2

Og at terningen enten da bare har 1, og 2 øyne, og kanskje må ta bort dobbeltkast

En annen mulighet er at hver spiller feks har 2 plasser hver som hjemmebane, også er det 3 i midten. Så 7 totalt.
Men at vi kanskje inkluderer Bar i denne versjonen feks

```
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
#                   1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 14, 15
q_table = np.zeros((2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 10, 10), np.float16)

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
print(q_table[1, 3, 0, 2, 0, 4, 0, 1, 1, 1, 0, 0, 1, 1, 2])

# Man kan da også sjekke hvor mange gig ram det tar ved å kjøre linja under her:
print((q_table.size * q_table.itemsize) / (1024**2))

# Resultatet er 15.4 elns GB RAM
```