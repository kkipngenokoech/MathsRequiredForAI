import random
class DeckCard:
    shapes = [ "Hearts", "Diamonds", "Clubs", "Spades" ]
    numbers = [ "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K" ]
    
    def __init__(self):
        self.cards = [Card(number, shape) for number in DeckCard.numbers for shape in DeckCard.shapes]
    
    def __str__(self):
        return ', '.join(str(card) for card in self.cards)
    def __repr__(self):
        return self.__str__()
    def shuffle(self):
        random.shuffle(self.cards)
    def draw(self):
        return self.cards.pop() if len(self.cards) > 0 else None
        
class Card:
    def __init__(self, number, shape):
        self.number = number
        self.shape = shape

    def __str__(self):
        return f"{self.number} {self.shape}"
    def __repr__(self):
        return self.__str__()
    
deck = DeckCard()
deck.shuffle()

'''
drawing 10 cards from the deck, without replacement
'''
drawn_cards = [deck.draw() for _ in range(10) if deck.cards] or print("No more cards to draw")
print(drawn_cards)

'''
probability of drawing a card with number 2
'''
number_2_cards = [card for card in deck.cards if card.number == "2"]
simulated_2_probability = len(number_2_cards) / len(deck.cards)
print(simulated_2_probability)
shape_hearts = [card for card in deck.cards if card.shape == "Hearts"]
simulated_heart_probability = len(shape_hearts) / len(deck.cards)
print(simulated_heart_probability)
suit_diamonds_3 = [card for card in deck.cards if card.shape == "Diamonds" and card.number == "3"]
simulated_diamonds_of_three_probability = len(suit_diamonds_3) / len(deck.cards)
print(simulated_diamonds_of_three_probability)

# THEORITICAL PROBABILITY
'''
probability of drawing a card with number 2
'''
numbr_2_cards_theoritical = 4
theory_2_probability = numbr_2_cards_theoritical / len(deck.cards)
print(theory_2_probability)

shape_of_heart_theoritical = 13
theory_heart_probability = shape_of_heart_theoritical / len(deck.cards)
print(theory_heart_probability)

suit_diamonds_3_theoritical = 1
theory_diamonds_of_three_probability = suit_diamonds_3_theoritical / len(deck.cards)
print(theory_diamonds_of_three_probability)

